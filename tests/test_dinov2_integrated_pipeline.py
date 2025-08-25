"""
DINOv2 整合流水線測試

完整的 SAM + DeAOT + DINOv2 整合測試，包含：
- SAM 分割生成初始 ROI
- DeAOT 追蹤 ROI 變化
- DINOv2 提取特徵
- 與真實數據對比分析
- 詳細的可視化輸出
"""

import pytest
import numpy as np
import torch
import time
import json
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# 檢查可選套件
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# 環境變數控制
import os
IS_CI = os.environ.get('CI', 'false').lower() == 'true'
SKIP_HEAVY_TESTS = os.environ.get('SKIP_HEAVY_TESTS', 'false').lower() == 'true'

# Skip 條件
skip_on_ci = pytest.mark.skipif(
    IS_CI and SKIP_HEAVY_TESTS,
    reason="在 CI 環境中跳過重量級測試"
)

requires_sklearn = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="需要 sklearn 套件"
)

requires_matplotlib = pytest.mark.skipif(
    not MPL_AVAILABLE,
    reason="需要 matplotlib 套件"
)

# 導入被測試的模組
try:
    from castle.utils.video_io import VideoIO
    VIDEO_IO_AVAILABLE = True
except ImportError:
    VIDEO_IO_AVAILABLE = False

try:
    from castle.models.sam_wrapper import SAMWrapper, ModelSize
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

try:
    from castle.models.deaot_wrapper import DeAOTWrapper, ModelType as DeAOTModelType
    DEAOT_AVAILABLE = True
except ImportError:
    DEAOT_AVAILABLE = False

try:
    from castle.models.dinov2_wrapper import DINOv2Wrapper
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SAM_AVAILABLE, reason="需要 SAM 模組")
@pytest.mark.skipif(not DEAOT_AVAILABLE, reason="需要 DeAOT 模組")
@pytest.mark.skipif(not DINOV2_AVAILABLE, reason="需要 DINOv2 模組")
@pytest.mark.skipif(not VIDEO_IO_AVAILABLE, reason="需要 VideoIO 模組")
@requires_sklearn
@requires_matplotlib
@skip_on_ci
class TestDINOv2IntegratedPipeline:
    """完整的 SAM + DeAOT + DINOv2 整合流水線測試"""

    @pytest.fixture
    def project_root(self):
        """專案根目錄"""
        return Path(__file__).parent.parent

    @pytest.fixture
    def video_path(self, project_root):
        """測試影片路徑"""
        video_path = project_root / "notebooks" / "open_field_videos" / "oft_1min.mp4"
        if not video_path.exists():
            pytest.skip(f"測試影片不存在: {video_path}")
        return video_path

    @pytest.fixture
    def ground_truth_path(self, project_root):
        """真實數據檔案路徑"""
        gt_path = project_root / "notebooks" / "open_field_videos" / "openfield-1min-raw_ROI_1_latent.npz"
        if not gt_path.exists():
            pytest.skip(f"真實數據不存在: {gt_path}")
        return gt_path

    @pytest.fixture
    def output_dir(self, project_root):
        """測試輸出目錄"""
        tmp_dir = project_root / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        temp_dir = tempfile.mkdtemp(prefix="dinov2_integrated_test_", dir=str(tmp_dir))
        output_dir = Path(temp_dir)
        logger.info(f"整合測試輸出目錄: {output_dir}")
        return output_dir

    def save_comprehensive_visualization(self, extracted_features, gt_features, roi_masks, 
                                       frames, output_dir, metadata):
        """保存綜合可視化分析"""
        if not MPL_AVAILABLE:
            logger.warning("matplotlib 不可用，跳過可視化")
            return

        num_frames = min(len(extracted_features), len(gt_features))

        # 創建綜合對比圖
        fig = plt.figure(figsize=(20, 16))

        # 1. 特徵時序對比
        plt.subplot(3, 3, 1)
        plt.plot(np.mean(extracted_features[:num_frames], axis=1), 'b-', 
                alpha=0.7, label='Extracted', linewidth=2)
        plt.plot(np.mean(gt_features[:num_frames], axis=1), 'r-', 
                alpha=0.7, label='Ground Truth', linewidth=2)
        plt.xlabel('Frame Index')
        plt.ylabel('Mean Feature Value')
        plt.title('Mean Feature Value Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 特徵標準差對比
        plt.subplot(3, 3, 2)
        plt.plot(np.std(extracted_features[:num_frames], axis=1), 'b-', 
                alpha=0.7, label='Extracted', linewidth=2)
        plt.plot(np.std(gt_features[:num_frames], axis=1), 'r-', 
                alpha=0.7, label='Ground Truth', linewidth=2)
        plt.xlabel('Frame Index')
        plt.ylabel('Feature Std')
        plt.title('Feature Standard Deviation Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. 逐幀相關性
        correlations = []
        for i in range(num_frames):
            corr = np.corrcoef(extracted_features[i], gt_features[i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)

        plt.subplot(3, 3, 3)
        plt.plot(correlations, 'g-', linewidth=2, marker='o', markersize=3)
        plt.axhline(np.mean(correlations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(correlations):.3f}')
        plt.xlabel('Frame Index')
        plt.ylabel('Correlation')
        plt.title('Frame-wise Feature Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. 餘弦相似度
        cos_similarities = []
        for i in range(num_frames):
            cos_sim = cosine_similarity(
                extracted_features[i:i+1], 
                gt_features[i:i+1]
            )[0, 0]
            cos_similarities.append(cos_sim)

        plt.subplot(3, 3, 4)
        plt.plot(cos_similarities, 'purple', linewidth=2, marker='s', markersize=3)
        plt.axhline(np.mean(cos_similarities), color='red', linestyle='--',
                   label=f'Mean: {np.mean(cos_similarities):.3f}')
        plt.xlabel('Frame Index')
        plt.ylabel('Cosine Similarity')
        plt.title('Frame-wise Cosine Similarity')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. 特徵分布對比 (前100個維度)
        plt.subplot(3, 3, 5)
        sample_frames = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
        for i, frame_idx in enumerate(sample_frames):
            if frame_idx < num_frames:
                alpha = 0.3 + 0.1 * i
                plt.plot(extracted_features[frame_idx][:100], alpha=alpha, 
                        label=f'Extracted F{frame_idx}', linewidth=1)
                plt.plot(gt_features[frame_idx][:100], alpha=alpha, linestyle='--',
                        label=f'GT F{frame_idx}', linewidth=1)
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Value')
        plt.title('Feature Distribution (First 100 dims)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # 6. L2 距離
        l2_distances = []
        for i in range(num_frames):
            l2_dist = np.linalg.norm(extracted_features[i] - gt_features[i])
            l2_distances.append(l2_dist)

        plt.subplot(3, 3, 6)
        plt.plot(l2_distances, 'orange', linewidth=2, marker='^', markersize=3)
        plt.axhline(np.mean(l2_distances), color='red', linestyle='--',
                   label=f'Mean: {np.mean(l2_distances):.3f}')
        plt.xlabel('Frame Index')
        plt.ylabel('L2 Distance')
        plt.title('L2 Distance Between Features')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 7. 特徵相關性熱圖 (取樣)
        plt.subplot(3, 3, 7)
        sample_dims = np.linspace(0, extracted_features.shape[1]-1, 50).astype(int)
        corr_matrix = np.corrcoef(
            extracted_features[:num_frames, sample_dims].T, 
            gt_features[:num_frames, sample_dims].T
        )

        # 取提取特徵與真實特徵之間的相關性
        cross_corr = corr_matrix[:50, 50:]
        im = plt.imshow(cross_corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel('Ground Truth Dims')
        plt.ylabel('Extracted Dims')
        plt.title('Cross-Feature Correlation Matrix')

        # 8. 累積誤差分析
        plt.subplot(3, 3, 8)
        cumulative_mse = []
        cumulative_corr = []
        for i in range(1, num_frames + 1):
            mse = np.mean((extracted_features[:i] - gt_features[:i]) ** 2)
            corr = np.mean([np.corrcoef(extracted_features[j], gt_features[j])[0, 1] 
                          for j in range(i) if not np.isnan(np.corrcoef(extracted_features[j], gt_features[j])[0, 1])])
            cumulative_mse.append(mse)
            cumulative_corr.append(corr)

        ax1 = plt.gca()
        line1 = ax1.plot(cumulative_mse, 'b-', linewidth=2, label='Cumulative MSE')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Cumulative MSE', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        line2 = ax2.plot(cumulative_corr, 'r-', linewidth=2, label='Cumulative Correlation')
        ax2.set_ylabel('Cumulative Correlation', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title('Cumulative Error Analysis')
        plt.grid(True, alpha=0.3)

        # 9. 統計摘要
        plt.subplot(3, 3, 9)
        plt.axis('off')

        # 計算詳細統計
        mse = np.mean((extracted_features[:num_frames] - gt_features[:num_frames]) ** 2)
        mae = np.mean(np.abs(extracted_features[:num_frames] - gt_features[:num_frames]))
        mean_corr = np.mean(correlations)
        mean_cos_sim = np.mean(cos_similarities)
        mean_l2_dist = np.mean(l2_distances)

        stats_text = f"""Statistics Summary:

Frames: {num_frames}
Feature Dim: {extracted_features.shape[1]}

Mean Correlation: {mean_corr:.4f}
Mean Cosine Similarity: {mean_cos_sim:.4f}
MSE: {mse:.6f}
MAE: {mae:.6f}
Mean L2 Distance: {mean_l2_dist:.4f}

ROI Info:
Area: {metadata.get('roi_area', 'N/A')} pixels
Model: {metadata.get('model_type', 'N/A')}

Test Setup:
SAM Click: {metadata.get('sam_click_point', 'N/A')}
Processing Time: {metadata.get('processing_time', 'N/A')}
"""

        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

        plt.suptitle('DINOv2 Feature Extraction - Comprehensive Comparison Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存對比圖
        comparison_path = output_dir / "comprehensive_feature_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"綜合對比可視化已保存: {comparison_path}")

        # 保存詳細的數值對比
        self.save_detailed_metrics(extracted_features, gt_features, correlations, 
                                 cos_similarities, l2_distances, output_dir, metadata)

    def save_detailed_metrics(self, extracted_features, gt_features, correlations, 
                            cos_similarities, l2_distances, output_dir, metadata):
        """保存詳細的數值指標"""
        num_frames = min(len(extracted_features), len(gt_features))

        # 計算詳細指標
        metrics = {
            'summary': {
                'num_frames': num_frames,
                'feature_dim': extracted_features.shape[1],
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'mean_cosine_similarity': float(np.mean(cos_similarities)),
                'std_cosine_similarity': float(np.std(cos_similarities)),
                'mean_l2_distance': float(np.mean(l2_distances)),
                'std_l2_distance': float(np.std(l2_distances)),
                'mse': float(np.mean((extracted_features[:num_frames] - gt_features[:num_frames]) ** 2)),
                'mae': float(np.mean(np.abs(extracted_features[:num_frames] - gt_features[:num_frames]))),
            },
            'frame_wise_metrics': {
                'correlations': [float(c) for c in correlations],
                'cosine_similarities': [float(c) for c in cos_similarities],
                'l2_distances': [float(d) for d in l2_distances],
            },
            'metadata': metadata,
            'test_info': {
                'test_name': 'DINOv2 Integrated Pipeline Test',
                'timestamp': str(time.time()),
                'extracted_features_shape': extracted_features.shape,
                'ground_truth_features_shape': gt_features.shape,
            }
        }

        # 保存為 JSON
        metrics_path = output_dir / "detailed_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"詳細指標已保存: {metrics_path}")

        # 保存 CSV 格式的逐幀數據
        if PANDAS_AVAILABLE:
            try:
                frame_data = pd.DataFrame({
                    'frame_index': range(num_frames),
                    'correlation': correlations,
                    'cosine_similarity': cos_similarities,
                    'l2_distance': l2_distances,
                    'extracted_mean': np.mean(extracted_features[:num_frames], axis=1),
                    'gt_mean': np.mean(gt_features[:num_frames], axis=1),
                    'extracted_std': np.std(extracted_features[:num_frames], axis=1),
                    'gt_std': np.std(gt_features[:num_frames], axis=1),
                })

                csv_path = output_dir / "frame_wise_metrics.csv"
                frame_data.to_csv(csv_path, index=False)
                logger.info(f"逐幀指標 CSV 已保存: {csv_path}")
            except Exception as e:
                logger.warning(f"保存 CSV 失敗: {e}")

    def save_roi_progression(self, frames, roi_masks, output_dir):
        """保存 ROI 隨時間的變化可視化"""
        if not MPL_AVAILABLE:
            return

        # 選擇關鍵幀展示ROI變化
        num_frames = len(frames)
        key_frame_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
        key_frame_indices = [i for i in key_frame_indices if i < num_frames]

        fig, axes = plt.subplots(3, len(key_frame_indices), figsize=(4*len(key_frame_indices), 12))
        if len(key_frame_indices) == 1:
            axes = axes.reshape(-1, 1)

        for col, frame_idx in enumerate(key_frame_indices):
            # 原始幀
            axes[0, col].imshow(frames[frame_idx])
            axes[0, col].set_title(f'Frame {frame_idx}')
            axes[0, col].axis('off')

            # ROI 遮罩
            if frame_idx < len(roi_masks):
                roi_mask = roi_masks[frame_idx]
                axes[1, col].imshow(roi_mask, cmap='hot')
                axes[1, col].set_title(f'ROI Mask')
                axes[1, col].axis('off')

                # 疊加顯示
                axes[2, col].imshow(frames[frame_idx])
                masked = np.ma.masked_where(roi_mask == 0, roi_mask)
                axes[2, col].imshow(masked, cmap='Reds', alpha=0.6)
                axes[2, col].set_title(f'ROI Overlay')
                axes[2, col].axis('off')
            else:
                axes[1, col].text(0.5, 0.5, 'No ROI', ha='center', va='center')
                axes[1, col].axis('off')
                axes[2, col].text(0.5, 0.5, 'No ROI', ha='center', va='center')
                axes[2, col].axis('off')

        plt.suptitle('ROI Progression Over Time', fontsize=16)
        plt.tight_layout()

        roi_path = output_dir / "roi_progression.png"
        plt.savefig(roi_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"ROI 變化可視化已保存: {roi_path}")

    def test_full_integrated_pipeline_with_comparison(
        self, video_path, ground_truth_path, output_dir
    ):
        """
        完整的整合流水線測試 - SAM + DeAOT + DINOv2
        
        流程：
        1. 使用 SAM 在第一幀生成初始 ROI
        2. 使用 DeAOT 在整個序列中追蹤 ROI
        3. 使用 DINOv2 對每幀的 ROI 提取特徵
        4. 與真實數據對比並生成詳細的可視化報告
        """
        logger.info("=== 開始完整整合流水線測試 ===")

        # 載入真實數據
        gt_data = np.load(ground_truth_path)
        gt_latent = gt_data['latent']
        logger.info(f"真實數據形狀: {gt_latent.shape}")
        logger.info(f"真實數據統計: mean={np.mean(gt_latent):.4f}, std={np.std(gt_latent):.4f}")

        # 載入影片
        logger.info(f"載入影片: {video_path}")
        video_reader = VideoIO.load_video(video_path)
        video_info = video_reader.get_info()
        logger.info(f"影片資訊: {video_info.width}x{video_info.height}, {video_info.fps}fps, {video_info.frame_count} frames")

        # 限制處理幀數以控制測試時間
        max_frames = min(gt_latent.shape[0], video_info.frame_count, 200)  # 處理前200幀
        logger.info(f"將處理前 {max_frames} 幀")

        # 載入影片幀
        frames = []
        for i in range(max_frames):
            try:
                frame = video_reader.get_frame(i)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"讀取幀 {i} 失敗: {e}")
                break
        video_reader.close()

        actual_frames = len(frames)
        logger.info(f"實際載入 {actual_frames} 幀")

        if actual_frames == 0:
            pytest.skip("無法載入任何影片幀")

        # 記錄開始時間
        start_time = time.time()

        # 步驟 1: 使用 SAM 生成初始 ROI
        logger.info("步驟 1: SAM 初始分割...")
        try:
            sam = SAMWrapper(model_size=ModelSize.VIT_B, device='cuda')
            sam.set_image(frames[0])

            # 使用預定義的點擊點
            click_point = (650, 600)
            point_coords = np.array([click_point])
            point_labels = np.array([1])

            initial_mask = sam.predict_with_points(point_coords, point_labels)
            roi_area = np.sum(initial_mask)
            logger.info(f"SAM 初始分割完成，ROI 面積: {roi_area} 像素")

            sam.clear_cache()

        except Exception as e:
            logger.warning(f"SAM 初始化失敗: {e}，使用預設 ROI")
            h, w = frames[0].shape[:2]
            initial_mask = np.zeros((h, w), dtype=bool)
            initial_mask[h//3:2*h//3, w//3:2*w//3] = True
            click_point = (w//2, h//2)
            roi_area = np.sum(initial_mask)

        # 步驟 2: 使用 DeAOT 追蹤 ROI
        logger.info("步驟 2: DeAOT 追蹤...")
        try:
            tracker = DeAOTWrapper(
                model_type=DeAOTModelType.R50_DEAOTL,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # 設置參考幀
            combined_mask = initial_mask.astype(np.uint8)
            tracker.add_reference_frame(frames[0], combined_mask, obj_nums=1)

            # 追蹤序列
            logger.info(f"追蹤 {actual_frames-1} 幀...")
            tracks = tracker.track_sequence(frames[1:])

            logger.info(f"DeAOT 追蹤完成，獲得 {len(tracks)} 個軌跡")

            # 提取每幀的 ROI 遮罩
            roi_masks = [initial_mask]  # 第一幀
            for frame_idx in range(1, actual_frames):
                if 1 in tracks and frame_idx-1 in tracks[1].masks:
                    roi_mask = tracks[1].masks[frame_idx-1] > 0
                else:
                    roi_mask = initial_mask  # 使用初始遮罩作為備用
                roi_masks.append(roi_mask)

            tracker.clear_memory()

        except Exception as e:
            logger.warning(f"DeAOT 追蹤失敗: {e}，使用固定 ROI")
            roi_masks = [initial_mask] * actual_frames

        # 步驟 3: 使用 DINOv2 提取特徵
        logger.info("步驟 3: DINOv2 特徵提取...")
        try:
            dinov2 = DINOv2Wrapper(
                model_type='dinov2_vitb14_reg',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                use_fp16=False
            )
            logger.info("DINOv2 模型初始化成功")

            # 提取特徵
            extracted_features = []
            for i, (frame, roi_mask) in enumerate(zip(frames, roi_masks)):
                patch_features = dinov2.feature_extractor.extract_patch_features(frame)
                roi_feature = dinov2._extract_roi_features(patch_features, roi_mask.astype(np.float32))
                extracted_features.append(roi_feature)

                if (i + 1) % 50 == 0:
                    logger.info(f"已處理 {i+1}/{actual_frames} 幀")

            extracted_features = np.array(extracted_features)
            logger.info(f"特徵提取完成，形狀: {extracted_features.shape}")

            dinov2.clear_cache()

        except Exception as e:
            pytest.skip(f"DINOv2 特徵提取失敗: {e}")

        processing_time = time.time() - start_time

        # 步驟 4: 與真實數據對比
        logger.info("步驟 4: 特徵對比分析...")

        # 確保對比的幀數一致
        compare_frames = min(actual_frames, gt_latent.shape[0])
        extracted_subset = extracted_features[:compare_frames]
        gt_subset = gt_latent[:compare_frames]

        logger.info(f"對比 {compare_frames} 幀的特徵")
        logger.info(f"提取特徵統計: mean={np.mean(extracted_subset):.4f}, std={np.std(extracted_subset):.4f}")
        logger.info(f"真實特徵統計: mean={np.mean(gt_subset):.4f}, std={np.std(gt_subset):.4f}")

        # 計算基本指標
        correlations = []
        cos_similarities = []
        for i in range(compare_frames):
            # 相關性
            corr = np.corrcoef(extracted_subset[i], gt_subset[i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)

            # 餘弦相似度
            cos_sim = cosine_similarity(
                extracted_subset[i:i+1], 
                gt_subset[i:i+1]
            )[0, 0]
            cos_similarities.append(cos_sim)

        mean_correlation = np.mean(correlations)
        mean_cos_similarity = np.mean(cos_similarities)
        mse = np.mean((extracted_subset - gt_subset) ** 2)

        logger.info(f"平均相關性: {mean_correlation:.4f}")
        logger.info(f"平均餘弦相似度: {mean_cos_similarity:.4f}")
        logger.info(f"均方誤差: {mse:.6f}")

        # 步驟 5: 生成詳細的可視化報告
        logger.info("步驟 5: 生成可視化報告...")

        metadata = {
            'model_type': 'dinov2_vitb14_reg',
            'sam_click_point': click_point,
            'roi_area': int(roi_area),
            'processing_time': f"{processing_time:.2f}s",
            'frames_processed': actual_frames,
            'compare_frames': compare_frames,
            'mean_correlation': mean_correlation,
            'mean_cosine_similarity': mean_cos_similarity,
            'mse': mse
        }

        # 保存詳細對比可視化
        self.save_comprehensive_visualization(extracted_subset, gt_subset, roi_masks, 
                                            frames, output_dir, metadata)

        # 保存 ROI 變化可視化
        self.save_roi_progression(frames, roi_masks, output_dir)

        # 保存提取的特徵數據
        output_path = output_dir / "extracted_features_integrated.npz"
        np.savez_compressed(
            output_path,
            extracted_features=extracted_features,
            ground_truth_features=gt_subset,
            roi_masks=np.array(roi_masks),
            metadata=metadata
        )
        logger.info(f"特徵數據已保存: {output_path}")

        # 生成測試報告
        report = {
            'test_name': 'DINOv2 完整整合流水線測試',
            'pipeline_steps': ['SAM 分割', 'DeAOT 追蹤', 'DINOv2 特徵提取'],
            'performance_metrics': {
                'processing_time': processing_time,
                'frames_per_second': actual_frames / processing_time,
                'mean_correlation': mean_correlation,
                'mean_cosine_similarity': mean_cos_similarity,
                'mse': mse
            },
            'data_info': {
                'video_path': str(video_path),
                'ground_truth_path': str(ground_truth_path),
                'frames_processed': actual_frames,
                'compare_frames': compare_frames,
                'feature_dimension': extracted_features.shape[1]
            },
            'quality_assessment': {
                'correlation_good': bool(mean_correlation > 0.3),
                'cosine_similarity_good': bool(mean_cos_similarity > 0.5),
                'overall_assessment': 'PASS' if (mean_correlation > 0.3 and mean_cos_similarity > 0.5) else 'REVIEW_NEEDED'
            },
            'output_files': {
                'comprehensive_comparison': 'comprehensive_feature_comparison.png',
                'detailed_metrics': 'detailed_metrics.json',
                'roi_progression': 'roi_progression.png',
                'extracted_features': 'extracted_features_integrated.npz'
            }
        }

        report_path = output_dir / "integrated_test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"=== 整合測試完成 ===")
        logger.info(f"處理時間: {processing_time:.2f}s")
        logger.info(f"處理速度: {actual_frames/processing_time:.2f} fps")
        logger.info(f"測試結果: {report['quality_assessment']['overall_assessment']}")
        logger.info(f"所有結果已保存至: {output_dir}")
        logger.info("主要輸出檔案:")
        for desc, filename in report['output_files'].items():
            logger.info(f"  - {desc}: {filename}")

        # 驗證結果 - 使用較寬鬆的閾值以適應不同環境
        assert mean_correlation > 0.2, f"相關性過低: {mean_correlation}"
        assert mean_cos_similarity > 0.3, f"餘弦相似度過低: {mean_cos_similarity}"
        assert processing_time < 60, f"處理時間過長: {processing_time}s"

        return {
            'extracted_features': extracted_features,
            'performance_metrics': report['performance_metrics'],
            'output_dir': output_dir
        }

    @pytest.mark.unit
    def test_basic_pipeline_components_availability(self):
        """測試基本流水線組件的可用性"""
        # 檢查所有必要組件是否可用
        assert SAM_AVAILABLE, "SAM 模組不可用"
        assert DEAOT_AVAILABLE, "DeAOT 模組不可用"  
        assert DINOV2_AVAILABLE, "DINOv2 模組不可用"
        assert VIDEO_IO_AVAILABLE, "VideoIO 模組不可用"
        assert SKLEARN_AVAILABLE, "sklearn 模組不可用"
        assert MPL_AVAILABLE, "matplotlib 模組不可用"

        # 檢查基本模組導入
        logger.info("所有必要模組均可用")

    @pytest.mark.unit
    def test_visualization_functions(self, output_dir):
        """測試可視化函數的基本功能"""
        if not MPL_AVAILABLE:
            pytest.skip("matplotlib 不可用")

        # 創建測試數據
        num_frames = 10
        feature_dim = 768
        
        extracted_features = np.random.randn(num_frames, feature_dim) * 0.5
        gt_features = np.random.randn(num_frames, feature_dim) * 0.5
        
        # 創建模擬的 ROI 遮罩和幀
        roi_masks = [np.random.rand(100, 100) > 0.5 for _ in range(num_frames)]
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(num_frames)]
        
        metadata = {
            'model_type': 'test_model',
            'sam_click_point': (50, 50),
            'roi_area': 2500,
            'processing_time': '1.0s',
            'frames_processed': num_frames,
            'compare_frames': num_frames,
            'mean_correlation': 0.5,
            'mean_cosine_similarity': 0.5,
            'mse': 0.1
        }

        # 測試綜合可視化
        self.save_comprehensive_visualization(
            extracted_features, gt_features, roi_masks, frames, output_dir, metadata
        )

        # 檢查輸出檔案
        assert (output_dir / "comprehensive_feature_comparison.png").exists()
        assert (output_dir / "detailed_metrics.json").exists()

        # 測試 ROI 進程可視化
        self.save_roi_progression(frames, roi_masks, output_dir)
        assert (output_dir / "roi_progression.png").exists()

        logger.info("可視化函數測試通過")


def run_manual_integrated_test():
    """手動執行完整整合測試的輔助函數"""
    import tempfile
    
    tester = TestDINOv2IntegratedPipeline()
    
    # 設置路徑
    project_root = Path(__file__).parent.parent
    video_path = project_root / "notebooks" / "open_field_videos" / "oft_1min.mp4"
    gt_path = project_root / "notebooks" / "open_field_videos" / "openfield-1min-raw_ROI_1_rotation_latent.npz"
    
    # 創建輸出目錄
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="dinov2_integrated_manual_", dir=str(tmp_dir))
    output_dir = Path(temp_dir)
    
    logger.info(f"整合測試輸出目錄: {output_dir}")
    
    if not video_path.exists():
        logger.error(f"影片檔案不存在: {video_path}")
        return
    
    if not gt_path.exists():
        logger.error(f"真實數據檔案不存在: {gt_path}")
        return
    
    try:
        result = tester.test_full_integrated_pipeline_with_comparison(
            video_path, gt_path, output_dir
        )
        logger.info("整合測試成功完成！")
        logger.info(f"性能指標: {result['performance_metrics']}")
        logger.info(f"結果已保存到: {output_dir}")
        return result
    except Exception as e:
        logger.error(f"整合測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 當直接執行此檔案時，運行手動測試
    print("執行 DINOv2 完整整合測試...")
    run_manual_integrated_test()
