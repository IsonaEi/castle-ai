# DeAOT 視覺化測試指南

## 概述

已成功將視覺化測試功能融合到 `test_deaot_wrapper.py` 中。這些測試可以生成影片和圖片來可視化 DeAOT 追蹤器的性能，非常適合用於：

- 🎯 **演示目的**: 展示追蹤器的實際效果
- 🔍 **性能分析**: 可視化追蹤品質和軌跡
- 🐛 **調試問題**: 手動檢查追蹤失敗的原因
- 📊 **結果驗證**: 生成詳細的追蹤指標報告

## 新增的測試類別

### `TestDeAOTWrapperVisualization`

包含兩個主要測試方法：

#### 1. `test_synthetic_video_tracking`
- 生成合成測試影片（包含移動的圓形、矩形、三角形）
- 執行 DeAOT 多物件追蹤
- 生成可視化結果

#### 2. `test_real_video_tracking_with_sam`
- 使用真實影片進行測試
- 利用 SAM 模型生成第一幀的分割遮罩
- 用 DeAOT 進行後續幀追蹤
- 需要 `oft_1min.mp4` 測試影片

## 運行方式

### 1. 運行所有視覺化測試

```bash
# 運行視覺化測試類別
python -m pytest tests/test_deaot_wrapper.py::TestDeAOTWrapperVisualization -v

# 強制運行（忽略 CI 跳過）
SKIP_HEAVY_TESTS=false CI=false python -m pytest tests/test_deaot_wrapper.py::TestDeAOTWrapperVisualization -v
```

### 2. 運行特定測試

```bash
# 只運行合成影片測試
python -m pytest tests/test_deaot_wrapper.py::TestDeAOTWrapperVisualization::test_synthetic_video_tracking -v -s

# 只運行真實影片測試 (需要 SAM 和 VideoIO)
python -m pytest tests/test_deaot_wrapper.py::TestDeAOTWrapperVisualization::test_real_video_tracking_with_sam -v -s
```

### 3. 手動執行視覺化測試

```bash
# 直接執行視覺化測試腳本
python tests/test_deaot_wrapper.py viz
```

## 依賴需求

視覺化測試需要以下依賴：

- ✅ **AOT_AVAILABLE**: DeAOT 模型和依賴
- ✅ **MPL_AVAILABLE**: Matplotlib (用於圖表生成)
- ✅ **VIDEO_IO_AVAILABLE**: Castle VideoIO 模組 (用於影片處理)
- ✅ **SAM_AVAILABLE**: SAM 模型 (僅 `test_real_video_tracking_with_sam` 需要)

## 輸出結果

測試會在 `tmp/` 目錄下創建輸出檔案：

### 合成影片測試輸出：
```
tmp/deaot_test_*/
├── synthetic_original.mp4              # 原始合成影片
├── synthetic_first_frame_annotation.png # 第一幀標註
├── synthetic_tracking.mp4              # 追蹤結果影片
├── synthetic_tracking_key_frames.png   # 關鍵幀對比
├── synthetic_tracking_trajectories.png # 物件軌跡圖
└── synthetic_tracking_metrics.json     # 追蹤統計指標
```

### 真實影片測試輸出：
```
tmp/deaot_test_*/
├── real_video_original_clip.mp4         # 原始影片片段
├── real_video_sam_segmentation.png     # SAM 分割結果
├── real_video_tracking.mp4             # 追蹤結果影片
├── real_video_tracking_key_frames.png  # 關鍵幀對比
├── real_video_tracking_trajectories.png # 物件軌跡圖
└── real_video_tracking_metrics.json    # 追蹤統計指標
```

## 測試標記

這些測試使用以下 pytest 標記：

- `@pytest.mark.integration`: 整合測試
- `@pytest.mark.slow`: 耗時測試
- `@pytest.mark.model_required`: 需要模型檔案
- `@skip_on_ci`: 在 CI 環境中跳過
- `@pytest.mark.skipif`: 根據依賴可用性決定是否跳過

## 使用範例

### 快速驗證 DeAOT 功能
```bash
# 運行合成影片測試來快速驗證
SKIP_HEAVY_TESTS=false python -m pytest tests/test_deaot_wrapper.py::TestDeAOTWrapperVisualization::test_synthetic_video_tracking -v -s
```

### 完整的端到端測試
```bash
# 運行所有視覺化測試（需要完整環境）
SKIP_HEAVY_TESTS=false CI=false python -m pytest tests/test_deaot_wrapper.py::TestDeAOTWrapperVisualization -v -s
```

## 故障排除

### 常見問題：

1. **測試被跳過**：
   - 檢查必要的模型檔案是否存在
   - 確認所有依賴套件已安裝
   - 使用 `SKIP_HEAVY_TESTS=false CI=false` 強制運行

2. **記憶體不足**：
   - 視覺化測試可能消耗較多記憶體
   - 考慮在有足夠 RAM 的機器上運行

3. **影片不存在**：
   - 真實影片測試需要 `notebooks/open_field_videos/oft_1min.mp4`
   - 如果沒有該檔案，測試會自動跳過

## 技術細節

### 合成影片生成
- 生成包含 3 個移動物件的 30 幀影片
- 圓形：順時針圓周運動
- 矩形：左右正弦波移動
- 三角形：對角線移動

### 可視化功能
- 使用 matplotlib 生成文字標註
- 支援多物件顏色編碼
- 自動生成軌跡圖和統計報告
- 關鍵幀比較功能

### 追蹤指標
- 置信度統計（最小值、最大值、平均值）
- 物件區域統計
- 軌跡長度和完整性
- 質心位置追蹤

---

**注意**: 這些視覺化測試主要用於開發和演示目的。在 CI/CD 流程中，建議使用更輕量級的單元測試來確保程式碼品質。
