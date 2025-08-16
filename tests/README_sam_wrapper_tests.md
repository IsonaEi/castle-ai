# SAM Wrapper 測試指南

## 概述

`test_sam_wrapper.py` 是 SAM (Segment Anything Model) Wrapper 的綜合測試套件，包含單元測試、整合測試和端到端的真實分割測試。

## 特色

### ✅ 真實性
- 自動下載 SAM 模型權重（約 375MB）
- 使用真實影片檔案進行分割測試
- 測試完整的影像分割流程

### 🎯 測試涵蓋範圍

#### SAMWrapper 單元測試 (6個)
- ✅ ModelSize 枚舉測試
- ✅ SegmentationResult 資料類別測試
- ✅ SAM 模型初始化（含自動下載）
- ✅ 影像格式轉換處理
- ✅ 點擊分割輸入驗證
- ✅ 快取清除功能

#### VideoIO 整合測試 (3個)
- ✅ 影片檔案存在檢查
- ✅ 影片幀提取功能
- ✅ 真實影片分割測試（關鍵測試）

#### 邊界情況測試 (2個)
- ✅ 設備回退機制（CUDA→CPU）
- ✅ 模型大小字串轉換

## 關鍵測試：真實影片分割

### 測試目標
使用 `notebooks/open_field_videos/oft_1min.mp4` 的第 0 幀進行分割：
- **點擊座標**: (650, 600)
- **標籤**: 1（前景點）
- **輸出位置**: 專案根目錄下的 `tmp/` 資料夾

### 輸出檔案
測試會產生以下檔案：
- `segmentation_visualization.png` - 完整可視化（原圖 + 遮罩 + 重疊圖）
- `results_combined.png` - 三合一組合圖
- `original_frame.png` - 原始影像（如果 PIL 可用）
- `segmentation_mask.png` - 分割遮罩（如果 PIL 可用）
- `mask.npy` - 原始遮罩數據

## 測試結果

### 快速測試（跳過模型下載）
```
pytest tests/test_sam_wrapper.py -m "not model_required" -v
```
**預期結果**: 7-8 passed, 3-4 deselected

### 完整測試（包含模型下載）
```
pytest tests/test_sam_wrapper.py -v
```
**預期結果**: 10-11 passed, 約需 2-5 分鐘（首次下載模型）

## 運行測試

### 安裝依賴
```bash
pip install -e ".[test]"
# 確保安裝 torch, torchvision, PIL 等依賴
```

### 基本運行
```bash
# 運行所有測試（包含模型下載）
pytest tests/test_sam_wrapper.py -v

# 運行不需要模型的測試
pytest tests/test_sam_wrapper.py -m "not model_required" -v

# 運行特定測試類
pytest tests/test_sam_wrapper.py::TestSAMWrapper -v

# 運行關鍵的分割測試
pytest tests/test_sam_wrapper.py::TestSAMWrapperWithVideoIO::test_real_video_segmentation_with_point_annotation -v -s
```

### 手動測試
```bash
# 直接運行測試腳本
python tests/test_sam_wrapper.py
```

### 除錯模式
```bash
# 顯示詳細輸出
pytest tests/test_sam_wrapper.py -v -s

# 在第一個失敗時停止
pytest tests/test_sam_wrapper.py -x

# 顯示跳過的測試原因
pytest tests/test_sam_wrapper.py -v -rs
```

## 測試標記

測試使用以下 pytest 標記分類：

- `@pytest.mark.unit`: 單元測試
- `@pytest.mark.integration`: 整合測試
- `@pytest.mark.model_required`: 需要 SAM 模型權重
- `@pytest.mark.slow`: 執行時間較長的測試

```bash
# 只運行單元測試
pytest tests/test_sam_wrapper.py -m unit -v

# 跳過需要模型的測試
pytest tests/test_sam_wrapper.py -m "not model_required" -v

# 跳過慢速測試
pytest tests/test_sam_wrapper.py -m "not slow" -v
```

## 模型下載機制

### 自動下載
- 首次運行時自動下載 SAM VIT-B 模型（約 375MB）
- 下載位置：`~/.cache/castle/models/sam_vit_b_01ec64.pth`
- 包含進度條顯示下載狀態

### 下載過程
```
正在下載 SAM 權重檔案：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
下載進度: |██████████████████████████████| 100%
已下載權重檔案至 ~/.cache/castle/models/sam_vit_b_01ec64.pth
```

## 輸出管理

### 輸出位置
- **專案 tmp 目錄**: `castle-ai/tmp/sam_test_*`
- **自動創建**: 測試自動創建唯一的子目錄
- **Git 忽略**: tmp 目錄已在 .gitignore 中排除

### 檔案結構
```
castle-ai/
├── tmp/
│   ├── sam_test_abc123/
│   │   ├── segmentation_visualization.png
│   │   ├── results_combined.png
│   │   ├── original_frame.png
│   │   ├── segmentation_mask.png
│   │   └── mask.npy
│   └── sam_manual_test_def456/
│       └── ...
```

## 依賴性處理

### 核心依賴
- `torch`: SAM 模型運行
- `numpy`: 陣列處理
- `matplotlib`: 可視化和影像保存
- `PIL (Pillow)`: 影像 I/O（可選）

### 依賴回退
- **無 PIL**: 使用 matplotlib 進行影像保存
- **無 CUDA**: 自動回退到 CPU 模式
- **無 cv2**: 使用純 numpy 實現（已移除 cv2 依賴）

## 錯誤處理和故障排除

### 常見問題

1. **模型下載失敗**
   ```
   錯誤: 網路連線問題
   解決: 檢查網路連線，或手動下載模型檔案
   ```

2. **影片檔案不存在**
   ```
   錯誤: 測試影片檔案不存在
   解決: 確保 notebooks/open_field_videos/oft_1min.mp4 存在
   ```

3. **記憶體不足**
   ```
   錯誤: CUDA out of memory
   解決: 測試會自動使用 CPU 模式
   ```

4. **NumPy 版本衝突**
   ```
   錯誤: numpy.core.multiarray failed to import
   解決: pip install "numpy<2.0"
   ```

### 調試技巧

```bash
# 查看詳細錯誤資訊
pytest tests/test_sam_wrapper.py::failing_test -vvv

# 保留輸出檔案檢查
pytest tests/test_sam_wrapper.py -v -s --tb=short

# 只運行成功的測試
pytest tests/test_sam_wrapper.py -k "not (model_required and slow)"
```

## 效能基準

### 執行時間
- **Mock 測試**: < 5 秒
- **整合測試**: 10-30 秒
- **完整測試（首次）**: 2-5 分鐘（包含下載）
- **完整測試（後續）**: 30-60 秒

### 資源使用
- **CPU 記憶體**: ~2-4 GB
- **儲存空間**: ~400 MB（模型權重）
- **網路頻寬**: ~375 MB（首次下載）

## 與其他測試的整合

### CI/CD 配置
```yaml
# 快速測試（不下載模型）
- name: Run SAM unit tests
  run: pytest tests/test_sam_wrapper.py -m "not model_required" -v

# 完整測試（在夜間或發布前）
- name: Run SAM integration tests
  run: pytest tests/test_sam_wrapper.py -v
```

### 本地開發
```bash
# 開發期間的快速驗證
pytest tests/test_sam_wrapper.py -m "not slow" -v

# 發布前的完整驗證
pytest tests/test_sam_wrapper.py -v
```

## 測試資料驗證

### 分割結果驗證
測試會驗證以下指標：
- ✅ 遮罩形狀正確 (H, W)
- ✅ 遮罩類型為 boolean
- ✅ 分割區域面積 > 0
- ✅ 點擊座標標註正確

### 輸出檔案驗證
- ✅ 可視化圖片生成成功
- ✅ 遮罩數據保存完整
- ✅ 檔案路徑和命名正確

## 最佳實踐

### 1. 測試隔離
- 每個測試使用獨立的輸出目錄
- 使用 fixtures 管理測試資源
- 自動清理臨時檔案

### 2. 錯誤處理
- 優雅處理網路錯誤
- 自動回退到替代方案
- 提供清楚的錯誤訊息

### 3. 效能優化
- 使用較小的模型（VIT-B）進行測試
- 快取模型權重避免重複下載
- 平行運行獨立測試

### 4. 可維護性
- 清楚的測試分類和標記
- 詳細的文檔和註解
- 模組化的測試設計

## 貢獻指南

### 添加新測試
1. **命名規範**: `test_<功能描述>`
2. **適當標記**: 使用正確的 pytest 標記
3. **文檔字串**: 清楚描述測試目的和預期結果
4. **資源管理**: 使用 fixtures 管理測試資源

### 修改現有測試
1. **保持向後相容**: 不破壞現有功能
2. **更新文檔**: 同步更新相關文檔
3. **驗證輸出**: 確保測試結果一致性

## 結論

SAM Wrapper 測試套件提供了全面的測試覆蓋，從基本的單元測試到端到端的真實分割測試。透過自動模型下載和智慧的依賴處理，確保測試在各種環境中都能可靠運行，為 SAM 功能的正確性提供了強有力的保證。
