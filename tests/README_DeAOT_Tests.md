# DeAOT Wrapper 測試說明

## 概述

這個測試套件為 `DeAOTWrapper` 提供全面的驗證，包括基本功能測試、真實影片追蹤測試，以及質心軌跡分析。

## 測試文件

### `test_deaot_wrapper.py`
主要測試文件，包含：

#### 基本功能測試 (`TestDeAOTWrapper`)
- **模型類型枚舉測試**: 驗證 `ModelType` 枚舉的正確性
- **資料類別測試**: 測試 `TrackingResult` 和 `ObjectTrack` 資料結構
- **初始化測試**: 驗證 DeAOT 模型的正確初始化
- **輸入驗證測試**: 檢查無效輸入的處理

#### 整合測試 (`TestDeAOTWrapperWithVideoIO`)
- **影片讀取測試**: 驗證與 `video_io` 模組的整合
- **真實影片追蹤測試**: 使用實際影片進行多物件追蹤
- **質心軌跡計算**: 計算並保存物件的移動軌跡
- **結果可視化**: 生成追蹤結果的可視化圖表

#### 邊界情況測試 (`TestDeAOTWrapperEdgeCases`)
- **設備回退測試**: 驗證設備不可用時的回退機制
- **字符串轉換測試**: 測試模型類型的字符串轉換

### `run_deaot_test.py`
簡化的手動測試運行腳本，可以快速執行真實影片測試。

## 運行測試

### 1. 基本單元測試（不需要模型）
```bash
# 運行所有基本測試
pytest tests/test_deaot_wrapper.py::TestDeAOTWrapper -v

# 運行特定測試
pytest tests/test_deaot_wrapper.py::TestDeAOTWrapper::test_model_type_enum -v
```

### 2. 整合測試（需要測試影片）
```bash
# 運行影片相關測試
pytest tests/test_deaot_wrapper.py::TestDeAOTWrapperWithVideoIO::test_video_frame_extraction -v -m integration
```

### 3. 完整測試（需要下載模型）
```bash
# 運行需要實際模型的測試（會自動下載模型權重）
pytest tests/test_deaot_wrapper.py -v -m "model_required and slow"
```

### 4. 手動測試
```bash
# 使用簡化腳本
python tests/run_deaot_test.py

# 或直接運行測試文件
python tests/test_deaot_wrapper.py
```

## 測試標記

- `@pytest.mark.integration`: 整合測試，需要外部資源
- `@pytest.mark.model_required`: 需要下載實際模型權重
- `@pytest.mark.slow`: 執行時間較長的測試

## 輸出文件

測試會在 `tmp/` 目錄下生成以下文件：

### 軌跡數據
- `centroid_traces.json`: JSON 格式的質心軌跡數據
- `all_traces_combined.csv`: 統合的 CSV 軌跡數據
- `object_{id}_trace.csv`: 個別物件的軌跡數據

### 可視化結果
- `centroid_traces.png`: 物件質心軌跡圖
- `tracking_analysis.png`: 時間序列分析圖
- `tracking_frame_{id:03d}.png`: 關鍵幀的追蹤結果

### 統計數據
- `tracking_statistics.json`: 追蹤統計信息

## 依賴要求

### 必需依賴
- `numpy`
- `matplotlib`
- `pytest`
- `torch`
- `castle.models.deaot_wrapper`
- `castle.utils.video_io`

### 可選依賴
- `pandas`: 用於 CSV 數據處理（如不可用會自動回退到手動生成）
- `PIL`: 用於影像處理（如不可用會使用 matplotlib）
- `cv2`: 用於影像標注

## 測試影片要求

測試使用的影片路徑：
```
notebooks/open_field_videos/oft_1min.mp4
```

確保影片文件存在，或修改測試中的路徑配置。

## 質心軌跡分析

測試會自動計算每個追蹤物件的：

1. **質心座標**: 每幀中物件的中心位置 (x, y)
2. **面積變化**: 物件遮罩的像素面積
3. **移動距離**: 總移動距離和幀間移動速度
4. **軌跡可視化**: 2D 軌跡圖和時間序列分析

## 故障排除

### 常見問題

1. **模型下載失敗**
   - 檢查網絡連接
   - 確認 Google Drive 訪問權限

2. **CUDA 相關錯誤**
   - 安裝 CUDA 兼容的 PyTorch 版本
   - 或強制使用 CPU：在測試中設置 `device='cpu'`

3. **影片讀取失敗**
   - 確認測試影片存在
   - 檢查 video_io 模組是否正常工作

4. **依賴缺失**
   - 安裝缺失的套件：`pip install pandas pillow opencv-python`

### 調試模式

啟用詳細日誌：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 擴展測試

要添加新的測試用例：

1. 在適當的測試類中添加新方法
2. 使用適當的測試標記
3. 遵循現有的命名慣例
4. 添加詳細的文檔字符串

## 性能注意事項

- 模型下載：首次運行會下載約 300MB 的模型權重
- 記憶使用：追蹤長影片序列會消耗較多記憶
- GPU 加速：強烈建議使用 CUDA 進行測試以獲得更好的性能


