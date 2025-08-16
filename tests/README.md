# 測試指南

## video_io 模組測試套件

本專案提供兩種測試套件：

### 📦 Mock 測試 (`test_video_io.py`)
- 使用 mock 模擬 `av` 庫行為
- 快速執行，無外部依賴
- 專注於 API 層面的單元測試
- 適合開發期間的快速驗證

### 🎬 真實測試 (`test_video_io_real.py`)  
- 使用真實的影片檔案進行端到端測試
- 測試實際的編碼-解碼流程
- 可能因系統環境而有差異
- 提供更高的信心度

詳細的真實測試說明請參考 [`README_real_tests.md`](README_real_tests.md)

## 運行 video_io 模組測試

### 安裝測試依賴

```bash
pip install -e ".[test]"
```

### 運行測試

#### Mock 測試
```bash
# 運行 Mock 測試 (快速)
pytest tests/test_video_io.py -v

# 運行測試並顯示覆蓋率
pytest tests/test_video_io.py --cov=castle.utils.video_io --cov-report=html

# 運行特定測試類
pytest tests/test_video_io.py::TestVideoReader -v
```

#### 真實測試
```bash
# 運行真實測試 (較慢但更可靠)
pytest tests/test_video_io_real.py -v

# 顯示跳過的測試原因
pytest tests/test_video_io_real.py -v -rs

# 運行特定測試類
pytest tests/test_video_io_real.py::TestVideoWriterReal -v
```

#### 運行所有測試
```bash
# 運行兩種測試套件
pytest tests/test_video_io.py tests/test_video_io_real.py -v

# 只運行通過的測試 (跳過可能失敗的)
pytest tests/ -k "video_io" --tb=short
```

### 測試標記

測試使用以下標記分類：

- `unit`: 單元測試
- `integration`: 整合測試
- `slow`: 較慢的測試

```bash
# 只運行單元測試
pytest tests/test_video_io.py -m unit -v

# 跳過慢速測試
pytest tests/test_video_io.py -m "not slow" -v
```

### 測試覆蓋範圍

測試涵蓋以下功能：

#### VideoReader 類
- ✅ 初始化和錯誤處理
- ✅ 影格讀取和快取機制
- ✅ 批次讀取功能
- ✅ 迭代器功能
- ✅ Context manager 支援
- ✅ 魔術方法 (`__len__`, `__getitem__`)
- ✅ 影格數量計算（包括二分搜尋）

#### VideoWriter 類
- ✅ 初始化和配置
- ✅ 影格寫入功能
- ✅ 格式轉換（float/uint8）
- ✅ Context manager 支援
- ✅ 錯誤處理

#### VideoIO 靜態方法
- ✅ 載入影片功能
- ✅ 儲存影片功能
- ✅ 錯誤處理

#### SubtitleGenerator 類
- ✅ 字幕添加和驗證
- ✅ SRT 格式輸出
- ✅ WebVTT 格式輸出
- ✅ 時間格式化
- ✅ 工具方法

#### 整合測試
- ✅ 完整影片處理流程
- ✅ 字幕工作流程
- ✅ 效能測試

### Mock 策略

測試使用 Mock 物件模擬：
- `av` 庫的容器和影格物件
- 檔案系統操作
- 外部依賴

這確保測試：
- 快速執行
- 不依賴外部檔案
- 可重現
- 隔離測試環境

### 故障排除

#### 常見問題

1. **ModuleNotFoundError**: 確保已安裝 `av` 庫
   ```bash
   pip install av==12.1.0
   ```

2. **Import 錯誤**: 確保 PYTHONPATH 正確設定
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **權限錯誤**: 確保測試目錄有寫入權限

#### 除錯模式

```bash
# 在失敗時進入 pdb
pytest tests/test_video_io.py --pdb

# 顯示詳細輸出
pytest tests/test_video_io.py -s -v

# 運行特定失敗的測試
pytest tests/test_video_io.py::TestVideoReader::test_video_reader_init_success -vvv
```

### 貢獻測試

添加新測試時請遵循：

1. **命名規範**: `test_<功能描述>`
2. **文檔字串**: 清楚描述測試目的
3. **Mock 使用**: 適當使用 Mock 隔離依賴
4. **斷言**: 使用明確的斷言訊息
5. **清理**: 確保測試後清理臨時資源

### 效能基準

測試包含基本的效能驗證：
- 快取機制有效性
- 批次讀取效率
- 記憶體使用合理性

運行效能測試：
```bash
pytest tests/test_video_io.py::TestVideoIOPerformance -v
```
