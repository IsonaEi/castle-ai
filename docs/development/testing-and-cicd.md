# Testing and CI/CD Guide

## 概述

Castle-AI 專案包含完整的測試和持續整合/持續部署 (CI/CD) 系統，特別針對 ROI Manager 模組進行了深度測試覆蓋。

## 🧪 測試架構

### 測試類型

1. **單元測試** (Unit Tests)
   - 測試個別函數和方法
   - 快速執行，適合開發過程中頻繁運行
   - 標記：`@pytest.mark.unit`

2. **整合測試** (Integration Tests)
   - 測試多個組件間的互動
   - 驗證 HDF5 檔案結構和資料完整性
   - 標記：`@pytest.mark.integration`

3. **性能測試** (Performance Tests)
   - 測試大量資料處理的性能
   - 記憶體使用測試
   - 標記：`@pytest.mark.slow`

### 測試覆蓋範圍

ROI Manager 測試涵蓋：
- ✅ 基本 CRUD 操作
- ✅ Meta 資訊管理
- ✅ Kinematic 資料追蹤
- ✅ 錯誤處理和邊界條件
- ✅ 檔案結構完整性
- ✅ 向後相容性
- ✅ 性能和記憶體使用

## 🚀 本地開發測試

### 快速開始

```bash
# 安裝開發依賴
make install-dev

# 運行所有 ROI Manager 測試
make test-roi

# 快速測試（跳過慢測試）
make test-quick

# 生成覆蓋率報告
make test-coverage
```

### 進階測試命令

```bash
# 只運行單元測試
python scripts/test_roi_manager.py --unit --verbose

# 只運行整合測試
python scripts/test_roi_manager.py --integration --verbose

# 運行性能測試
python scripts/test_roi_manager.py --performance --verbose

# 生成詳細覆蓋率報告
python scripts/test_roi_manager.py --coverage --verbose
```

### 使用 pytest 直接運行

```bash
# 運行所有 ROI Manager 測試
pytest tests/test_roi_manager.py -v

# 運行特定測試類型
pytest tests/test_roi_manager.py -m unit -v
pytest tests/test_roi_manager.py -m integration -v
pytest tests/test_roi_manager.py -m slow -v

# 生成覆蓋率報告
pytest tests/test_roi_manager.py --cov=castle.utils.roi_manager --cov-report=html
```

## 🔄 GitHub Actions CI/CD

### 工作流程

專案包含兩個主要的 CI/CD 工作流程：

#### 1. 主要 CI 工作流程 (`.github/workflows/ci.yml`)

**觸發條件:**
- 推送到 `main` 或 `develop` 分支
- Pull Request 到 `main` 或 `develop` 分支

**執行內容:**
- 多環境測試 (Ubuntu, macOS, Windows)
- 多 Python 版本測試 (3.8, 3.9, 3.10, 3.11)
- 代碼風格檢查 (flake8)
- 測試覆蓋率報告
- 整合測試和性能測試

#### 2. ROI Manager 專門工作流程 (`.github/workflows/test-roi-manager.yml`)

**觸發條件:**
- ROI Manager 相關檔案變更時
- 測試檔案變更時

**執行內容:**
- 針對性的 ROI Manager 測試
- 記憶體使用驗證
- HDF5 檔案結構驗證
- 覆蓋率要求：最低 80%

### CI 狀態徽章

在專案 README 中添加以下徽章來顯示 CI 狀態：

```markdown
[![CI](https://github.com/your-username/castle-ai/workflows/CI/badge.svg)](https://github.com/your-username/castle-ai/actions)
[![ROI Manager Tests](https://github.com/your-username/castle-ai/workflows/Test%20ROI%20Manager/badge.svg)](https://github.com/your-username/castle-ai/actions)
[![codecov](https://codecov.io/gh/your-username/castle-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/castle-ai)
```

## 📊 覆蓋率報告

### 本地覆蓋率

執行覆蓋率測試後：

```bash
# 在瀏覽器中查看 HTML 報告
open htmlcov/index.html

# 或查看終端報告
cat coverage.txt
```

### CI 覆蓋率

- 覆蓋率報告自動上傳到 Codecov
- 最低覆蓋率要求：80%
- PR 會自動顯示覆蓋率變化

## 🛠️ 開發工具

### Pre-commit Hooks

設定 pre-commit hooks 來確保代碼品質：

```bash
# 安裝 pre-commit hooks
make install-hooks

# 手動運行所有檢查
pre-commit run --all-files
```

### 代碼品質工具

```bash
# 代碼檢查
make lint

# 代碼格式化
make format

# 提交前檢查
make pre-commit
```

## 📝 編寫新測試

### 測試結構

```python
import pytest
from castle.utils.roi_manager import ROIManager

class TestNewFeature:
    @pytest.mark.unit
    def test_basic_functionality(self, temp_h5_file):
        """測試基本功能"""
        roi_manager = ROIManager(temp_h5_file)
        # 測試邏輯
        
    @pytest.mark.integration
    def test_integration_scenario(self, temp_h5_file, sample_mask):
        """測試整合場景"""
        # 整合測試邏輯
        
    @pytest.mark.slow
    def test_performance(self, temp_h5_file):
        """測試性能"""
        # 性能測試邏輯
```

### 測試最佳實踐

1. **使用 Fixtures**: 利用 `temp_h5_file` 和 `sample_mask` fixtures
2. **適當標記**: 使用 `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
3. **清理資源**: 確保測試後清理臨時檔案
4. **斷言清晰**: 使用清楚的錯誤訊息
5. **文檔化**: 為測試方法添加清楚的 docstring

### 新增測試檔案

如果需要新增其他模組的測試：

1. 在 `tests/` 目錄下創建 `test_module_name.py`
2. 遵循相同的測試結構和標記慣例
3. 更新 CI 配置以包含新測試
4. 添加適當的覆蓋率目標

## 🚨 疑難排解

### 常見問題

1. **Import 錯誤**
   ```bash
   # 確保安裝在開發模式
   pip install -e .
   ```

2. **HDF5 相關錯誤**
   ```bash
   # 確保安裝 h5py
   pip install h5py
   ```

3. **權限錯誤**
   ```bash
   # 確保測試目錄可寫
   chmod 755 tests/
   ```

### CI 失敗處理

1. **檢查日誌**: 查看 GitHub Actions 的詳細日誌
2. **本地重現**: 使用相同的 Python 版本和依賴在本地重現
3. **覆蓋率不足**: 添加更多測試以達到 80% 覆蓋率要求

## 📈 持續改進

### 監控指標

- 測試執行時間
- 覆蓋率趨勢
- 失敗率統計
- 性能基準

### 未來改進方向

- [ ] 添加性能基準測試
- [ ] 集成更多的代碼品質工具
- [ ] 自動化文檔生成
- [ ] 添加安全性掃描
- [ ] 實現自動化發布流程

---

有問題或建議？請在 GitHub Issues 中提出或參考 [Contributing Guide](contributing.md)。
