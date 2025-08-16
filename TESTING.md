# 🧪 Testing Guide for Castle-AI

## Quick Start

```bash
# 快速測試 ROI Manager
make test-roi

# 生成覆蓋率報告
make test-coverage

# 查看所有可用命令
make help
```

## Available Test Commands

| 命令 | 描述 |
|------|------|
| `make test-roi` | 運行所有 ROI Manager 測試 |
| `make test-quick` | 快速測試（跳過慢測試） |
| `make test-coverage` | 生成測試覆蓋率報告 |
| `make test-performance` | 運行性能測試 |
| `make pre-commit` | 提交前完整檢查 |

## Manual Testing

```bash
# 使用自定義腳本
python scripts/test_roi_manager.py --help

# 直接使用 pytest
pytest tests/test_roi_manager.py -v

# 特定測試類型
pytest tests/test_roi_manager.py -m unit
pytest tests/test_roi_manager.py -m integration
pytest tests/test_roi_manager.py -m slow
```

## CI/CD Status

- ✅ GitHub Actions CI 配置完成
- ✅ 多平台測試 (Ubuntu, macOS, Windows)
- ✅ 多 Python 版本支援 (3.8-3.11)
- ✅ 自動覆蓋率報告
- ✅ Pre-commit hooks 配置

## Requirements

- **最低覆蓋率**: 80%
- **支援平台**: Linux, macOS, Windows
- **Python 版本**: 3.8+
- **主要依賴**: h5py, numpy, pytest

## Documentation

詳細的測試和 CI/CD 說明請參考：
- [Testing and CI/CD Guide](docs/development/testing-and-cicd.md)
- [Contributing Guide](docs/development/contributing.md)

---

🎉 **測試系統已完全整合！** 現在您可以自信地開發和部署 Castle-AI。
