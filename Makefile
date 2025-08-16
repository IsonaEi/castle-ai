# Castle-AI Makefile

.PHONY: help test test-roi test-quick test-coverage test-performance install-dev clean lint format

# 預設目標
help:
	@echo "Castle-AI 開發工具"
	@echo ""
	@echo "可用命令:"
	@echo "  install-dev     安裝開發依賴"
	@echo "  test           運行所有測試"
	@echo "  test-roi       運行 ROI Manager 測試"
	@echo "  test-quick     運行快速測試（跳過慢測試）"
	@echo "  test-coverage  運行測試並生成覆蓋率報告"
	@echo "  test-performance  運行性能測試"
	@echo "  lint           代碼檢查"
	@echo "  format         代碼格式化"
	@echo "  clean          清理臨時檔案"

# 安裝開發依賴
install-dev:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# 運行所有測試
test:
	python -m pytest tests/ -v

# 運行 ROI Manager 測試
test-roi:
	python scripts/test_roi_manager.py --verbose

# 快速測試
test-quick:
	python scripts/test_roi_manager.py --quick --verbose

# 測試覆蓋率
test-coverage:
	python scripts/test_roi_manager.py --coverage --verbose
	@echo ""
	@echo "📈 覆蓋率報告已生成："
	@echo "   - 終端報告：已顯示上方"
	@echo "   - HTML 報告：htmlcov/index.html"

# 性能測試
test-performance:
	python scripts/test_roi_manager.py --performance --verbose

# 代碼檢查
lint:
	@echo "🔍 運行 flake8 代碼檢查..."
	python -m flake8 castle/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "✅ 代碼檢查完成"

# 代碼格式化
format:
	@echo "🎨 格式化代碼..."
	python -m black castle/ tests/ --line-length=100
	python -m isort castle/ tests/ --profile black
	@echo "✅ 代碼格式化完成"

# 清理臨時檔案
clean:
	@echo "🧹 清理臨時檔案..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	@echo "✅ 清理完成"

# CI 測試（GitHub Actions 使用）
test-ci:
	python -m pytest tests/test_roi_manager.py \
		-v \
		--cov=castle.utils.roi_manager \
		--cov-report=xml \
		--cov-report=term-missing \
		--cov-fail-under=80

# 本地完整檢查（提交前運行）
pre-commit: clean lint test-coverage
	@echo ""
	@echo "🎉 所有檢查都通過！可以安全提交。"
	@echo ""

# 安裝 pre-commit hooks
install-hooks:
	pip install pre-commit
	pre-commit install

# 發布檢查
release-check: clean lint test
	python setup.py check --strict --metadata
	python setup.py bdist_wheel
	@echo "✅ 發布檢查完成"
