#!/usr/bin/env python3
"""
ROI Manager 測試運行腳本

此腳本提供便利的方式來運行 ROI Manager 的各種測試：
- 單元測試
- 整合測試  
- 性能測試
- 覆蓋率報告

用法:
    python scripts/test_roi_manager.py [選項]

選項:
    --unit       只運行單元測試
    --integration 只運行整合測試
    --performance 只運行性能測試
    --coverage   生成覆蓋率報告
    --verbose    詳細輸出
    --quick      快速測試（跳過慢測試）
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 確保腳本可以找到 castle 模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description):
    """運行命令並處理結果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"運行命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} - 成功完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - 失敗 (exit code: {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="ROI Manager 測試運行腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="只運行單元測試"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="只運行整合測試"
    )
    parser.add_argument(
        "--performance", 
        action="store_true", 
        help="只運行性能測試"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="生成覆蓋率報告"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="詳細輸出"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="快速測試（跳過慢測試）"
    )
    
    args = parser.parse_args()
    
    # 基本 pytest 命令
    base_cmd = ["python", "-m", "pytest", "tests/test_roi_manager.py"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    success_count = 0
    total_tests = 0
    
    # 根據參數決定運行哪些測試
    if args.unit:
        cmd = base_cmd + ["-m", "unit"]
        if args.coverage:
            cmd.extend([
                "--cov=castle.utils.roi_manager",
                "--cov-report=term-missing",
                "--cov-report=html"
            ])
        total_tests += 1
        if run_command(cmd, "單元測試"):
            success_count += 1
            
    elif args.integration:
        cmd = base_cmd + ["-m", "integration"]
        total_tests += 1
        if run_command(cmd, "整合測試"):
            success_count += 1
            
    elif args.performance:
        cmd = base_cmd + ["-m", "slow"]
        total_tests += 1
        if run_command(cmd, "性能測試"):
            success_count += 1
            
    else:
        # 運行所有測試
        cmd = base_cmd.copy()
        
        if args.quick:
            cmd.extend(["-m", "not slow"])
            test_name = "快速測試"
        else:
            test_name = "所有測試"
            
        if args.coverage:
            cmd.extend([
                "--cov=castle.utils.roi_manager",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--cov-fail-under=80"
            ])
            
        total_tests += 1
        if run_command(cmd, test_name):
            success_count += 1
    
    # 結果總結
    print(f"\n{'='*60}")
    print("📊 測試結果總結")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count}/{total_tests}")
    print(f"❌ 失敗: {total_tests - success_count}/{total_tests}")
    
    if args.coverage and success_count > 0:
        print(f"\n📈 覆蓋率報告已生成到 htmlcov/ 目錄")
        print(f"🌐 在瀏覽器中開啟: htmlcov/index.html")
    
    # 額外的開發提示
    if success_count == total_tests:
        print(f"\n🎉 所有測試都通過了！")
        print(f"💡 提示:")
        print(f"   - 使用 --coverage 查看覆蓋率")
        print(f"   - 使用 --performance 測試性能")
        print(f"   - 使用 --quick 快速測試")
    else:
        print(f"\n⚠️  有測試失敗，請檢查上面的錯誤訊息")
        
    # 返回適當的退出碼
    sys.exit(0 if success_count == total_tests else 1)


if __name__ == "__main__":
    main()
