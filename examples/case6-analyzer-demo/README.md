# Case 6: Analyzer 示範案例

此案例展示如何使用 Castle Analyzer 進行視訊分析和 ROI 預測。

## 目錄結構

```
case6-analyzer-demo/
├── videos/          # 原始視訊檔案
├── roi_prompt/      # ROI 提示檔案（由 analyzer.save_roi_prompt() 生成）
├── outputs/         # 分析結果和中間檔案
├── config.json      # 分析設定檔
└── README.md        # 此說明檔案
```

## 使用方式

1. 將您的視訊檔案放在 `videos/` 目錄下
2. 執行對應的 notebook 進行分析
3. ROI 提示會自動儲存在 `roi_prompt/` 目錄
4. 分析結果將儲存在 `outputs/` 目錄

## 相關檔案

- **Notebook**: `notebooks/analyzer_tutorial.ipynb`
- **配置檔**: `config.json`

