# subtitle-translator

极简 SRT 字幕翻译工具 -- 自动检测源语言，翻译为简体中文，输出双语字幕（中文 + 原文）。

通过 DeepSeek API 进行翻译。

## 使用方法

### 1. 配置

编辑 `config/deepseek.yaml`：

```yaml
api_key: sk-your-api-key      # 自己的DeepSeek API 密钥（申请api：https://platform.deepseek.com/api_keys）
in_srt: /path/to/subtitle.srt # 要翻译的字幕文件的路径
```

### 2. 运行

```bash
python src/srt_trans.py
```

翻译结果输出到 `data/subs/{文件名}.zh-llm.srt`。
