# 多模态预训练数据格式 (JSONL Schema)

本仓库的预训练 dataloader 通过统一 JSONL 行格式同时驱动**纯文本**与**视觉-语言**训练。本文档定义字段的语义、必需性，以及如何从公开 OCR / VLM 数据集映射进来。

## 行级 Schema

```jsonc
{
  "text":          "<image> 第一段说明文字 ...",        // 必填: 含 <image> 占位符的提示
  "images":        ["/abs/path/page0001.png"],          // 可选: 一行的图片列表
  "image_sizes":   [[1024, 768]],                       // 可选: 与 images 对齐, [[H, W], ...]
  "videos":        [],                                  // 可选: 预留, 暂未消费
  "video_sizes":   [],                                  // 可选
  "ocr_labels":    [[12, 47, 33, 99]],                  // 可选: 每张图的 token id 序列
  "reading_order": [[0, 1, 2, 3]]                       // 可选: 每张图的 patch 阅读序
}
```

### 必填字段
- `input_ids` / `attention_mask` / `labels` 由 `PretrainingDataBuilder` 在 `build_pretraining_data` 阶段从 `text` 生成；下游不需要手填。
- `text`：自然语言提示。`<image>` 占位符会被 `MultimodalProcessor` 展开成 `n_image_tokens` 个 `<image_patch>` 槽位（默认 8，由 `OMVTConfig.compress_to` 控制）。

### 多模态可选字段
| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `images` | `list[str \| bytes \| dict]` | PIL 可读的图片来源；`PILImageProcessor` 在 collator 内即时加载 |
| `image_sizes` | `list[[int, int]]` | 原图 `[H, W]`，仅作为元数据；OMVT 自身会 resize 到 `OMVTConfig.image_size` |
| `ocr_labels` | `list[list[int]]` | 每张图对应的 token id 序列；不在的图保留 `null`；缺失时 OCR 头的 loss 自动跳过 |
| `reading_order` | `list[list[int]]` | 每张图的 patch 阅读序 (长度 ≤ `compress_to`)；不足部分自动 `arange` 补齐 |

### 同批一致性约束
`PretrainingCollator._build_pixel_batch` 要求**同一 micro-batch 内每行的图片数量一致**（典型用法：每行 1 张图）。混合 0/1 图的 batch 必须通过 bucketed dataloader 拆分；强行混批会抛出明确错误。

## 三个训练入口的开关

| 脚本 | 启用多模态的方式 |
| --- | --- |
| `scripts/train_rdt.py` | `--multimodal --image-size 56 --n-image-tokens 8 --data <jsonl>` |
| `scripts/train_vlm_align.py` | `--data <jsonl>`（同时锁住视觉塔可加 `--frozen-vision`） |
| `scripts/train_omvt_ssl.py` | `--data <jsonl>`（orientation 自监督自动生效；OCR/layout 头按字段降级） |

## 公开数据集 → 本 Schema 的迁移骨架

### LAION-COCO / SBU 之类 caption 数据
```python
# 每行 {"image": "...", "caption": "..."}
{
  "text": f"<image> {row['caption']}",
  "images": [row["image"]],
  "image_sizes": [[h, w]],
}
```

### OCR 数据 (IIT-CDIP / DocBank / SynthText)
若标注里含分词后的 token 序列：
```python
{
  "text": "<image>",
  "images": [page_path],
  "ocr_labels": [token_ids],          # 由分词器编码后的 id 序列
  "reading_order": [layout_order],    # 可选: 视觉 layout 模型预测的阅读序
}
```
也可用 `Tokenizer/tools/build_ocr_data.py` 一键转换 `{stem.png, stem.txt or stem.json}` 配对目录：
```bash
python -m Tokenizer.tools.build_ocr_data \
  --input  data/raw_ocr/ \
  --output data/ocr_shard_00.jsonl
```
使用 `--demo` 还可在指定目录生成一个 4 张 PNG 的最小合成集，方便接管前先跑通 smoke。

## 与 OMVT 的耦合
- `images` 在 collator 里被 `PILImageProcessor` 加载为 `[B, 3, H, W]` 张量，随后送入 `Model/omvt/patcher.collate_omvt_batch` 切成 4 个尺度（vertical/horizontal/square/layout）。
- `pixel_values` 是一个 `dict[str, Tensor]`；`train_one_step` 已经会把它递归搬到模型设备并透传给 `RDTForCausalLM(pixel_values=...)`。
- 占位符 `<image_patch>` 已在 `DEFAULT_LABEL_IGNORE_TOKENS` 里，模型不会在这些位置计算 LM 损失。

## 依赖
- 启用多模态需要 `Pillow>=9`；可通过 `pip install -e .[image]` 安装。
- 不强依赖 `torchvision`：`PILImageProcessor` 用纯 PIL + `torch.frombuffer` 实现，无 numpy 依赖。
