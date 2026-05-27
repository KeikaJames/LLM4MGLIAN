# -*- coding: utf-8 -*-
"""Collect short Mongolian phrases from Menksoft's public MT endpoint.

The endpoint is useful for building a review queue and tokenizer hit-rate
experiments. Treat the output as silver data: useful for coverage tests, but
still worth human review before training a final production tokenizer.
"""

from __future__ import annotations

import argparse
import html
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

MENKSOFT_TRANSLATE_URL = "https://mts.menksoft.com/Home/ToTranslate"

DEFAULT_ZH_SEEDS = [
    "你好",
    "谢谢你",
    "明天见",
    "我是学生",
    "我们学习蒙古文",
    "蒙古文很重要",
    "老师正在讲课",
    "孩子们在学校读书",
    "牧民住在草原上",
    "春天来了",
    "夏天的雨很多",
    "秋天的风很凉",
    "冬天的雪很厚",
    "太阳从东方升起",
    "月亮照着故乡",
    "我的家在草原",
    "这是我的书",
    "那匹马跑得很快",
    "白云飘过蓝天",
    "清晨的空气很好",
    "我们保护母语",
    "他喜欢读历史",
    "她正在写信",
    "父亲去了牧场",
    "母亲做了奶茶",
    "朋友从远方来",
    "今天工作很忙",
    "请把门打开",
    "请慢慢说",
    "这个词怎么写",
    "这句话是什么意思",
    "我想学习传统蒙古文",
    "他们在河边唱歌",
    "孩子把花送给老师",
    "马群穿过山谷",
    "羊群回到家附近",
    "我们一起去市场",
    "这座城市很安静",
    "草原上的夜晚很美",
    "风吹过高高的山",
    "河水流向远方",
    "老人讲古老的故事",
    "年轻人学习新的技术",
    "请给我一杯水",
    "我们今天开始实验",
    "这个模型需要更多数据",
    "分词器应该尊重词根和后缀",
    "不要把后缀合并到错误的位置",
    "这个单词有多个附加成分",
    "边界召回率需要提高",
    "未知词比例必须降低",
    "训练语料要保持干净",
    "编码转换以后再训练",
    "标准Unicode文本最适合分词",
    "机器翻译可以帮助收集短语",
    "自动校对可以发现未识别词",
    "我们需要高质量短句",
    "短句适合检查命中率",
    "长句适合检查上下文",
    "请保存原文和译文",
    "每天增加一点数据",
    "先做小实验再扩大规模",
    "这个结果可以人工审核",
    "评估脚本输出JSON指标",
    "模型训练以前先看分词",
    "蒙古语是黏着语",
    "词根后面可以接很多后缀",
    "后缀顺序很重要",
    "元音和谐也很重要",
    "学生们正在认真学习",
    "老师检查了作业",
    "我们写下新的词汇",
    "这本书讲草原文化",
    "历史和语言关系密切",
    "家乡的河水很清",
    "远处传来马蹄声",
    "夜空中有很多星星",
    "我把消息告诉朋友",
    "请把这段文字翻译成蒙古文",
    "我们明天继续测试",
]


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def translate_batch(
    phrases: list[str],
    timeout: float = 30.0,
    verify_tls: bool = True,
) -> list[dict]:
    body = urllib.parse.urlencode(
        {
            "from": "zh",
            "to": "mw",
            "Content": "\n".join(phrases),
            "IsUploadFile": "false",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        MENKSOFT_TRANSLATE_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "LLM4MGLIAN-tokenizer-hit-rate/0.1",
        },
        method="POST",
    )
    context = None if verify_tls else ssl._create_unverified_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not payload.get("Success"):
        raise RuntimeError(payload.get("Msg") or "Menksoft translation failed")

    rows = json.loads(payload.get("Text") or "[]")
    pid = payload.get("Pid")
    out = []
    for row in rows:
        translated = row.get("TransList", {}).get("Trans", "")
        translated = html.unescape(translated).replace("\xa0", " ").strip()
        out.append(
            {
                "source": "menksoft_mt",
                "provider": MENKSOFT_TRANSLATE_URL,
                "pid": pid,
                "source_zh": row.get("Content", ""),
                "text": translated,
            }
        )
    return out


def load_phrases(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_ZH_SEEDS)
    phrases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                phrases.append(line)
    return phrases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="UTF-8 Chinese phrase list, one per line")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS certificate verification for local collection only",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    phrases = load_phrases(args.input)[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(batched(phrases, args.batch_size)):
            for row in translate_batch(
                batch,
                timeout=args.timeout,
                verify_tls=not args.insecure,
            ):
                if row["text"]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            if batch_idx < (len(phrases) - 1) // args.batch_size:
                time.sleep(args.sleep)

    print(f"wrote={count}", file=sys.stderr)


if __name__ == "__main__":
    main()
