"""
SRT 字幕翻译工具（自动检测源语言 -> 简体中文）

功能：
├─ 解析 SRT 字幕文件
├─ 分批调用 DeepSeek API 翻译（自动检测源语言 -> 中文）
├─ 支持断点续翻（已翻译部分自动跳过）
├─ 多线程并发请求
└─ 输出双语 SRT（中文在上，原文在下）

输入：
  配置文件路径：config/deepseek.yaml
  数据文件路径：由配置中 in_srt 指定

输出：
  数据文件路径：data/subs/{原文件名}.zh-llm.srt
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


# ======= Path =======
CFG_YAML = Path("config/deepseek.yaml")
OUT_DIR = Path("data/subs")


# ======= SRT =======
@dataclass(frozen=True)
class SrtBlock:
    idx: int
    time: str
    lines: list[str]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_srt(s: str) -> list[SrtBlock]:
    lines = s.splitlines()
    i = 0
    blocks: list[SrtBlock] = []

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        idx = int(lines[i].strip())
        i += 1
        t = lines[i].strip()
        i += 1

        txt: list[str] = []
        while i < len(lines) and lines[i].strip():
            txt.append(lines[i].rstrip("\n"))
            i += 1

        blocks.append(SrtBlock(idx=idx, time=t, lines=txt))
        while i < len(lines) and not lines[i].strip():
            i += 1

    return blocks


def write_bi_srt_line(f, b: SrtBlock, zh_lines: list[str]) -> None:
    f.write(f"{b.idx}\n")
    f.write(f"{b.time}\n")
    if zh_lines:
        for line in zh_lines:
            f.write(f"{line}\n")
    if b.lines:
        for line in b.lines:
            f.write(f"{line}\n")
    f.write("\n")


# ======= Config (KV-YAML) =======
def load_yaml_kv(path: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        k, v = raw.split(":", 1)
        key = k.strip()
        val = v.strip()

        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            out[key] = val[1:-1]
            continue

        if val.lower() == "true":
            out[key] = True
            continue
        if val.lower() == "false":
            out[key] = False
            continue

        try:
            out[key] = int(val)
            continue
        except ValueError:
            pass

        try:
            out[key] = float(val)
            continue
        except ValueError:
            pass

        out[key] = val

    return out


# ======= DeepSeek API =======
API_URL = "https://api.deepseek.com/chat/completions"
MARK_RE = re.compile(r"<<<SRT:(\d{6})>>>")


def mk_mark(idx: int) -> str:
    return f"<<<SRT:{idx:06d}>>>"


def mk_batch_text(blocks: list[SrtBlock]) -> str:
    out: list[str] = []
    for b in blocks:
        out.append(mk_mark(b.idx))
        if b.lines:
            out.extend(b.lines)
        else:
            out.append("")
    return "\n".join(out) + "\n"


def split_batches(blocks: list[SrtBlock], max_blocks: int, max_chars: int) -> list[list[SrtBlock]]:
    batches: list[list[SrtBlock]] = []
    cur: list[SrtBlock] = []
    cur_chars = 0

    for b in blocks:
        s = "\n".join(b.lines)
        add_chars = len(s) + 32
        hit_blocks = cur and len(cur) >= max_blocks
        hit_chars = cur and (cur_chars + add_chars) > max_chars

        if hit_blocks or hit_chars:
            batches.append(cur)
            cur = []
            cur_chars = 0

        cur.append(b)
        cur_chars += add_chars

    if cur:
        batches.append(cur)

    return batches


def post_chat(api_key: str, payload: dict, timeout_s: int) -> dict:
    req = urllib.request.Request(
        url=API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fmt_bar(done: int, total: int, width: int) -> str:
    if total <= 0:
        return ""

    d = done
    if d < 0:
        d = 0
    if d > total:
        d = total

    ratio = d / total
    fill = int(ratio * width)
    bar = ("▓" * fill) + ("░" * (width - fill))
    pct = int(ratio * 100)
    return f"{bar} {pct:3d}% {d}/{total}"


def show_bar(done: int, total: int, width: int) -> None:
    s = fmt_bar(done, total, width)
    if not s:
        return
    print(f"\r翻译进度 {s}", end="", flush=True)
    if done >= total:
        print("", flush=True)


def translate_one(cfg: dict[str, object], src_lines: list[str]) -> list[str]:
    if not src_lines:
        return []

    sys = (
        "你是字幕翻译器。自动检测输入字幕的语言，将其翻译成简体中文。\n"
        "如果输入已经是中文，则原样输出。\n"
        "只输出中文译文，不要输出原文，不要输出解释，不要使用 Markdown 代码块。\n"
    )
    payload = {
        "model": cfg["model"],
        "temperature": cfg["temperature"],
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": "\n".join(src_lines)},
        ],
        "stream": False,
    }
    res = post_chat(
        api_key=str(cfg["api_key"]),
        payload=payload,
        timeout_s=int(cfg["timeout_s"]),
    )
    content = res["choices"][0]["message"]["content"]
    lines = [x.rstrip() for x in content.splitlines() if x.strip() and not x.strip().startswith("```")]
    return lines


def parse_zh_map(content: str) -> dict[int, list[str]]:
    cleaned = "\n".join([x for x in content.splitlines() if not x.strip().startswith("```")])
    marks = list(MARK_RE.finditer(cleaned))

    out: dict[int, list[str]] = {}
    for i, m in enumerate(marks):
        idx = int(m.group(1))
        seg_start = m.end()
        seg_end = marks[i + 1].start() if i + 1 < len(marks) else len(cleaned)
        seg = cleaned[seg_start:seg_end].strip()
        lines = [x.rstrip() for x in seg.splitlines() if x.strip()]
        out[idx] = lines

    return out


def translate_batch(cfg: dict[str, object], blocks: list[SrtBlock]) -> dict[int, list[str]]:
    sys = (
        "你是字幕翻译器。自动检测输入字幕的语言，将其翻译成简体中文。\n"
        "如果输入已经是中文，则原样输出。\n"
        "输入会包含若干标记行，形如 <<<SRT:000001>>>。\n"
        "输出必须严格遵守：\n"
        "1) 必须原样输出每个标记行（不要翻译/改动标记）。\n"
        "2) 每个标记行后，紧跟该条字幕的中文翻译（可多行）。\n"
        "3) 不要输出任何原文，不要输出解释，不要使用 Markdown 代码块。\n"
    )

    payload = {
        "model": cfg["model"],
        "temperature": cfg["temperature"],
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": mk_batch_text(blocks)},
        ],
        "stream": False,
    }

    res = post_chat(
        api_key=str(cfg["api_key"]),
        payload=payload,
        timeout_s=int(cfg["timeout_s"]),
    )
    content = res["choices"][0]["message"]["content"]
    zh_map = parse_zh_map(content)

    need = {b.idx for b in blocks}
    got = set(zh_map.keys())
    missing = sorted(need - got)
    if missing:
        for b in blocks:
            if b.idx in missing:
                zh_map[b.idx] = translate_one(cfg, b.lines)

    return zh_map


# ======= Main =======
def last_idx_in_srt(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    last = 0
    for i, l in enumerate(lines[:-1]):
        if l.strip().isdigit() and "-->" in lines[i + 1]:
            last = int(l.strip())
    return last


def run_batch(cfg: dict[str, object], batch: list[SrtBlock]) -> dict[int, list[str]]:
    zh_map = translate_batch(cfg, batch)
    pause_s = float(cfg["pause_s"])
    if pause_s > 0:
        time.sleep(pause_s)
    return zh_map


def main() -> None:
    cfg = load_yaml_kv(CFG_YAML)
    in_srt = Path(str(cfg["in_srt"]))
    out_srt = OUT_DIR / f"{in_srt.stem}.zh-llm.srt"

    blocks = parse_srt(read_text(in_srt))
    total = len(blocks)
    done = last_idx_in_srt(out_srt) if out_srt.exists() else 0
    todo = [b for b in blocks if b.idx > done]
    if not todo:
        show_bar(total, total, int(cfg["progress_len"]))
        print("[done] 输出已完整存在", flush=True)
        return

    batches = split_batches(todo, int(cfg["max_blocks"]), int(cfg["max_chars"]))
    max_workers = int(cfg["max_workers"])
    bar_len = int(cfg["progress_len"])

    out_srt.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if done else "w"
    with out_srt.open(mode, encoding="utf-8", newline="\n") as f:
        show_bar(done, total, bar_len)

        pending: dict[int, dict[int, list[str]]] = {}
        next_write = 0
        cur_done = done

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2i = {ex.submit(run_batch, cfg, batch): i for i, batch in enumerate(batches)}

            for fut in as_completed(fut2i):
                i = fut2i[fut]
                pending[i] = fut.result()
                cur_done += len(batches[i])
                show_bar(cur_done, total, bar_len)

                while next_write in pending:
                    zh_map = pending.pop(next_write)
                    for b in batches[next_write]:
                        write_bi_srt_line(f, b, zh_map[b.idx])
                    next_write += 1


if __name__ == "__main__":
    main()
