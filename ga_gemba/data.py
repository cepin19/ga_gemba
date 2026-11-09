"""Reading inputs, shard ranges and resumable outputs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f]


def read_grouped(path: str, group_size: int, n_groups: int, what: str) -> List[List[str]]:
    lines = read_lines(path)
    if len(lines) != group_size * n_groups:
        raise ValueError(
            f"{path}: expected {group_size} x {n_groups} = {group_size * n_groups} lines of {what}, found {len(lines)}"
        )
    return [lines[i * group_size : (i + 1) * group_size] for i in range(n_groups)]


def split_phrases(text: str) -> List[str]:
    """Dictionary and wordlist entries are separated by ';' (newlines also work for wordlists)."""
    return [p.strip() for p in text.replace("\n", ";").split(";") if p.strip()]


def parse_weights(spec: str) -> Dict[str, float]:
    """'comet=0.4,bleu=0.1' -> {'comet': 0.4, 'bleu': 0.1}"""
    weights = {}
    for item in spec.split(","):
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Bad weight '{item}', expected name=value")
        weights[name.strip()] = float(value)
    return weights


def parse_shard(spec: Optional[str]) -> Tuple[int, int]:
    if not spec:
        return 0, 1
    index, _, total = spec.partition("/")
    i, n = int(index), int(total)
    if not 0 <= i < n:
        raise ValueError(f"Bad shard {spec}: expected I/N with 0 <= I < N")
    return i, n


def shard_range(n_items: int, shard: int, n_shards: int) -> range:
    return range(n_items * shard // n_shards, n_items * (shard + 1) // n_shards)


def _complete_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line for line in f if line.endswith("\n")]


@dataclass
class OutputWriter:
    """Writes the text and JSONL outputs in step; flushes after every segment."""

    text_path: str
    jsonl_path: str
    n_done: int = 0

    def __post_init__(self):
        self._text = None
        self._jsonl = None

    def open(self, resume: bool) -> "OutputWriter":
        texts, records = _complete_lines(self.text_path), _complete_lines(self.jsonl_path)
        if (texts or records) and not resume:
            raise FileExistsError(f"{self.text_path} or {self.jsonl_path} already has content; use --resume")
        # A job killed mid-write can leave the files out of step: keep what both have.
        self.n_done = min(len(texts), len(records))
        for path, lines in ((self.text_path, texts), (self.jsonl_path, records)):
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines[: self.n_done])
        self._text = open(self.text_path, "a", encoding="utf-8")
        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8")
        return self

    def done_records(self) -> List[dict]:
        return [json.loads(line) for line in _complete_lines(self.jsonl_path)]

    def write(self, text: str, record: dict) -> None:
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        self._text.write(text.replace("\n", " ") + "\n")
        self._text.flush()

    def close(self) -> None:
        for handle in (self._text, self._jsonl):
            if handle:
                handle.close()


def summarize(records: Sequence[dict]) -> str:
    """Per-metric means for the best initial candidate and the GA result."""
    if not records:
        return "No segments."
    names: List[str] = []
    for record in records:
        names.extend(n for n in record["best"]["scores"] if n not in names)
    lines = [f"{'metric':<20}{'original':>10}{'ga':>10}{'change':>10}{'n':>7}"]
    for name in names:
        pairs = [
            (r["original"]["scores"].get(name), r["best"]["scores"].get(name)) for r in records
        ]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if pairs:
            before = sum(a for a, _ in pairs) / len(pairs)
            after = sum(b for _, b in pairs) / len(pairs)
            lines.append(f"{name:<20}{before:>10.4f}{after:>10.4f}{after - before:>+10.4f}{len(pairs):>7}")
    improved = sum(r["best"]["fitness"] > r["original"]["fitness"] for r in records)
    lines.append(f"Fitness improved on {improved}/{len(records)} segments.")
    return "\n".join(lines)


def collect(paths: Iterable[str], text_out: Optional[str], jsonl_out: Optional[str], allow_partial: bool) -> List[dict]:
    records: Dict[int, dict] = {}
    total = None
    for path in paths:
        for line in _complete_lines(path):
            record = json.loads(line)
            if record["index"] in records:
                raise ValueError(f"Segment {record['index']} appears twice (again in {path})")
            records[record["index"]] = record
            total = record["total"]
    if total is None:
        raise ValueError("No finished segments found in the given files")
    missing = [i for i in range(total) if i not in records]
    if missing and not allow_partial:
        shown = ", ".join(map(str, missing[:10]))
        raise ValueError(f"{len(missing)}/{total} segments missing (first: {shown}); use --allow-partial to collect anyway")
    ordered = [records[i] for i in sorted(records)]
    if text_out:
        with open(text_out, "w", encoding="utf-8") as f:
            f.writelines(r["best"]["text"].replace("\n", " ") + "\n" for r in ordered)
    if jsonl_out:
        with open(jsonl_out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered)
    return ordered
