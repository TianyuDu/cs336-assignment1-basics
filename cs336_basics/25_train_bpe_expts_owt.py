"""
Run the OpenWebText BPE experiment and dump a small report.
"""

import json
import os
import subprocess
import threading
from pathlib import Path

from cs336_basics.train_bpe import train_bpe

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "owt_train.txt"
OUTPUT_DIR = ROOT / "data" / "bpe_owt"

VOCAB_SIZE = 32_000
SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_TOKEN_BYTES = SPECIAL_TOKEN.encode("utf-8")
RSS_SAMPLE_INTERVAL_S = 0.5
NUM_LONGEST_TOKENS_TO_SHOW = 10

STAGES = [
    ("read_input_s", "Read input"),
    ("find_chunk_boundaries_s", "Find chunk boundaries"),
    ("pretokenize_s", "Pre-tokenize documents"),
    ("build_token_sequences_s", "Build token sequences"),
    ("prepare_merge_structures_s", "Build pair counts + heap"),
    ("merge_loop_s", "Run BPE merge loop"),
]


def process_tree_rss_kib(root_pid: int) -> int:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        capture_output=True,
        check=True,
        text=True,
    )

    rss_by_pid: dict[int, int] = {}
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, rss_kib = map(int, parts)
        rss_by_pid[pid] = rss_kib
        children.setdefault(ppid, []).append(pid)

    total_rss_kib = 0
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if pid not in rss_by_pid:
            continue
        total_rss_kib += rss_by_pid[pid]
        stack.extend(children.get(pid, []))

    return total_rss_kib


class ProcessTreeRSSMonitor:
    def __init__(self, root_pid: int, sample_interval_s: float = RSS_SAMPLE_INTERVAL_S) -> None:
        self.root_pid = root_pid
        self.sample_interval_s = sample_interval_s
        self.peak_rss_kib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> int:
        self._sample()
        self._stop.set()
        self._thread.join()
        return self.peak_rss_kib

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            self._sample()

    def _sample(self) -> None:
        try:
            self.peak_rss_kib = max(self.peak_rss_kib, process_tree_rss_kib(self.root_pid))
        except Exception:
            pass


def gib_from_kib(kib: int) -> float:
    return kib / (1024**2)


def token_text(token: bytes) -> str:
    return token.decode("utf-8", errors="replace").encode("unicode_escape").decode("ascii")


def stage_rows(stats: dict[str, int | float]) -> list[tuple[str, float, float]]:
    total_s = float(stats["total_s"])
    rows: list[tuple[str, float, float]] = []
    for key, label in STAGES:
        seconds = float(stats[key])
        share = 100.0 * seconds / total_s if total_s else 0.0
        rows.append((label, seconds, share))
    return rows


def longest_tokens(vocab: dict[int, bytes], limit: int = NUM_LONGEST_TOKENS_TO_SHOW) -> list[bytes]:
    non_special_tokens = [token for token in vocab.values() if token != SPECIAL_TOKEN_BYTES]
    non_special_tokens.sort(key=lambda token: (len(token), token), reverse=True)
    return non_special_tokens[:limit]


def save_vocab_and_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    vocab_json = {str(token_id): list(token_bytes) for token_id, token_bytes in vocab.items()}
    with open(OUTPUT_DIR / "vocab.json", "w") as f:
        json.dump(vocab_json, f)

    with open(OUTPUT_DIR / "merges.txt", "w") as f:
        for left, right in merges:
            f.write(f"{left!r} {right!r}\n")


def write_report(
    stats: dict[str, int | float],
    peak_rss_kib: int,
    longest_token_list: list[bytes],
    special_token_present: bool,
    rows: list[tuple[str, float, float]],
) -> None:
    bottleneck_label, bottleneck_s, bottleneck_share = max(rows, key=lambda row: row[1])

    report = {
        "data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "vocab_size_target": VOCAB_SIZE,
        "final_vocab_size": int(stats["final_vocab_size"]),
        "special_token": SPECIAL_TOKEN,
        "special_token_present": special_token_present,
        "peak_process_tree_rss_kib": peak_rss_kib,
        "peak_process_tree_rss_gib": gib_from_kib(peak_rss_kib),
        "longest_tokens": [
            {
                "bytes_repr": repr(token),
                "decoded": token_text(token),
                "length_bytes": len(token),
            }
            for token in longest_token_list
        ],
        "stage_timings": [
            {"stage": label, "seconds": seconds, "share_pct": share}
            for label, seconds, share in rows
        ],
        "bottleneck": {
            "stage": bottleneck_label,
            "seconds": bottleneck_s,
            "share_pct": bottleneck_share,
        },
        "train_bpe_stats": stats,
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# OpenWebText BPE experiment",
        "",
        f"- Data: `{DATA_PATH}`",
        f"- Output directory: `{OUTPUT_DIR}`",
        f"- Target vocab size: `{VOCAB_SIZE}`",
        f"- Special token present: `{special_token_present}`",
        f"- Peak process-tree RSS (sampled): `{gib_from_kib(peak_rss_kib):.2f} GiB`",
        "",
        "## Longest tokens",
    ]
    lines.extend(
        f"- `{repr(token)}` decoded as `{token_text(token)}` ({len(token)} bytes)"
        for token in longest_token_list
    )
    lines.append("")
    lines.append("## Stage timings")
    lines.extend(
        f"- {label}: {seconds:.1f}s ({share:.1f}% of total)" for label, seconds, share in rows
    )
    lines.append(
        f"- Bottleneck: {bottleneck_label} at {bottleneck_s:.1f}s ({bottleneck_share:.1f}% of total)"
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def print_summary(
    stats: dict[str, int | float],
    peak_rss_kib: int,
    longest_token_list: list[bytes],
    special_token_present: bool,
    rows: list[tuple[str, float, float]],
) -> None:
    total_s = float(stats["total_s"])
    input_bytes = int(stats["input_bytes"])
    input_gib = input_bytes / (1024**3)
    bottleneck_label, bottleneck_s, bottleneck_share = max(rows, key=lambda row: row[1])

    print("\n" + "=" * 72)
    print("OpenWebText BPE Experiment")
    print("=" * 72)
    print(f"Data path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target vocab size: {VOCAB_SIZE}")
    print(f"Special token: {SPECIAL_TOKEN}")

    print("\nArtifacts")
    print(f"- vocab.json: {OUTPUT_DIR / 'vocab.json'}")
    print(f"- merges.txt: {OUTPUT_DIR / 'merges.txt'}")
    print(f"- report.md: {OUTPUT_DIR / 'report.md'}")
    print(f"- report.json: {OUTPUT_DIR / 'report.json'}")

    print("\nTraining summary")
    print(f"- Input size: {input_bytes} bytes ({input_gib:.2f} GiB)")
    print(f"- Total wall-clock time: {total_s:.1f}s")
    print(
        f"- Approx peak memory (sampled process-tree RSS): "
        f"{gib_from_kib(peak_rss_kib):.2f} GiB ({peak_rss_kib} KiB)"
    )
    print(f"- Requested workers: {int(stats['requested_workers'])}")
    print(f"- Worker processes used: {int(stats['worker_processes'])}")
    print(f"- Number of chunks: {int(stats['num_chunks'])}")
    print(f"- Final vocab size: {int(stats['final_vocab_size'])}")
    print(f"- Merges learned: {int(stats['num_merges_learned'])}")
    print(f"- Special token present in vocab: {special_token_present}")
    print(f"- Unique pretokens: {int(stats['unique_pretokens'])}")
    print(f"- Unique token sequences: {int(stats['unique_token_sequences'])}")
    print(f"- Initial distinct adjacent pairs: {int(stats['initial_pair_count'])}")

    print("\nLongest tokens")
    for i, token in enumerate(longest_token_list, start=1):
        print(f"- {i}. {token!r} decoded as {token_text(token)!r} ({len(token)} bytes)")

    print("\nStage timings")
    for label, seconds, share in rows:
        print(f"- {label}: {seconds:.1f}s ({share:.1f}% of total)")
    print(
        f"- Bottleneck: {bottleneck_label} at {bottleneck_s:.1f}s ({bottleneck_share:.1f}% of total)"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE on {DATA_PATH} ...")
    monitor = ProcessTreeRSSMonitor(os.getpid())
    monitor.start()
    try:
        vocab, merges, stats = train_bpe(
            input_path=str(DATA_PATH),
            vocab_size=VOCAB_SIZE,
            special_tokens=[SPECIAL_TOKEN],
            collect_stats=True,
        )
    finally:
        peak_rss_kib = monitor.stop()

    special_token_present = SPECIAL_TOKEN_BYTES in vocab.values()
    rows = stage_rows(stats)
    longest_token_list = longest_tokens(vocab)

    save_vocab_and_merges(vocab, merges)
    write_report(stats, peak_rss_kib, longest_token_list, special_token_present, rows)
    print_summary(stats, peak_rss_kib, longest_token_list, special_token_present, rows)


if __name__ == "__main__":
    main()
