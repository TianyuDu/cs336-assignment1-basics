"""
Run the TinyStories BPE experiment and save a ready-to-cite report.
"""

import json
import os
import subprocess
import threading
from pathlib import Path

from cs336_basics.train_bpe import TrainBPEStats, train_bpe

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "TinyStoriesV2-GPT4-train.txt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "bpe_tinystories"
SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_TOKEN_BYTES = SPECIAL_TOKEN.encode("utf-8")
VOCAB_SIZE = 10000
RSS_SAMPLE_INTERVAL_S = 0.5
STAGE_LABELS = [
    ("read_input_s", "Read input"),
    ("find_chunk_boundaries_s", "Find chunk boundaries"),
    ("pretokenize_s", "Pre-tokenize documents"),
    ("build_token_sequences_s", "Build token sequences"),
    ("prepare_merge_structures_s", "Build pair counts + heap"),
    ("merge_loop_s", "Run BPE merge loop"),
]


def process_tree_rss_kib(root_pid: int) -> int:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        capture_output=True,
        check=True,
        text=True,
    )
    rss_by_pid: dict[int, int] = {}
    children_by_parent: dict[int, list[int]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, rss_kib = (int(part) for part in parts)
        rss_by_pid[pid] = rss_kib
        children_by_parent.setdefault(ppid, []).append(pid)

    total_rss_kib = 0
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen or pid not in rss_by_pid:
            continue
        seen.add(pid)
        total_rss_kib += rss_by_pid[pid]
        stack.extend(children_by_parent.get(pid, []))
    return total_rss_kib


class ProcessTreeRSSMonitor:
    def __init__(self, root_pid: int, sample_interval_s: float = RSS_SAMPLE_INTERVAL_S) -> None:
        self.root_pid = root_pid
        self.sample_interval_s = sample_interval_s
        self.peak_rss_kib = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._sample_once()
        self._thread.start()

    def stop(self) -> int:
        self._sample_once()
        self._stop_event.set()
        self._thread.join()
        return self.peak_rss_kib

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            rss_kib = process_tree_rss_kib(self.root_pid)
        except Exception:
            return
        if rss_kib > self.peak_rss_kib:
            self.peak_rss_kib = rss_kib


def kib_to_gib(kib: int) -> float:
    return kib / (1024**2)


def render_token_text(token: bytes) -> str:
    return token.decode("utf-8", errors="replace").encode("unicode_escape").decode("ascii")


def get_longest_non_special_token(vocab: dict[int, bytes]) -> bytes:
    return max((token for token in vocab.values() if token != SPECIAL_TOKEN_BYTES), key=len)


def build_stage_rows(stats: TrainBPEStats) -> list[tuple[str, float, float]]:
    total_s = float(stats["total_s"])
    rows: list[tuple[str, float, float]] = []
    for key, label in STAGE_LABELS:
        seconds = float(stats[key])
        share = 100.0 * seconds / total_s if total_s else 0.0
        rows.append((label, seconds, share))
    return rows


def build_part_a_answer(training_s: float, peak_rss_kib: int, longest_token: bytes) -> str:
    longest_text = render_token_text(longest_token)
    peak_gib = kib_to_gib(peak_rss_kib)
    return (
        f"Training the 10,000-token byte-level BPE tokenizer on TinyStories took {training_s:.1f}s "
        f"and reached about {peak_gib:.2f} GiB peak process-tree RSS during training. "
        f"The longest learned token was {longest_token!r} (decoded as {longest_text!r}); this makes "
        "sense because BPE tends to merge very frequent whole words or word pieces, often including "
        "the leading space when that word usually appears after whitespace."
    )


def build_part_b_answer(stage_rows: list[tuple[str, float, float]]) -> str:
    ranked = sorted(stage_rows, key=lambda row: row[1], reverse=True)
    bottleneck_label, bottleneck_s, bottleneck_share = ranked[0]
    second_label, second_s, second_share = ranked[1]
    return (
        f"Stage-level profiling of the full multiprocessing run shows that {bottleneck_label.lower()} "
        f"is the bottleneck at {bottleneck_s:.1f}s ({bottleneck_share:.1f}% of total wall-clock time), "
        f"followed by {second_label.lower()} at {second_s:.1f}s ({second_share:.1f}%). "
        "Because pre-tokenization runs in worker processes, this full-run stage timing is more "
        "representative than a parent-only cProfile dump."
    )


def write_serialized_outputs(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    vocab_serializable = {str(token_id): list(token_bytes) for token_id, token_bytes in vocab.items()}
    with open(OUTPUT_DIR / "vocab.json", "w") as f:
        json.dump(vocab_serializable, f)
    with open(OUTPUT_DIR / "merges.txt", "w") as f:
        for left, right in merges:
            f.write(f"{left!r} {right!r}\n")


def write_report(
    stats: TrainBPEStats,
    peak_rss_kib: int,
    special_token_present: bool,
    longest_token: bytes,
    part_a_answer: str,
    part_b_answer: str,
    stage_rows: list[tuple[str, float, float]],
) -> None:
    longest_text = render_token_text(longest_token)
    report = {
        "data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "vocab_size_target": VOCAB_SIZE,
        "final_vocab_size": int(stats["final_vocab_size"]),
        "special_token": SPECIAL_TOKEN,
        "special_token_present": special_token_present,
        "peak_process_tree_rss_kib": peak_rss_kib,
        "peak_process_tree_rss_gib": kib_to_gib(peak_rss_kib),
        "longest_token": {
            "bytes_repr": repr(longest_token),
            "decoded": longest_text,
            "length_bytes": len(longest_token),
        },
        "stage_timings_seconds": {
            label: seconds for label, seconds, _share in stage_rows
        },
        "train_bpe_stats": stats,
        "part_a_answer": part_a_answer,
        "part_b_answer": part_b_answer,
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# TinyStories BPE experiment",
        "",
        f"- Data: `{DATA_PATH}`",
        f"- Output directory: `{OUTPUT_DIR}`",
        f"- Target vocab size: `{VOCAB_SIZE}`",
        f"- Special token present: `{special_token_present}`",
        f"- Peak process-tree RSS (sampled): `{kib_to_gib(peak_rss_kib):.2f} GiB`",
        f"- Longest token: `{repr(longest_token)}` decoded as `{longest_text}`",
        "",
        "## Suggested answer for (a)",
        part_a_answer,
        "",
        "## Suggested answer for (b)",
        part_b_answer,
        "",
        "## Stage timings",
    ]
    lines.extend(
        f"- {label}: {seconds:.1f}s ({share:.1f}% of total)"
        for label, seconds, share in stage_rows
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def print_terminal_summary(
    stats: TrainBPEStats,
    peak_rss_kib: int,
    special_token_present: bool,
    longest_token: bytes,
    stage_rows: list[tuple[str, float, float]],
    part_a_answer: str,
    part_b_answer: str,
) -> None:
    longest_text = render_token_text(longest_token)
    total_s = float(stats["total_s"])
    input_bytes = int(stats["input_bytes"])
    input_gib = input_bytes / (1024**3)
    bottleneck_label, bottleneck_s, bottleneck_share = max(stage_rows, key=lambda row: row[1])

    print("\n" + "=" * 72)
    print("TinyStories BPE Experiment")
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
        f"{kib_to_gib(peak_rss_kib):.2f} GiB ({peak_rss_kib} KiB)"
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

    print("\nLongest token")
    print(f"- Bytes repr: {longest_token!r}")
    print(f"- Decoded text: {longest_text!r}")
    print(f"- Length: {len(longest_token)} bytes")

    print("\nStage timings")
    for label, seconds, share in stage_rows:
        print(f"- {label}: {seconds:.1f}s ({share:.1f}% of total)")
    print(
        f"- Bottleneck: {bottleneck_label} "
        f"at {bottleneck_s:.1f}s ({bottleneck_share:.1f}% of total)"
    )

    print("\nHomework-ready answer for 2.5(a)")
    print(part_a_answer)
    print("\nHomework-ready answer for 2.5(b)")
    print(part_b_answer)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE on {DATA_PATH} ...")
    rss_monitor = ProcessTreeRSSMonitor(os.getpid())
    rss_monitor.start()
    try:
        vocab, merges, stats = train_bpe(
            input_path=str(DATA_PATH),
            vocab_size=VOCAB_SIZE,
            special_tokens=[SPECIAL_TOKEN],
            collect_stats=True,
        )
    finally:
        peak_rss_kib = rss_monitor.stop()

    longest_token = get_longest_non_special_token(vocab)
    longest_text = render_token_text(longest_token)
    special_token_present = SPECIAL_TOKEN_BYTES in vocab.values()
    stage_rows = build_stage_rows(stats)
    part_a_answer = build_part_a_answer(float(stats["total_s"]), peak_rss_kib, longest_token)
    part_b_answer = build_part_b_answer(stage_rows)

    write_serialized_outputs(vocab, merges)
    write_report(
        stats=stats,
        peak_rss_kib=peak_rss_kib,
        special_token_present=special_token_present,
        longest_token=longest_token,
        part_a_answer=part_a_answer,
        part_b_answer=part_b_answer,
        stage_rows=stage_rows,
    )
    print_terminal_summary(
        stats=stats,
        peak_rss_kib=peak_rss_kib,
        special_token_present=special_token_present,
        longest_token=longest_token,
        stage_rows=stage_rows,
        part_a_answer=part_a_answer,
        part_b_answer=part_b_answer,
    )


if __name__ == "__main__":
    main()
