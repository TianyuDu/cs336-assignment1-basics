"""
Problem 2.5: Train BPE on TinyStories and profile it.

Results:
- Time: ~115s for the merge loop, ~388s total (including parallel pre-tokenization)
- Peak memory: 2.39 GB
- Longest token: b' accomplishment' (15 bytes) — a common word in children's stories
- Bottleneck: pre-tokenization (regex matching over ~2.1M documents) dominates wall-clock time,
  followed by the BPE merge loop (heappop operations for finding the max-frequency pair).
"""

import cProfile
import json
import time
import tracemalloc
from pathlib import Path

from cs336_basics.train_bpe import train_bpe

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "TinyStoriesV2-GPT4-train.txt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "bpe_tinystories"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE on {DATA_PATH} ...")
    tracemalloc.start()
    start = time.time()

    vocab, merges = train_bpe(
        input_path=str(DATA_PATH),
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.time() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Time: {elapsed:.1f}s")
    print(f"Peak memory: {peak_memory / 1e9:.2f} GB")
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges: {len(merges)}")

    # longest token (excluding special tokens)
    longest = max(
        (v for v in vocab.values() if v != b"<|endoftext|>"),
        key=len,
    )
    print(f"Longest token: {longest} (len={len(longest)})")

    # serialize
    vocab_serializable = {str(k): list(v) for k, v in vocab.items()}
    with open(OUTPUT_DIR / "vocab.json", "w") as f:
        json.dump(vocab_serializable, f)
    with open(OUTPUT_DIR / "merges.txt", "w") as f:
        for t1, t2 in merges:
            f.write(f"{t1!r} {t2!r}\n")
    print(f"Saved vocab and merges to {OUTPUT_DIR}")


def profile():
    print("\nProfiling (top 20 by cumulative time)...")
    profiler = cProfile.Profile()
    profiler.enable()
    train_bpe(str(DATA_PATH), 10000, ["<|endoftext|>"])
    profiler.disable()
    profiler.print_stats(sort="tottime", lines=20)


if __name__ == "__main__":
    main()
    profile()
