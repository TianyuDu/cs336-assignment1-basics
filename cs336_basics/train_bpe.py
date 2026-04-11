import multiprocessing as mp
import os
import re
from collections import Counter

import regex

GPT2_PRETOKENIZER_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# globals for worker processes (avoid pickling the corpus)
_WORKER_RAW_BYTES = b""
_WORKER_SPECIAL_TOKENS: frozenset[bytes] = frozenset()
_WORKER_SPLIT_RE: re.Pattern[bytes] | None = None


def _init_worker(raw_bytes: bytes, special_tokens: tuple[bytes, ...]) -> None:
    global _WORKER_RAW_BYTES, _WORKER_SPECIAL_TOKENS, _WORKER_SPLIT_RE
    _WORKER_RAW_BYTES = raw_bytes
    _WORKER_SPECIAL_TOKENS = frozenset(special_tokens)
    if special_tokens:
        pattern = b"(" + b"|".join(re.escape(tok) for tok in special_tokens) + b")"
        _WORKER_SPLIT_RE = re.compile(pattern)
    else:
        _WORKER_SPLIT_RE = None


def _pretokenize_chunk(chunk_range: tuple[int, int]) -> Counter[bytes]:
    start, end = chunk_range
    chunk = _WORKER_RAW_BYTES[start:end]
    counts: Counter[bytes] = Counter()

    parts = [chunk] if _WORKER_SPLIT_RE is None else _WORKER_SPLIT_RE.split(chunk)
    for part in parts:
        if not part or part in _WORKER_SPECIAL_TOKENS:
            continue
        text = part.decode("utf-8", errors="ignore")
        for match in GPT2_PRETOKENIZER_PATTERN.finditer(text):
            counts[match.group(0).encode("utf-8")] += 1

    return counts


def _find_chunk_boundaries(
    raw_bytes: bytes, desired_num_chunks: int, special_tokens: tuple[bytes, ...]
) -> list[int]:
    """Boundaries always land on a special token so chunks are independently pre-tokenizable."""
    if not raw_bytes:
        return [0]
    if desired_num_chunks <= 1 or not special_tokens:
        return [0, len(raw_bytes)]

    pattern = b"(" + b"|".join(re.escape(tok) for tok in special_tokens) + b")"
    split_re = re.compile(pattern)

    file_size = len(raw_bytes)
    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [0]

    for i in range(1, desired_num_chunks):
        guess = i * chunk_size
        match = split_re.search(raw_bytes, guess)
        if match is None:
            break
        if match.start() > boundaries[-1]:
            boundaries.append(match.start())

    boundaries.append(file_size)
    return boundaries


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens_bytes = tuple(tok.encode("utf-8") for tok in special_tokens)

    with open(input_path, "rb") as f:
        raw_bytes = f.read()

    # parallel pre-tokenization
    num_chunks = os.cpu_count() or 1
    chunk_boundaries = _find_chunk_boundaries(raw_bytes, num_chunks, special_tokens_bytes)
    chunk_ranges = [
        (start, end)
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
        if start < end
    ]

    _init_worker(raw_bytes, special_tokens_bytes)
    pretoken_counts: Counter[bytes] = Counter()

    if len(chunk_ranges) <= 1:
        if chunk_ranges:
            pretoken_counts = _pretokenize_chunk(chunk_ranges[0])
    else:
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else None)
        with ctx.Pool(
            processes=min(len(chunk_ranges), num_chunks),
            initializer=_init_worker,
            initargs=(raw_bytes, special_tokens_bytes),
        ) as pool:
            for chunk_counts in pool.imap_unordered(_pretokenize_chunk, chunk_ranges):
                pretoken_counts.update(chunk_counts)

    # convert pretokens to byte-tuple sequences with their counts
    token_counts: Counter[tuple[bytes, ...]] = Counter()
    for pretoken, count in pretoken_counts.items():
        token_counts[tuple(bytes([b]) for b in pretoken)] += count

    # vocab: special tokens first, then 256 byte values
    vocab: dict[int, bytes] = {}
    next_id = 0
    for tok in special_tokens_bytes:
        vocab[next_id] = tok
        next_id += 1
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    # BPE merge loop with incremental pair count updates
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for seq, count in token_counts.items():
        for left, right in zip(seq, seq[1:]):
            pair_counts[(left, right)] += count

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[next_id] = merged_token
        next_id += 1

        new_token_counts: Counter[tuple[bytes, ...]] = Counter()
        for seq, count in token_counts.items():
            has_pair = False
            for i in range(len(seq) - 1):
                if seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                    has_pair = True
                    break
            if not has_pair:
                new_token_counts[seq] += count
                continue

            new_seq: list[bytes] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                    if new_seq:
                        pair_counts[(new_seq[-1], best_pair[0])] -= count
                    if i + 2 < len(seq):
                        pair_counts[(best_pair[1], seq[i + 2])] -= count
                    if new_seq:
                        pair_counts[(new_seq[-1], merged_token)] += count
                    new_seq.append(merged_token)
                    i += 2
                    if i < len(seq):
                        pair_counts[(merged_token, seq[i])] += count
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_token_counts[tuple(new_seq)] += count

        del pair_counts[best_pair]
        pair_counts = Counter({k: v for k, v in pair_counts.items() if v > 0})
        token_counts = new_token_counts

    return vocab, merges
