"""
Sample TinyStories/OpenWebText documents, measure tokenizer compression,
and serialize tokenized corpora for later language-model training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_TOKEN_BYTES = SPECIAL_TOKEN.encode("utf-8")
SAMPLE_SIZE = 10
SEED = 0
CHUNK_SIZE_BYTES = 4 * 1024 * 1024
WRITE_BUFFER_TOKENS = 1_000_000
TOKEN_ID_DTYPE = np.uint16
OUTPUT_DIR = ROOT / "data" / "tokenizer_experiments"
ENCODED_OUTPUT_DIR = OUTPUT_DIR / "encoded"
TOP_DOCUMENTS_PER_CORPUS = 100
CORPORA = {
    "tinystories": {
        "label": "TinyStories",
        "train_path": ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        "valid_path": ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt",
        "seed_offset": 0,
        "tokenizer_name": "tinystories_10k",
    },
    "openwebtext": {
        "label": "OpenWebText",
        "train_path": ROOT / "data" / "owt_train.txt",
        "valid_path": ROOT / "data" / "owt_valid.txt",
        "seed_offset": 1,
        "tokenizer_name": "openwebtext_32k",
    },
}
TOKENIZERS = {
    "tinystories_10k": {
        "label": "TinyStories tokenizer (10K)",
        "vocab_path": ROOT / "data" / "bpe_tinystories" / "vocab.json",
        "merges_path": ROOT / "data" / "bpe_tinystories" / "merges.txt",
    },
    "openwebtext_32k": {
        "label": "OpenWebText tokenizer (32K)",
        "vocab_path": ROOT / "data" / "bpe_owt" / "vocab.json",
        "merges_path": ROOT / "data" / "bpe_owt" / "merges.txt",
    },
}


def run_part_abc_report(
    sample_size: int = SAMPLE_SIZE,
    seed: int = SEED,
    output_dir: Path = OUTPUT_DIR,
    chunk_size_bytes: int = CHUNK_SIZE_BYTES,
) -> dict[str, object]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive.")

    tokenizers: dict[str, Tokenizer] = {}
    for corpus in CORPORA.values():
        corpus_path = Path(corpus["train_path"])
        if not corpus_path.exists():
            raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    # Load the saved tokenizers once so we can reuse them across both corpora.
    for tokenizer_name, tokenizer in TOKENIZERS.items():
        vocab_path = Path(tokenizer["vocab_path"])
        merges_path = Path(tokenizer["merges_path"])
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
        if not merges_path.exists():
            raise FileNotFoundError(f"Missing merges file: {merges_path}")
        tokenizers[tokenizer_name] = Tokenizer.from_files(
            vocab_filepath=str(vocab_path),
            merges_filepath=str(merges_path),
            special_tokens=[SPECIAL_TOKEN],
        )

    compression_totals = {
        tokenizer_name: {"bytes": 0, "tokens": 0, "by_corpus": {}}
        for tokenizer_name in TOKENIZERS
    }
    tokenize_time_seconds = {tokenizer_name: 0.0 for tokenizer_name in TOKENIZERS}
    report: dict[str, object] = {
        "sample_size_per_corpus": sample_size,
        "seed": seed,
        "special_token": SPECIAL_TOKEN,
        "corpora": {},
        "compression_ratios_bytes_per_token": {},
        "artifacts": {},
    }

    # Stream each corpus, split on <|endoftext|>, and keep only the first
    # TOP_DOCUMENTS_PER_CORPUS docs in memory. Sample from those documents.
    for corpus_name, corpus in CORPORA.items():
        corpus_label = str(corpus["label"])
        corpus_path = Path(corpus["train_path"])
        rng = random.Random(seed + int(corpus["seed_offset"]))
        candidate_documents: list[dict[str, object]] = []
        documents_seen = 0
        buffer = b""
        separator_len = len(SPECIAL_TOKEN_BYTES)

        with corpus_path.open("rb") as f:
            while documents_seen < TOP_DOCUMENTS_PER_CORPUS:
                chunk = f.read(chunk_size_bytes)
                if not chunk:
                    if buffer.strip():
                        buffer += SPECIAL_TOKEN_BYTES
                    else:
                        break
                else:
                    buffer += chunk

                while True:
                    separator_index = buffer.find(SPECIAL_TOKEN_BYTES)
                    if separator_index == -1:
                        break

                    document_bytes = buffer[:separator_index]
                    buffer = buffer[separator_index + separator_len :]
                    if not document_bytes.strip():
                        continue

                    documents_seen += 1
                    text = document_bytes.decode("utf-8")
                    preview = " ".join(text.split())
                    document = {
                        "document_index": documents_seen,
                        "utf8_bytes": len(document_bytes),
                        "text": text,
                        "text_preview": preview if len(preview) <= 120 else preview[:117] + "...",
                        "encodings": {},
                    }
                    candidate_documents.append(document)

                if documents_seen >= TOP_DOCUMENTS_PER_CORPUS:
                    break

        if documents_seen < sample_size:
            raise ValueError(
                f"Corpus {corpus_path} only contained {documents_seen} non-empty documents in top {TOP_DOCUMENTS_PER_CORPUS}."
            )

        # Randomly sample 10 docs from the first TOP_DOCUMENTS_PER_CORPUS docs.
        if len(candidate_documents) < sample_size:
            raise ValueError(
                f"Need at least {sample_size} docs in top {TOP_DOCUMENTS_PER_CORPUS},"
                f" but found {len(candidate_documents)} in {corpus_path}."
            )
        sample_indices = rng.sample(range(len(candidate_documents)), k=sample_size)
        sampled_documents = [
            candidate_documents[idx]
            for idx in sorted(sample_indices)
        ]

        total_sampled_bytes = sum(int(document["utf8_bytes"]) for document in sampled_documents)

        # Encode the sampled documents with both tokenizers and compute
        # bytes/token on each corpus plus an overall combined score.
        for tokenizer_name, tokenizer in tokenizers.items():
            total_tokens = 0
            t0 = time.perf_counter()

            for document in sampled_documents:
                text = str(document["text"])
                utf8_bytes = int(document["utf8_bytes"])
                ids = tokenizer.encode(text)
                if not ids:
                    raise ValueError(
                        f"Tokenizer {TOKENIZERS[tokenizer_name]['label']} "
                        f"produced zero tokens for {corpus_label}."
                    )

                document["encodings"][tokenizer_name] = {
                    "num_tokens": len(ids),
                    "bytes_per_token": utf8_bytes / len(ids),
                    "ids": ids,
                }
                total_tokens += len(ids)

            tokenize_time_seconds[tokenizer_name] += time.perf_counter() - t0
            compression_totals[tokenizer_name]["bytes"] += total_sampled_bytes
            compression_totals[tokenizer_name]["tokens"] += total_tokens
            compression_totals[tokenizer_name]["by_corpus"][corpus_name] = (
                total_sampled_bytes / total_tokens
            )

        for document in sampled_documents:
            del document["text"]

        report["corpora"][corpus_name] = {
            "label": corpus_label,
            "data_path": str(corpus_path),
            "documents_seen": documents_seen,
            "sampled_documents": sampled_documents,
        }

    for tokenizer_name, totals in compression_totals.items():
        total_tokens = compression_totals[tokenizer_name]["tokens"]
        if total_tokens == 0:
            raise ValueError(f"Tokenizer {TOKENIZERS[tokenizer_name]['label']} produced zero total tokens.")

        total_bytes = totals["bytes"]
        total_secs = tokenize_time_seconds[tokenizer_name]
        report["compression_ratios_bytes_per_token"][tokenizer_name] = {
            "label": TOKENIZERS[tokenizer_name]["label"],
            "by_corpus": totals["by_corpus"],
            "overall": totals["bytes"] / total_tokens,
            "sample_throughput_bytes_per_s": 0.0 if total_secs == 0 else total_bytes / total_secs,
            "sample_seconds": total_secs,
        }

    # Save a detailed JSON report and a compact text summary for the writeup.
    output_dir.mkdir(parents=True, exist_ok=True)
    json_report_path = output_dir / "part_a_report.json"
    markdown_report_path = output_dir / "part_a_report.md"
    report["artifacts"] = {
        "json_report_path": str(json_report_path),
        "markdown_report_path": str(markdown_report_path),
    }
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Report-only values for parts (b) and (c).
    ts_scores = report["compression_ratios_bytes_per_token"]["tinystories_10k"]
    owt_scores = report["compression_ratios_bytes_per_token"]["openwebtext_32k"]
    ts_on_owt = ts_scores["by_corpus"]["openwebtext"]
    ts_on_ts = ts_scores["by_corpus"]["tinystories"]

    owt_tokenizer_throughput = owt_scores["sample_throughput_bytes_per_s"]
    ts_tokenizer_throughput = ts_scores["sample_throughput_bytes_per_s"]
    pile_bytes = 825 * 1024**3
    owt_pile_hours = pile_bytes / owt_tokenizer_throughput / 3600 if owt_tokenizer_throughput else float("inf")
    ts_pile_hours = pile_bytes / ts_tokenizer_throughput / 3600 if ts_tokenizer_throughput else float("inf")

    summary_lines = [f"sample_size={sample_size} seed={seed}"]
    for corpus_name, corpus in CORPORA.items():
        sample_indices = ", ".join(
            str(document["document_index"])
            for document in report["corpora"][corpus_name]["sampled_documents"]
        )
        summary_lines.append(
            f"{corpus['label']} docs: {sample_indices} "
            f"(sampled from {report['corpora'][corpus_name]['documents_seen']} documents)"
        )
    for tokenizer_name, tokenizer in TOKENIZERS.items():
        scores = report["compression_ratios_bytes_per_token"][tokenizer_name]
        summary_lines.append(
            f"{tokenizer['label']} compression_ratio_bytes/token: "
            f"tinystories={scores['by_corpus']['tinystories']:.4f} "
            f"openwebtext={scores['by_corpus']['openwebtext']:.4f} "
            f"combined={scores['overall']:.4f} "
            f"sample_throughput_bytes_per_s={scores['sample_throughput_bytes_per_s']:.0f}B/s"
        )
    summary_lines.append(f"part_b_tinystories_on_owt_bytes_per_token={ts_on_owt:.6f}")
    summary_lines.append(f"part_b_tinystories_on_tinystories_bytes_per_token={ts_on_ts:.6f}")
    summary_lines.append(f"part_b_delta_bytes_per_token={ts_on_owt-ts_on_ts:.6f}")
    summary_lines.append(f"part_c_ts_bytes_per_second={ts_tokenizer_throughput:.0f}")
    summary_lines.append(f"part_c_ts_hours_for_825GB={ts_pile_hours:.2f}")
    summary_lines.append(f"part_c_owt_bytes_per_second={owt_tokenizer_throughput:.0f}")
    summary_lines.append(f"part_c_owt_hours_for_825GB={owt_pile_hours:.2f}")
    summary_lines.append(f"reports: {json_report_path} | {markdown_report_path}")
    markdown_report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))

    return report


def run_part_d_encoding(
    output_dir: Path = ENCODED_OUTPUT_DIR,
    chunk_size_bytes: int = CHUNK_SIZE_BYTES,
    write_buffer_tokens: int = WRITE_BUFFER_TOKENS,
) -> dict[str, object]:
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive.")
    if write_buffer_tokens <= 0:
        raise ValueError("write_buffer_tokens must be positive.")

    token_dtype = np.dtype(TOKEN_ID_DTYPE)
    max_token_value = int(np.iinfo(token_dtype).max)
    separator_len = len(SPECIAL_TOKEN_BYTES)

    tokenizers: dict[str, Tokenizer] = {}
    for tokenizer_name, tokenizer in TOKENIZERS.items():
        vocab_path = Path(tokenizer["vocab_path"])
        merges_path = Path(tokenizer["merges_path"])
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
        if not merges_path.exists():
            raise FileNotFoundError(f"Missing merges file: {merges_path}")

        loaded_tokenizer = Tokenizer.from_files(
            vocab_filepath=str(vocab_path),
            merges_filepath=str(merges_path),
            special_tokens=[SPECIAL_TOKEN],
        )
        if max(loaded_tokenizer.vocab) > max_token_value:
            raise ValueError(
                f"Tokenizer {TOKENIZERS[tokenizer_name]['label']} does not fit in {token_dtype.name}."
            )
        tokenizers[tokenizer_name] = loaded_tokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "special_token": SPECIAL_TOKEN,
        "token_id_dtype": token_dtype.name,
        "corpora": {},
        "artifacts": {},
    }
    summary_lines = [
        f"token_id_dtype={token_dtype.name}",
        f"special_token={SPECIAL_TOKEN}",
    ]

    for corpus_name, corpus in CORPORA.items():
        corpus_label = str(corpus["label"])
        tokenizer_name = str(corpus["tokenizer_name"])
        tokenizer = tokenizers[tokenizer_name]
        special_token_id = tokenizer.token_to_id[SPECIAL_TOKEN_BYTES]
        corpus_output_dir = output_dir / tokenizer_name
        corpus_output_dir.mkdir(parents=True, exist_ok=True)
        split_artifacts: dict[str, object] = {}

        for split_name in ("train", "valid"):
            dataset_path = Path(corpus[f"{split_name}_path"])
            if not dataset_path.exists():
                raise FileNotFoundError(f"Missing corpus file: {dataset_path}")
            output_path = corpus_output_dir / f"{split_name}.npy"
            print(
                f"[part d] encoding {corpus_label} {split_name} split "
                f"with {TOKENIZERS[tokenizer_name]['label']} -> {output_path}"
            )

            t0 = time.perf_counter()

            # First count the tokens so we know the exact size of the final
            # uint16 NumPy array before writing it.
            num_tokens = 0
            buffer = b""
            with dataset_path.open("rb") as f:
                while True:
                    chunk = f.read(chunk_size_bytes)
                    if chunk:
                        buffer += chunk
                    else:
                        if buffer:
                            num_tokens += len(tokenizer.encode(buffer.decode("utf-8")))
                        break

                    while True:
                        separator_index = buffer.find(SPECIAL_TOKEN_BYTES)
                        if separator_index == -1:
                            break

                        segment_bytes = buffer[:separator_index]
                        buffer = buffer[separator_index + separator_len :]
                        if segment_bytes:
                            num_tokens += len(tokenizer.encode(segment_bytes.decode("utf-8")))
                        num_tokens += 1

            if num_tokens == 0:
                np.save(output_path, np.empty((0,), dtype=token_dtype))
            else:
                # Then tokenize again and stream the ids into the preallocated array.
                tokens = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=token_dtype,
                    shape=(num_tokens,),
                )
                write_index = 0
                buffer = b""
                token_buffer: list[int] = []

                with dataset_path.open("rb") as f:
                    while True:
                        chunk = f.read(chunk_size_bytes)
                        if chunk:
                            buffer += chunk
                        else:
                            if buffer:
                                token_buffer.extend(tokenizer.encode(buffer.decode("utf-8")))
                            break

                        while True:
                            separator_index = buffer.find(SPECIAL_TOKEN_BYTES)
                            if separator_index == -1:
                                break

                            segment_bytes = buffer[:separator_index]
                            buffer = buffer[separator_index + separator_len :]
                            if segment_bytes:
                                token_buffer.extend(tokenizer.encode(segment_bytes.decode("utf-8")))
                                if len(token_buffer) >= write_buffer_tokens:
                                    stop = write_index + len(token_buffer)
                                    tokens[write_index:stop] = np.asarray(token_buffer, dtype=token_dtype)
                                    write_index = stop
                                    token_buffer.clear()

                            token_buffer.append(special_token_id)
                            if len(token_buffer) >= write_buffer_tokens:
                                stop = write_index + len(token_buffer)
                                tokens[write_index:stop] = np.asarray(token_buffer, dtype=token_dtype)
                                write_index = stop
                                token_buffer.clear()

                    if token_buffer:
                        stop = write_index + len(token_buffer)
                        tokens[write_index:stop] = np.asarray(token_buffer, dtype=token_dtype)
                        write_index = stop

                if write_index != num_tokens:
                    raise RuntimeError(
                        f"Expected to write {num_tokens} tokens for {dataset_path}, wrote {write_index}."
                    )
                tokens.flush()
                del tokens

            elapsed = time.perf_counter() - t0
            source_bytes = dataset_path.stat().st_size
            split_artifacts[split_name] = {
                "input_path": str(dataset_path),
                "output_path": str(output_path),
                "num_tokens": num_tokens,
                "source_bytes": source_bytes,
                "seconds": elapsed,
            }
            summary_lines.append(
                f"{corpus_label} {split_name}: {num_tokens} tokens "
                f"from {source_bytes} bytes -> {output_path} ({elapsed:.2f}s)"
            )
            print(
                f"[part d] finished {corpus_label} {split_name}: "
                f"{num_tokens} tokens in {elapsed:.2f}s"
            )

        manifest["corpora"][corpus_name] = {
            "label": corpus_label,
            "tokenizer_name": tokenizer_name,
            "tokenizer_label": TOKENIZERS[tokenizer_name]["label"],
            "vocab_size": len(tokenizer.vocab),
            "splits": split_artifacts,
        }

    manifest_path = output_dir / "part_d_manifest.json"
    summary_path = output_dir / "part_d_report.md"
    manifest["artifacts"] = {
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_lines.append(f"manifest: {manifest_path}")
    summary_lines.append(f"summary: {summary_path}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tokenizer compression and dataset encoding experiments.")
    parser.add_argument(
        "--part",
        choices=["abc", "d", "all"],
        default="abc",
        help="Which experiment section to run.",
    )
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunk-size-bytes", type=int, default=CHUNK_SIZE_BYTES)
    parser.add_argument("--report-output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--encoding-output-dir", type=Path, default=ENCODED_OUTPUT_DIR)
    parser.add_argument("--write-buffer-tokens", type=int, default=WRITE_BUFFER_TOKENS)
    args = parser.parse_args()

    if args.part in {"abc", "all"}:
        run_part_abc_report(
            sample_size=args.sample_size,
            seed=args.seed,
            output_dir=args.report_output_dir,
            chunk_size_bytes=args.chunk_size_bytes,
        )

    if args.part in {"d", "all"}:
        run_part_d_encoding(
            output_dir=args.encoding_output_dir,
            chunk_size_bytes=args.chunk_size_bytes,
            write_buffer_tokens=args.write_buffer_tokens,
        )


if __name__ == "__main__":
    main()
