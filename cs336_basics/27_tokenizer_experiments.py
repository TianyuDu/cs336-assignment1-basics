"""
Sample TinyStories/OpenWebText documents and measure tokenizer compression.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SPECIAL_TOKEN = "<|endoftext|>"
SAMPLE_SIZE = 10
SEED = 0
CHUNK_SIZE_BYTES = 4 * 1024 * 1024
OUTPUT_DIR = ROOT / "data" / "tokenizer_experiments"
TOP_DOCUMENTS_PER_CORPUS = 100
CORPORA = {
    "tinystories": {
        "label": "TinyStories",
        "path": ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        "seed_offset": 0,
    },
    "openwebtext": {
        "label": "OpenWebText",
        "path": ROOT / "data" / "owt_train.txt",
        "seed_offset": 1,
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

    special_token_bytes = SPECIAL_TOKEN.encode("utf-8")
    for corpus in CORPORA.values():
        if not corpus["path"].exists():
            raise FileNotFoundError(f"Missing corpus file: {corpus['path']}")
    for tokenizer in TOKENIZERS.values():
        if not tokenizer["vocab_path"].exists():
            raise FileNotFoundError(f"Missing vocab file: {tokenizer['vocab_path']}")
        if not tokenizer["merges_path"].exists():
            raise FileNotFoundError(f"Missing merges file: {tokenizer['merges_path']}")

    # Load the previously trained tokenizers once.
    tokenizers: dict[str, Tokenizer] = {}
    for tokenizer_name, tokenizer in TOKENIZERS.items():
        tokenizers[tokenizer_name] = Tokenizer.from_files(
            vocab_filepath=str(tokenizer["vocab_path"]),
            merges_filepath=str(tokenizer["merges_path"]),
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
        corpus_path = Path(corpus["path"])
        rng = random.Random(seed + int(corpus["seed_offset"]))
        candidate_documents: list[dict[str, object]] = []
        documents_seen = 0
        buffer = b""
        separator_len = len(special_token_bytes)

        with corpus_path.open("rb") as f:
            while documents_seen < TOP_DOCUMENTS_PER_CORPUS:
                chunk = f.read(chunk_size_bytes)
                if not chunk:
                    if buffer.strip():
                        buffer += special_token_bytes
                    else:
                        break
                else:
                    buffer += chunk

                while True:
                    separator_index = buffer.find(special_token_bytes)
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


if __name__ == "__main__":
    run_part_abc_report()
