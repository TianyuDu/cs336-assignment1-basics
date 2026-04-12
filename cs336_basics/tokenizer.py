from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Iterator

import regex


GPT2_PRETOKENIZER_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
SINGLE_BYTE_TOKENS = tuple(bytes([byte]) for byte in range(256))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Keep special tokens as single vocabulary items.
        if self.special_tokens:
            vocab_values = set(self.vocab.values())
            next_token_id = max(self.vocab, default=-1) + 1
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in vocab_values:
                    self.vocab[next_token_id] = special_token_bytes
                    vocab_values.add(special_token_bytes)
                    next_token_id += 1

        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_ranks = {merge: rank for rank, merge in enumerate(merges)}
        self.special_tokens_set = set(self.special_tokens)
        self.special_token_to_id = {
            special_token: self.token_to_id[special_token.encode("utf-8")]
            for special_token in self.special_tokens
        }
        self.special_token_pattern = None
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_token_pattern = re.compile(
                "(" + "|".join(re.escape(token) for token in sorted_special_tokens) + ")"
            )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        # the tokenizer training scripts
        # save vocab as {token_id: [byte0, byte1, ...]} and merges as lines like:
        # b' ' b't'
        bytes_literal_pair_re = re.compile(
            r"""^\s*(b'(?:\\.|[^'])*'|b"(?:\\.|[^"])*")\s+(b'(?:\\.|[^'])*'|b"(?:\\.|[^"])*")\s*$"""
        )

        with open(vocab_filepath) as vocab_file:
            serialized_vocab = json.load(vocab_file)

        vocab = {
            int(token_id): bytes(token_bytes)
            for token_id, token_bytes in serialized_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath) as merges_file:
            for line in merges_file:
                cleaned_line = line.strip()
                if not cleaned_line or cleaned_line.startswith("#"):
                    continue

                match = bytes_literal_pair_re.match(cleaned_line)
                if match is None:
                    raise ValueError(f"Invalid merge line in {merges_filepath}: {cleaned_line}")

                left = ast.literal_eval(match.group(1))
                right = ast.literal_eval(match.group(2))
                merges.append((left, right))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        # Split around special tokens first so they stay intact.
        text_parts = [text]
        if self.special_token_pattern is not None:
            text_parts = self.special_token_pattern.split(text)

        for part in text_parts:
            if not part:
                continue

            special_token_id = self.special_token_to_id.get(part)
            if special_token_id is not None:
                ids.append(special_token_id)
                continue

            self._encode_plain_text(part, ids)

        return ids

    def _encode_plain_text(self, text: str, ids: list[int]) -> None:
        merge_ranks = self.merge_ranks
        token_to_id = self.token_to_id

        # Match the same GPT-2-style pre-tokens used during BPE training.
        for match in GPT2_PRETOKENIZER_PATTERN.finditer(text):
            token_bytes = match.group(0).encode("utf-8")
            tokens = [SINGLE_BYTE_TOKENS[byte] for byte in token_bytes]

            # Repeatedly apply the earliest merge that is present.
            while len(tokens) > 1:
                best_pair: tuple[bytes, bytes] | None = None
                best_rank = None

                prev_token = tokens[0]
                for token in tokens[1:]:
                    pair = (prev_token, token)
                    rank = merge_ranks.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_pair = pair
                        best_rank = rank
                    prev_token = token

                if best_pair is None:
                    break

                merged_token = best_pair[0] + best_pair[1]
                merged_tokens: list[bytes] = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        merged_tokens.append(merged_token)
                        i += 2
                    else:
                        merged_tokens.append(tokens[i])
                        i += 1
                tokens = merged_tokens

            ids.extend(token_to_id[token] for token in tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        all_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return all_bytes.decode("utf-8", errors="replace")