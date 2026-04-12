from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Iterator


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        # the tokenzer traininig scripts
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

        # add addiitonal user-provided special tokens to the vocab.
        if special_tokens:
            vocab_values = set(vocab.values())
            next_token_id = max(vocab, default=-1) + 1
            for special_token in special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in vocab_values:
                    vocab[next_token_id] = special_token_bytes
                    vocab_values.add(special_token_bytes)
                    next_token_id += 1

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError