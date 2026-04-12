import importlib
import json

import numpy as np

from cs336_basics.tokenizer import Tokenizer

experiments = importlib.import_module("cs336_basics.27_tokenizer_experiments")


def test_run_part_d_encoding_writes_uint16_arrays(tmp_path, monkeypatch):
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text(
        json.dumps({token_id: [token_id] for token_id in range(256)}),
        encoding="utf-8",
    )
    merges_path.write_text("", encoding="utf-8")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tinystories_train = f"A{experiments.SPECIAL_TOKEN}{experiments.SPECIAL_TOKEN}B"
    tinystories_valid = "tiny valid"
    openwebtext_train = "open web text"
    openwebtext_valid = f"X{experiments.SPECIAL_TOKEN}Y"

    (data_dir / "ts_train.txt").write_text(tinystories_train, encoding="utf-8")
    (data_dir / "ts_valid.txt").write_text(tinystories_valid, encoding="utf-8")
    (data_dir / "owt_train.txt").write_text(openwebtext_train, encoding="utf-8")
    (data_dir / "owt_valid.txt").write_text(openwebtext_valid, encoding="utf-8")

    monkeypatch.setattr(
        experiments,
        "TOKENIZERS",
        {
            "tinystories_10k": {
                "label": "TinyStories tokenizer (10K)",
                "vocab_path": vocab_path,
                "merges_path": merges_path,
            },
            "openwebtext_32k": {
                "label": "OpenWebText tokenizer (32K)",
                "vocab_path": vocab_path,
                "merges_path": merges_path,
            },
        },
    )
    monkeypatch.setattr(
        experiments,
        "CORPORA",
        {
            "tinystories": {
                "label": "TinyStories",
                "train_path": data_dir / "ts_train.txt",
                "valid_path": data_dir / "ts_valid.txt",
                "seed_offset": 0,
                "tokenizer_name": "tinystories_10k",
            },
            "openwebtext": {
                "label": "OpenWebText",
                "train_path": data_dir / "owt_train.txt",
                "valid_path": data_dir / "owt_valid.txt",
                "seed_offset": 1,
                "tokenizer_name": "openwebtext_32k",
            },
        },
    )

    manifest = experiments.run_part_d_encoding(
        output_dir=tmp_path / "encoded",
        chunk_size_bytes=4,
        write_buffer_tokens=2,
    )
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=[experiments.SPECIAL_TOKEN],
    )

    assert manifest["token_id_dtype"] == "uint16"

    np.testing.assert_array_equal(
        np.load(tmp_path / "encoded" / "tinystories_10k" / "train.npy"),
        np.array(tokenizer.encode(tinystories_train), dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "encoded" / "tinystories_10k" / "valid.npy"),
        np.array(tokenizer.encode(tinystories_valid), dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "encoded" / "openwebtext_32k" / "train.npy"),
        np.array(tokenizer.encode(openwebtext_train), dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "encoded" / "openwebtext_32k" / "valid.npy"),
        np.array(tokenizer.encode(openwebtext_valid), dtype=np.uint16),
    )
