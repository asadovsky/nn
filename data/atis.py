"""Utilities for reading ATIS data."""

from data.dataset_gen import DatasetGen

_DATA_DIR = "resources/atis"
_DATA_FILENAMES = {
    "train": f"{_DATA_DIR}/atis-2.train.w-intent.iob",
    "val": f"{_DATA_DIR}/atis-2.dev.w-intent.iob",
    "test": f"{_DATA_DIR}/atis.test.w-intent.iob",
}


def mk_dataset_gen(
    split: str,
) -> DatasetGen:
    """Returns a (words, tags, intent) generator for the given file."""
    with open(_DATA_FILENAMES[split]) as f:
        for line in f:
            parts = line.split("\t")
            assert len(parts) == 2
            words = parts[0].strip().split()
            tags = parts[1].strip().split()
            assert len(words) == len(tags)
            # The final tag (for the final word, "EOS") is the intent name; replace it
            # with "O".
            intent = tags[-1]
            tags[-1] = "O"
            yield words, tags, intent
