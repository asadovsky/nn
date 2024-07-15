"""Generates pruned GloVe embedding files."""

from absl import app

from data.atis import mk_dataset_gen
from data.vocab import Vocab
from modeling import embedding_util


def main(unused_argv: list) -> None:
    v = Vocab()
    v.add_dataset(mk_dataset_gen("train"))
    v.add_dataset(mk_dataset_gen("val"))
    v.add_dataset(mk_dataset_gen("test"))
    for dim in [50, 100, 200, 300]:
        embedding_util.prune_glove(dim, v.id2word)


if __name__ == "__main__":
    app.run(main)
