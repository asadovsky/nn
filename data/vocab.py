"""Defines Vocab."""

from data.dataset_gen import DatasetGen

PAD = "<pad>"
UNK = "<unk>"


def _proc_tokens(tokens: list[str], token2id: dict[str, int]) -> None:
    for token in tokens:
        if token not in token2id:
            token2id[token] = len(token2id)


def _mk_id2token(token2id: dict[str, int]) -> list[str]:
    id2token = [""] * len(token2id)
    for token, i in token2id.items():
        id2token[i] = token
    return id2token


class Vocab:
    """A vocab representation."""

    def __init__(self) -> None:
        self.word2id: dict[str, int] = {PAD: 0, UNK: 1}
        self.char2id: dict[str, int] = {PAD: 0, UNK: 1}
        self.tag2id: dict[str, int] = {PAD: 0, UNK: 1}
        self.intent2id: dict[str, int] = {UNK: 0}

        self.id2word: list[str] = []
        self.id2char: list[str] = []
        self.id2tag: list[str] = []
        self.id2intent: list[str] = []

    def add_dataset(self, dataset_gen: DatasetGen) -> None:
        """Adds words, tags, and intents from the given dataset."""
        for words, tags, intent in dataset_gen:
            _proc_tokens(words, self.word2id)
            for word in words:
                _proc_tokens(list(word), self.char2id)
            _proc_tokens(tags, self.tag2id)
            _proc_tokens([intent], self.intent2id)

        self.id2word = _mk_id2token(self.word2id)
        self.id2char = _mk_id2token(self.char2id)
        self.id2tag = _mk_id2token(self.tag2id)
        self.id2intent = _mk_id2token(self.intent2id)

    def add_words(self, words: list[str]) -> None:
        """Adds words from the given list."""
        _proc_tokens(words, self.word2id)
        for word in words:
            _proc_tokens(list(word), self.char2id)

        self.id2word = _mk_id2token(self.word2id)
        self.id2char = _mk_id2token(self.char2id)
