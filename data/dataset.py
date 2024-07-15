"""Defines Dataset."""

from collections import Counter

from data.dataset_gen import DatasetGen
from data.vocab import UNK, Vocab


def _proc_tokens(tokens: list[str], token2id: dict[str, int]) -> list[int]:
    token_ids = []
    for token in tokens:
        if token not in token2id:
            token = UNK
        token_ids.append(token2id[token])
    return token_ids


class Dataset:
    """A dataset representation."""

    def __init__(self, vocab: Vocab, dataset_gen: DatasetGen) -> None:
        self._vocab: Vocab = vocab
        self.word_id_seqs: list[list[int]] = []  # word seqs, one per example
        self.char_id_seqs: list[  # char seqs, one per word per example
            list[list[int]]
        ] = []
        self.tag_id_seqs: list[list[int]] = []  # IOB tag seqs, one per example
        self.intent_ids: list[int] = []  # intents, one per example

        for words, tags, intent in dataset_gen:
            self.word_id_seqs.append(_proc_tokens(words, vocab.word2id))
            self.char_id_seqs.append(
                [_proc_tokens(list(word), vocab.char2id) for word in words]
            )
            self.tag_id_seqs.append(_proc_tokens(tags, vocab.tag2id))
            self.intent_ids.append(_proc_tokens([intent], vocab.intent2id)[0])

    def drop_rare_words(self, max_freq_to_drop: float) -> None:
        """Drops words whose frequency is <= `max_freq_to_drop`."""
        if max_freq_to_drop == 0:
            return
        word_ids = sum(self.word_id_seqs, [])
        to_drop = set()
        for word_id, count in Counter(word_ids).items():
            if count / len(word_ids) <= max_freq_to_drop:
                to_drop.add(word_id)
        print(f"Dropping {len(to_drop)} words")
        unk_word_id = self.word2id[UNK]
        for seq in self.word_id_seqs:
            for i, v in enumerate(seq):
                if v in to_drop:
                    seq[i] = unk_word_id

    @property
    def word2id(self) -> dict[str, int]:
        return self._vocab.word2id

    @property
    def char2id(self) -> dict[str, int]:
        return self._vocab.char2id

    @property
    def tag2id(self) -> dict[str, int]:
        return self._vocab.tag2id

    @property
    def intent2id(self) -> dict[str, int]:
        return self._vocab.intent2id

    @property
    def id2word(self) -> list[str]:
        return self._vocab.id2word

    @property
    def id2char(self) -> list[str]:
        return self._vocab.id2char

    @property
    def id2tag(self) -> list[str]:
        return self._vocab.id2tag

    @property
    def id2intent(self) -> list[str]:
        return self._vocab.id2intent
