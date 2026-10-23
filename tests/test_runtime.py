"""Tests of the runtime's pure parts (no model files needed): batch planning, padding, order, prompts."""
import numpy as np

from eg2_ane.runtime import Embedder


class Stub(Embedder):
    """An Embedder without a model: run() returns the row's second token id in column 0, so the tests can see which
    rows went where."""

    def __init__(self, entries):
        self.entries = entries
        self.max_seq = max(s for s, _ in entries.values())
        self.calls = []

    def ids(self, text):
        return [2] + [int(w) for w in text.split()] + [1]

    def run(self, name, id_rows):
        assert len(id_rows) <= self.entries[name][1]
        self.calls.append((name, len(id_rows)))
        out = np.zeros((len(id_rows), 768), np.float32)
        out[:, 0] = [r[1] for r in id_rows]
        return out


ENTRIES = {"s128b1": (128, 1), "s128b16": (128, 16), "s512b1": (512, 1), "s512b4": (512, 4)}


def test_plan_uses_batch_1_for_one_text_and_the_batch_for_many():
    e = Stub(ENTRIES)
    assert e.plan(10, 1) == [("s128b1", 1)]
    assert e.plan(10, 16) == [("s128b16", 16)]
    assert e.plan(300, 1) == [("s512b1", 1)]
    assert e.plan(300, 4) == [("s512b4", 4)]


def test_plan_covers_every_text_once():
    e = Stub(ENTRIES)
    for n in range(1, 60):
        for seq in (128, 512):
            p = e.plan(seq, n)
            assert sum(t for _, t in p) == n
            assert all(t <= ENTRIES[k][1] and ENTRIES[k][0] == seq for k, t in p)


def test_embed_groups_by_length_and_keeps_order():
    e = Stub(ENTRIES)
    texts = [" ".join(["7"] * 200)] + [str(i + 10) for i in range(20)]
    out = e.embed(texts)
    assert out[0, 0] == 7 and list(out[1:, 0]) == [i + 10 for i in range(20)]
    assert ("s512b1", 1) in e.calls and ("s128b16", 16) in e.calls


def test_prompt_names_and_literal_prefix():
    e = Stub(ENTRIES)
    seen = []
    e.ids = lambda t: seen.append(t) or [2, 5, 1]
    e.embed(["x"], prompt="query")
    e.embed(["x"], prompt="custom: ")
    assert seen == ["task: search result | query: x", "custom: x"]
