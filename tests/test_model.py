"""Tests of the shipped model rewrite against the real sentence-transformers model (FP32, CPU, no Core AI).
Run with the build venv: `.venv-build/bin/python -m pytest -q tests`."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from eg2_ane import model as M  # noqa: E402

LONG = ("The Neural Engine is a matrix accelerator in Apple silicon. " * 60).strip()  # about 660 tokens


@pytest.fixture(scope="module")
def st():
    return M.load_sentence_transformer()


@pytest.fixture(scope="module")
def table(st):
    return M.embed_table(st)


def run(st, table, texts, seq, **kw):
    pairs = [M.tokenize(st.tokenizer, t, seq) for t in texts]
    x = torch.from_numpy(np.concatenate([M.host_embeds(table, i) for i, _ in pairs]).astype(np.float32))
    mask = torch.from_numpy(np.concatenate([m for _, m in pairs]))
    with torch.no_grad():
        return M.EG2TextANE(st, seq, **kw).eval()(x, mask).numpy()


def test_tokenize_matches_sentence_transformers(st):
    st.max_seq_length = 512
    for t in (LONG, "  a leading and trailing space  ", "short"):
        ids, mask = M.tokenize(st.tokenizer, t, 512)
        ref = st.tokenize([t])["input_ids"][0].numpy()
        assert np.array_equal(ids[0, : int(mask.sum())], ref)


@pytest.mark.parametrize("seq", [128, 512])
def test_ane_layout_equals_sentence_transformers(st, table, seq):
    st.max_seq_length = seq
    texts = [M.PROMPTS["query"] + "Which planet is known as the red planet?", M.PROMPTS["document"] + LONG]
    ref = st.encode(texts, normalize_embeddings=True)
    e = run(st, table, texts, seq)
    for a, b in zip(e, ref):
        assert float(a @ b) > 0.99999


def test_sigmoid_gelu_is_the_same_function():
    x = torch.linspace(-12, 12, 20001, dtype=torch.float64)
    assert torch.allclose(M.gelu(x, "tanh"), M.gelu(x, "sigmoid2"), atol=1e-12)


def test_fp16_scales_are_exact_in_fp32(st, table, monkeypatch):
    """PLE_IN, SQ and the output scales change the graph, not the math."""
    text = [M.PROMPTS["sts"] + "A man is playing a guitar on a stage."]
    a = run(st, table, text, 128)
    for name, value in (("PLE_IN", 1.0), ("SQ", 1.0), ("ATTN_OUT", 1.0), ("FF_OUT", 1.0), ("PLE_OUT", 1.0)):
        monkeypatch.setattr(M, name, value)
    b = run(st, table, text, 128)
    assert float((a * b).sum()) > 0.9999999


def test_band_is_used_only_above_the_window(st):
    assert not M.EG2TextANE(st, 512, layers=1).use_band
    assert M.EG2TextANE(st, 640, layers=1).use_band


def test_reference_file_matches_model(st, table):
    ref = Path(__file__).resolve().parents[1] / "reference" / "eg2_ref.npz"
    if not ref.exists():
        pytest.skip("reference/eg2_ref.npz not made")
    d = np.load(ref)
    x = torch.from_numpy(M.host_embeds(table, d["ids_128"][:4]).astype(np.float32))
    with torch.no_grad():
        e = M.EG2TextANE(st, 128).eval()(x, torch.from_numpy(d["mask_128"][:4])).numpy()
    assert ((e * d["ref_128"][:4]).sum(1) > 0.99999).all()
