"""Tests de ingest/embed.py con un modelo falso: comprueban qué texto se embebe, sin descargar nada."""
import json
import sys
import types

import numpy as np

from ingest import embed
from ingest.chunk import MODEL_NAME


class FakeModel:
    max_seq_length = 128

    def __init__(self, name=None):
        self.seen = []

    def get_embedding_dimension(self):
        return 4

    def encode(self, texts, normalize_embeddings=True, **kw):
        self.seen += list(texts)
        v = np.array([[len(t), 1, 2, 3] for t in texts], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_chunks(header):
    return [{"chunk_index": i, "text": f"texto {i}", "header": header, "n_tokens": 10, "model_name": MODEL_NAME}
            for i in range(3)]


def test_se_embebe_el_encabezado_si_existe_y_si_no_el_texto_solo():
    m = FakeModel()
    embed.embed_chunks(m, make_chunks("Cap › Sec"))
    assert m.seen[0] == "Cap › Sec\ntexto 0"
    m = FakeModel()
    embed.embed_chunks(m, make_chunks(""))
    assert m.seen[0] == "texto 0"


def test_main_de_punta_a_punta_con_variante(tmp_path, monkeypatch):
    chunks = make_chunks("Cap › Sec")
    (tmp_path / "c.json").write_text(json.dumps(chunks), encoding="utf-8")
    monkeypatch.setattr(embed, "variant_paths", lambda v="": (tmp_path / "c.json", tmp_path / "e.npz"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    embed.main("header")
    d = np.load(tmp_path / "e.npz")
    assert d["embeddings"].shape == (3, 4) and list(d["ids"]) == [0, 1, 2] and bool(d["normalized"])
    assert str(d["model_name"]) == MODEL_NAME