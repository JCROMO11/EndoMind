"""embed: calcula el embedding de cada chunk y lo guarda en un .npz.

Lo que se embebe es `embed_text(chunk)`: el texto, con el encabezado "capítulo › sección" delante
si el chunk lo trae (variante --header de chunk.py). Las consultas NUNCA llevan encabezado.

Uso:  uv run python -m ingest.embed                     # baseline
      uv run python -m ingest.embed --variant header    # lee/escribe los archivos de esa variante
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ingest.chunk import MODEL_LIMIT, MODEL_NAME, embed_text, variant_paths

PATH_CHUNKS, PATH_EMBS = variant_paths()      # baseline; smoke.py importa PATH_EMBS
NORMALIZE = True                              # con vectores unitarios, coseno = producto punto


def embed_chunks(model, chunks: list[dict], normalize: bool = NORMALIZE) -> np.ndarray:
    assert all(c["model_name"] == MODEL_NAME for c in chunks), "chunks hechos con otro modelo: re-chunkear"
    texts = [embed_text(c) for c in chunks]
    return model.encode(texts, normalize_embeddings=normalize, batch_size=64, show_progress_bar=True)


def checks(model, embs: np.ndarray, chunks: list[dict], normalize: bool) -> None:
    assert embs.shape == (len(chunks), model.get_embedding_dimension()), embs.shape
    assert np.isfinite(embs).all(), "hay NaN o inf en los embeddings"
    over = sum(c["n_tokens"] > MODEL_LIMIT for c in chunks)
    assert over == 0, f"{over} chunks superan {MODEL_LIMIT} tokens: el modelo los truncaría"
    if normalize:
        assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-4), "no están normalizados"
    print(f"max_seq_length={model.max_seq_length} · shape={embs.shape} · finitos=True · "
          f"con encabezado={any(c.get('header') for c in chunks)}")


def main(variant: str = ""):
    from sentence_transformers import SentenceTransformer     # import perezoso

    path_chunks, path_embs = variant_paths(variant)
    chunks = json.loads(Path(path_chunks).read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)
    embs = embed_chunks(model, chunks)
    checks(model, embs, chunks, NORMALIZE)
    ids = np.array([c["chunk_index"] for c in chunks])
    np.savez(path_embs, embeddings=embs, ids=ids, model_name=MODEL_NAME, normalized=NORMALIZE)
    print(f"Guardado en {path_embs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="")
    main(ap.parse_args().variant)