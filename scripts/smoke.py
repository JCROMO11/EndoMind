"""smoke: prueba de humo del retrieval sobre los embeddings guardados (sin base de datos).

Carga la matriz, comprueba que cada fila corresponde a su chunk, y para unas consultas
muestra los 5 chunks más cercanos con su puntuación. Usa el modelo y la normalización
que quedaron anotados en el .npz, para no depender de recordarlos.

Uso:  uv run python -m scripts.smoke
"""
import json
from pathlib import Path

import numpy as np

from ingest.chunk import OUT_PATH as PATH_CHUNKS
from ingest.embed import PATH_EMBS

QUERIES = [
    "tratamiento de la diabetes tipo 2 con metformina",              # directa: debería caer en el cap. 17
    "fármacos que bajan el azúcar en sangre sin dar hipoglucemia",   # otras palabras, mismo significado
    "receta de arroz con pollo",                                     # fuera de tema: puntuaciones bajas
]
K = 5


def top_k(M: np.ndarray, q: np.ndarray, k: int = K, normalized: bool = True):
    """Índices y puntuaciones (coseno) de las k filas de M más cercanas a q."""
    scores = M @ q
    if not normalized:                       # sin normalizar, el producto punto no es coseno
        scores = scores / (np.linalg.norm(M, axis=1) * np.linalg.norm(q))
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]


def main():
    from sentence_transformers import SentenceTransformer

    data = np.load(PATH_EMBS)
    M, ids = data["embeddings"], data["ids"]
    model_name, normalized = str(data["model_name"]), bool(data["normalized"])
    chunks = json.loads(Path(PATH_CHUNKS).read_text(encoding="utf-8"))

    assert len(chunks) == len(M) == len(ids), f"{len(chunks)} chunks, {len(M)} filas, {len(ids)} ids"
    assert all(c["chunk_index"] == i for c, i in zip(chunks, ids)), "las filas no corresponden a los chunks"
    print(f"Matriz {M.shape} · modelo {model_name} · normalizado={normalized}\n")

    model = SentenceTransformer(model_name)
    top1 = []
    for query in QUERIES:
        q = model.encode(query, normalize_embeddings=normalized)
        idx, scores = top_k(M, q, K, normalized)
        top1.append(scores[0])
        print(f"CONSULTA: {query}")
        for i, s in zip(idx, scores):
            c = chunks[i]
            print(f"  {s:.3f}  cap {c['chapter_num']:>2} · {c['section'][:38]:<38} · "
                  f"p.{c['printed_page_start']}  {c['text'][:110]!r}")
        print()
    print("Mejor puntuación por consulta: " + ", ".join(f"{s:.3f}" for s in top1))


if __name__ == "__main__":
    main()
