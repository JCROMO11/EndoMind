import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlalchemy import select, text

from config import EMBEDDING_MODEL
from db.models import Chunk
from db.session import SessionLocal

EF_SEARCH = 100  # candidatos que revisa HNSW (default 40): más recall, algo más de latencia


@dataclass(frozen=True)
class Hit:
    chunk_index: int
    chapter_num: int
    section: str
    printed_page_start: int
    printed_page_end: int
    content: str
    score: float  # similitud coseno = 1 - distancia


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def retrieve(query: str, k: int = 5, chapter: int | None = None) -> list[Hit]:
    q = get_model().encode(query, normalize_embeddings=True)
    dist = Chunk.embedding.cosine_distance(q)
    stmt = select(Chunk, dist.label("distance")).order_by(dist).limit(k)
    if chapter is not None:
        stmt = stmt.where(Chunk.chapter_num == chapter)

    with SessionLocal() as s:
        # SET LOCAL: solo vale dentro de esta transacción
        s.execute(text(f"SET LOCAL hnsw.ef_search = {max(EF_SEARCH, k)}"))
        if chapter is not None:
            # HNSW + filtro selectivo devuelve menos de k filas: el escaneo iterativo lo corrige
            s.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        rows = s.execute(stmt).all()

    hits = [
        Hit(c.chunk_index, c.chapter_num, c.section,
            c.printed_page_start, c.printed_page_end, c.content, 1 - d)
        for c, d in rows
    ]
    # relaxed_order puede devolver el orden ligeramente desordenado
    return sorted(hits, key=lambda h: h.score, reverse=True)


# ---------------------------------------------------------------- CLI

def check_against_numpy(k: int = 5) -> None:
    """Compara el top-k de Postgres (HNSW) con la búsqueda exacta en numpy."""
    import numpy as np
    import yaml

    queries = yaml.safe_load(
        Path("tests/queries.yaml").read_text(encoding="utf-8")
    )["queries"]
    with np.load("data/processed/greenspan_embs.npz") as f:
        ids, E = f["ids"], f["embeddings"]

    model = get_model()
    same = 0
    for q in (q["query"] for q in queries):
        v = model.encode(q, normalize_embeddings=True)
        exact = set(ids[np.argsort(-(E @ v))[:k]].tolist())
        pg = {h.chunk_index for h in retrieve(q, k=k)}
        if exact == pg:
            same += 1
        else:
            print(f"{q[:50]} | solo numpy: {exact - pg} | solo pg: {pg - exact}")
    print(f"{same}/{len(queries)} consultas con top-{k} idéntico")


def main() -> None:
    p = argparse.ArgumentParser(description="Búsqueda semántica sobre el corpus")
    p.add_argument("query", nargs="?", help="consulta en texto libre")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--chapter", type=int, default=None)
    p.add_argument("--check", action="store_true",
                   help="comparar Postgres contra numpy en las consultas golden")
    args = p.parse_args()

    if args.check:
        check_against_numpy(args.k)
        return
    if not args.query:
        p.error("falta la consulta (o usa --check)")

    for h in retrieve(args.query, k=args.k, chapter=args.chapter):
        print(f"{h.score:.3f}  #{h.chunk_index:<6} cap {h.chapter_num:<3} "
              f"p.{h.printed_page_start}-{h.printed_page_end}  {h.content[:80]}")


if __name__ == "__main__":
    main()