# ingest/load.py
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np
from sqlalchemy import create_engine, delete, func, insert, select
from sqlalchemy.orm import Session, sessionmaker

from config import EMBEDDING_MODEL, db_settings
from db.session import engine, SessionLocal
from db.models import Book, Chunk

PATH_CHUNKS = Path("data/processed/greenspan_chunks.json")
PATH_EMBS = Path("data/processed/greenspan_embs.npz")
PATH_PDF = Path("data/raw/Greenspan Endocrinología básica y clínica (10a. ed.) (Gardner, David G. (Editor) Shoback etc.) (z-lib.org).pdf")
BOOK_TITLE = "Greenspan Endocrinología básica y clínica, 10.ª ed."
BOOK_AUTHOR = "Gardner, David G.; Shoback, Dolores (eds.)"

EMB_DIM = 384
NORM_ATOL = 1e-5
BATCH_SIZE = 1_000

# ================================================================ puras

def load_inputs(
    chunks_path: Path, embs_path: Path
) -> tuple[list[dict], np.ndarray, np.ndarray, str]:
    """Lee chunks.json y el .npz, valida su consistencia y devuelve
    (chunks, ids, embs, model_name). Lanza ValueError ante la primera
    inconsistencia, antes de tocar la base de datos."""
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    with np.load(embs_path) as f:
        ids = f["ids"]
        embs = f["embeddings"]
        model_name = str(f["model_name"])

    n = len(chunks)

    # 1. No vacío
    if n == 0:
        raise ValueError(f"{chunks_path}: no contiene chunks")

    # 2. Conteos (antes que cualquier comparación fila a fila)
    if not (n == len(ids) == embs.shape[0]):
        raise ValueError(
            f"conteos distintos: chunks={n}, ids={len(ids)}, embs={embs.shape[0]}"
        )

    # 3. Forma y tipo
    if embs.ndim != 2 or embs.shape[1] != EMB_DIM:
        raise ValueError(f"forma de embs={embs.shape}, esperada ({n}, {EMB_DIM})")
    if embs.dtype != np.float32:
        raise ValueError(f"dtype de embs={embs.dtype}, esperado float32")

    # 4. Modelo: única fuente de verdad en config.EMBEDDING_MODEL
    if model_name != EMBEDDING_MODEL:
        raise ValueError(f"modelo en npz='{model_name}', esperado='{EMBEDDING_MODEL}'")

    # 5. Unicidad de chunk_index
    chunk_ids = np.array([c["chunk_index"] for c in chunks])
    uniq, counts = np.unique(chunk_ids, return_counts=True)
    dup = uniq[counts > 1]
    if dup.size:
        raise ValueError(
            f"chunk_index duplicado: {dup[:5].tolist()} ({dup.size} en total)"
        )

    # 6. Alineación ids ↔ chunk_index
    bad = np.flatnonzero(ids != chunk_ids)
    if bad.size:
        i = bad[0]
        raise ValueError(
            f"fila {i}: chunk_index={chunk_ids[i]}, id en npz={ids[i]} "
            f"({bad.size} filas desalineadas)"
        )

    # 7. Finitos (NaN o inf)
    bad = np.flatnonzero(~np.isfinite(embs).all(axis=1))
    if bad.size:
        i = bad[0]
        raise ValueError(
            f"fila {i} (chunk_index={chunk_ids[i]}): vector con NaN/inf "
            f"({bad.size} filas afectadas)"
        )

    # 8. Normalización
    norms = np.linalg.norm(embs, axis=1)
    bad = np.flatnonzero(np.abs(norms - 1.0) > NORM_ATOL)
    if bad.size:
        i = bad[0]
        raise ValueError(
            f"fila {i} (chunk_index={chunk_ids[i]}): norma={norms[i]:.6f}, "
            f"esperada 1.0 ({bad.size} filas)"
        )

    return chunks, ids, embs, model_name


def file_sha256(pdf_path: Path, block_size: int = 1 << 20) -> str:
    """Huella del PDF: identifica el libro."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while block := f.read(block_size):
            h.update(block)
    return h.hexdigest()


def content_sha256(
    chunks: list[dict], ids: np.ndarray, embs: np.ndarray, model_name: str
) -> str:
    """Huella de lo que el pipeline sacó del PDF (chunks + vectores + modelo).
    Se calcula sobre los datos, no sobre los archivos, para ser estable."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    for c in chunks:
        h.update(json.dumps(c, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\x00")
    h.update(np.ascontiguousarray(ids, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(embs, dtype=np.float32).tobytes())
    return h.hexdigest()


def chunk_row(c: dict, emb: np.ndarray, book_id: UUID) -> dict:
    """Traduce un chunk del JSON a las columnas de la tabla chunks."""
    return {
        "book_id": book_id,
        "chunk_index": c["chunk_index"],
        "chapter_num": c["chapter_num"],
        "chapter": c["chapter"],
        "section": c["section"],
        "printed_page_start": c["printed_page_start"],
        "printed_page_end": c["printed_page_end"],
        "content": c["text"],
        "n_tokens": c["n_tokens"],
        "embedding": emb,
    }


# ================================================================ base de datos
# Ninguna función hace commit: todas trabajan dentro de la transacción de main().

def upsert_book(
    s: Session, *, title: str, author: str, file_hash: str,
    content_hash: str, model_name: str,
) -> tuple[UUID, bool]:
    """Devuelve (book_id, necesita_carga)."""
    book = s.scalars(select(Book).where(Book.file_sha256 == file_hash)).one_or_none()

    # caso 1: libro nuevo
    if book is None:
        book = Book(title=title, author=author, file_sha256=file_hash,
                    content_sha256=content_hash, embedding_model=model_name)
        s.add(book)
        s.flush()  # genera book.id sin hacer commit
        return book.id, True

    # caso 2: misma carga → nada que hacer
    if book.content_sha256 == content_hash:
        return book.id, False

    # caso 3: mismo PDF, procesado distinto → reemplazar todos sus chunks.
    # Un cambio de modelo solo llega aquí si se cambió config.EMBEDDING_MODEL,
    # que también usa retrieve(); load_inputs rechaza cualquier otro.
    result = s.execute(delete(Chunk).where(Chunk.book_id == book.id))
    print(f"Recargando: borrados {result.rowcount} chunks anteriores.")
    book.content_sha256 = content_hash
    book.embedding_model = model_name
    s.flush()
    return book.id, True


def insert_chunks(s: Session, rows: list[dict]) -> None:
    for start in range(0, len(rows), BATCH_SIZE):
        s.execute(insert(Chunk), rows[start:start + BATCH_SIZE])


def verify(s: Session, book_id: UUID, expected: int) -> int:
    n = s.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.book_id == book_id)
    )
    if n != expected:
        raise RuntimeError(f"verify: {n} chunks en DB, esperados {expected}")
    return n


# ---------------------------------------------------------------- main

def main() -> None:
    chunks, ids, embs, model_name = load_inputs(PATH_CHUNKS, PATH_EMBS)
    file_hash = file_sha256(PATH_PDF)
    content_hash = content_sha256(chunks, ids, embs, model_name)

    with SessionLocal.begin() as s:  # commit al salir; rollback si hay excepción
        book_id, needs_load = upsert_book(
            s,
            title=BOOK_TITLE,
            author=BOOK_AUTHOR,
            file_hash=file_hash,
            content_hash=content_hash,
            model_name=model_name,
        )
        if not needs_load:
            print("Carga idéntica ya existe; nada que hacer.")
            return

        insert_chunks(s, [chunk_row(c, embs[i], book_id) for i, c in enumerate(chunks)])
        total = verify(s, book_id, len(chunks))

    print(f"Cargados {total} chunks.")  # fuera del with: ya hubo commit


if __name__ == "__main__":
    main()