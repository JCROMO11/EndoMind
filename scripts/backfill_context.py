"""Una sola vez: reemplaza context_chunks (ids) por context (chunks completos) en data/generated."""
import json
import shutil
import sys
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import select

from db.models import Chunk
from db.session import SessionLocal
from rag.schemas import GeneratedEmail

DIR = Path("data/generated")
BACKUP = Path("data/generated_bak")


def load_chunks(ids: set[int]) -> dict[int, dict]:
    """{chunk_index: dict con la forma de ContextChunk}. score=None: depende de la query, no está en la BD."""
    with SessionLocal() as s:
        rows = list(s.scalars(select(Chunk).where(Chunk.chunk_index.in_(ids))))
    if len(rows) != len({r.chunk_index for r in rows}):
        raise ValueError("chunk_index repetido entre libros: hay que filtrar por book_id")
    return {
        r.chunk_index: {
            "chunk_index": r.chunk_index,
            "section": r.section,
            "chapter": r.chapter_num,
            "page_start": r.printed_page_start,
            "page_end": r.printed_page_end,
            "content": r.content,
            "score": None,
        }
        for r in rows
    }


def write_atomic(path: Path, text: str) -> None:
    """Escribe a un temporal y lo renombra: si algo falla a mitad, el original queda intacto."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main(dry_run: bool) -> int:
    ok, migrated, skipped, failed = [], [], [], []
    pending: dict[Path, dict] = {}

    # 1) Clasificar cada archivo por separado: uno roto no tumba a los demás
    for p in sorted(DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            failed.append((p, f"JSON inválido: {e}"))
            continue
        if "context" in data:
            try:
                GeneratedEmail.model_validate(data)
                ok.append(p)
            except ValidationError as e:
                failed.append((p, f"tiene context pero no valida: {e.error_count()} errores"))
        elif "context_chunks" in data:
            pending[p] = data
        else:
            skipped.append((p, "formato anterior al refactor"))

    # 2) Una sola consulta para todos los ids
    ids = {i for d in pending.values() for i in d["context_chunks"]}
    chunks = load_chunks(ids) if ids else {}

    if pending and not dry_run and not BACKUP.exists():
        shutil.copytree(DIR, BACKUP)
        print(f"Backup en {BACKUP}")

    # 3) Migrar: validar antes de escribir
    for p, data in pending.items():
        missing = [i for i in data["context_chunks"] if i not in chunks]
        if missing:
            failed.append((p, f"no están en la BD: {missing}"))
            continue
        data["context"] = [chunks[i] for i in data.pop("context_chunks")]  # respeta el orden original
        try:
            email = GeneratedEmail.model_validate(data)
        except ValidationError as e:
            failed.append((p, f"no valida tras migrar: {e.error_count()} errores"))
            continue
        if not dry_run:
            write_atomic(p, email.model_dump_json(indent=2))
        migrated.append(p)

    # 4) Informe
    print(f"\n{'DRY RUN — nada se escribió' if dry_run else 'MIGRACIÓN'}")
    print(f"  ya válidos : {len(ok)}")
    print(f"  migrados   : {len(migrated)}")
    for p in migrated:
        print(f"     ✓ {p.name}")
    print(f"  omitidos   : {len(skipped)}")
    for p, why in skipped:
        print(f"     - {p.name}: {why}")
    print(f"  fallidos   : {len(failed)}")
    for p, why in failed:
        print(f"     ✗ {p.name}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(dry_run="--dry-run" in sys.argv))