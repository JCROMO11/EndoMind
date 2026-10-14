import sys
from pathlib import Path

from rag.schemas import GeneratedEmail


def show(path: Path) -> None:
    e = GeneratedEmail.model_validate_json(path.read_text(encoding="utf-8"))
    by_id = {c.chunk_index: c for c in e.context}
    cited = {s for p in e.paragraphs for s in p.sources}
    words = sum(len(p.text.split()) for p in e.paragraphs)

    print(f"=== {path.name} · día {e.day} · {words} palabras\nASUNTO: {e.subject}\n")
    for n, p in enumerate(e.paragraphs, 1):
        print(f"--- PÁRRAFO {n} · cita {p.sources}\n{p.text}\n")
        for s in p.sources:
            c = by_id[s]
            print(f"    [#{s}] ({c.section}) {c.content}\n")
    print(f"Contexto no citado: {sorted(set(by_id) - cited)}\n")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        show(Path(arg))
