"""chunk: divide las secciones de Greenspan en chunks que caben en el modelo de embeddings.

Reglas:
- La unidad mínima es la ORACIÓN. Un chunk junta oraciones consecutivas de la misma
  sección hasta MAX_TOKENS (medido con el tokenizer del modelo, sobre el texto ya unido).
- Una oración que sola supera MAX_TOKENS se parte por ';', luego por ',', y al final por palabras.
- Solapamiento: la última oración del chunk anterior abre el siguiente (nunca entre secciones).
- Cada chunk lleva capítulo, sección y páginas (PDF e impresa) para filtrar y citar.

Uso:  uv run python3 -m ingest.chunk
"""
import json
import random
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_LIMIT = 126        # max_seq_length 128 menos 2 tokens especiales: más allá, el modelo trunca
MAX_TOKENS = 120         # presupuesto por chunk, con un poco de margen
SMALL_CHUNK = 30         # por debajo de esto el embedding suele ser pobre
IN_PATH = Path("data/processed/greenspan_sections.json")
OUT_PATH = Path("data/processed/greenspan_chunks.json")

# --- Oraciones ---------------------------------------------------------------
PLACEHOLDER = ""   # carácter privado: no puede aparecer en el libro
ABREVIATURAS = ["ej", "Fig", "Figs", "aprox", "Dr", "Dra", "vs", "al", "cap", "pág", "núm"]
_PROTEGER = re.compile(r"\b(?:" + "|".join(ABREVIATURAS) + r")\.|\b[a-z]\.(?=\s)")
# corta tras . ? ! (o tras ".)" / '."') + espacio, si lo siguiente abre una oración
_CORTE = re.compile(r"(?:(?<=[.?!])|(?<=[.?!][)\]\"”]))\s+(?=[A-ZÁÉÍÓÚÑ0-9(¿¡])")


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer   # import perezoso: los tests no lo necesitan
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text: str) -> int:
    return len(get_tokenizer()(text, add_special_tokens=False)["input_ids"])


def load_sections(path: Path = IN_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_sentences(text: str) -> list[str]:
    protegido = _PROTEGER.sub(lambda m: m.group().replace(".", PLACEHOLDER), text)
    partes = _CORTE.split(protegido)
    return [p.replace(PLACEHOLDER, ".").strip() for p in partes if p.strip()]


def split_long(text: str, max_size: int, measure=count_tokens) -> list[str]:
    """Parte un texto que supera max_size: primero por ';', luego ',', al final por palabras."""
    if measure(text) <= max_size:
        return [text]
    parts = [text]
    for sep in (r"(?<=;)\s+", r"(?<=,)\s+", r"\s+"):
        parts = re.split(sep, text)
        if len(parts) > 1:
            break
    if len(parts) == 1:              # una sola "palabra" enorme: no hay dónde cortar
        return [text]
    pieces, cur = [], ""
    for p in parts:
        cand = f"{cur} {p}".strip()
        if cur and measure(cand) > max_size:
            pieces.append(cur)
            cur = p
        else:
            cur = cand
    pieces.append(cur)
    out = []
    for piece in pieces:             # un trozo todavía grande se parte con el separador siguiente
        out.extend(split_long(piece, max_size, measure) if measure(piece) > max_size else [piece])
    return out


# --- Chunks ------------------------------------------------------------------
def chunk_section(section: dict, max_size: int = MAX_TOKENS, measure=count_tokens) -> list[dict]:
    sents = [
        {"text": piece, "page": p["page"], "printed": p["printed_page"]}
        for p in section["paragraphs"]
        for o in split_sentences(p["text"])
        for piece in split_long(o, max_size, measure)
    ]

    def size(items):
        return measure(" ".join(x["text"] for x in items))

    grupos, actual = [], []
    for s in sents:
        if actual and size(actual + [s]) > max_size:
            grupos.append(actual)
            actual = [actual[-1], s]        # solapamiento: la última oración abre el siguiente
            if size(actual) > max_size:     # si no cabe con el solapamiento, empieza limpio
                actual = [s]
        else:
            actual.append(s)
    if actual:
        grupos.append(actual)

    return [
        {
            "chapter_num": section["chapter_num"],
            "chapter": section["chapter"],
            "section": section["section"],
            "page_start": min(x["page"] for x in g),
            "page_end": max(x["page"] for x in g),
            "printed_page_start": min(x["printed"] for x in g),
            "printed_page_end": max(x["printed"] for x in g),
            "text": " ".join(x["text"] for x in g),
        }
        for g in grupos
    ]


def build_chunks(section_chunks: list[list[dict]], measure=count_tokens) -> list[dict]:
    chunks = [c for lista in section_chunks for c in lista]
    for i, c in enumerate(chunks):
        c["chunk_index"] = i                      # único en el libro: (libro, chunk_index) es la clave
        c["n_tokens"] = measure(c["text"])        # medida independiente, no la del bucle de armado
        c["max_tokens"] = MAX_TOKENS
        c["model_name"] = MODEL_NAME
    return chunks


# --- Reporte -----------------------------------------------------------------
def source_chars(sections: list[dict]) -> int:
    return sum(len(" ".join(p["text"].split())) for s in sections for p in s["paragraphs"])


def report(chunks: list[dict], sections: list[dict]) -> str:
    lens = np.array([c["n_tokens"] for c in chunks])
    cobertura = sum(len(c["text"]) for c in chunks) / source_chars(sections)
    muestra = random.Random(0).sample(chunks, min(3, len(chunks)))
    lineas = [
        f"{'Total chunks':<30}: {len(chunks)}",
        f"{'Tokens mínimo':<30}: {lens.min()}",
        f"{'Tokens mediana':<30}: {np.median(lens):.1f}",
        f"{'90% por debajo de':<30}: {np.percentile(lens, 90):.1f}",
        f"{'Tokens máximo':<30}: {lens.max()}",
        f"{f'Chunks > {MAX_TOKENS} (presupuesto)':<30}: {(lens > MAX_TOKENS).sum()}",
        f"{f'Chunks > {MODEL_LIMIT} (TRUNCA)':<30}: {(lens > MODEL_LIMIT).sum()}",
        f"{f'Chunks < {SMALL_CHUNK} tokens':<30}: {(lens < SMALL_CHUNK).sum()}",
        f"{'Chars chunks / chars fuente':<30}: {cobertura:.2f}   (debe ser >= 1.00; el solapamiento suma ~0.1-0.3)",
        "\nMuestra:",
    ]
    for c in muestra:
        lineas.append(f"  [#{c['chunk_index']} cap {c['chapter_num']} · {c['section'][:40]} · "
                      f"p.{c['printed_page_start']}-{c['printed_page_end']} · {c['n_tokens']} tok]\n  {c['text'][:250]}")
    return "\n".join(lineas)


def main():
    sections = load_sections()
    chunks = build_chunks([chunk_section(s) for s in sections])
    print(report(chunks, sections))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()