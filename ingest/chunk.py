"""chunk: divide las secciones de Greenspan en chunks que caben en el modelo de embeddings.

Reglas:
- La unidad mínima es la ORACIÓN. Un chunk junta oraciones consecutivas de la misma
  sección hasta MAX_TOKENS (medido con el tokenizer del modelo, sobre el texto ya unido).
- Una oración que sola supera MAX_TOKENS se parte por ';', luego por ',', y al final por palabras.
- Solapamiento: la última oración del chunk anterior abre el siguiente (nunca entre secciones).
- Cada chunk lleva capítulo, sección y páginas (PDF e impresa) para filtrar y citar.
- Variante opcional --header: se antepone "capítulo › sección" al texto QUE SE EMBEBE (no al campo
  `text`, que sigue limpio). Ese encabezado gasta tokens, así que el cuerpo de cada sección se parte
  con presupuesto = max_tokens - tokens del encabezado.

Uso:  uv run python -m ingest.chunk                                  # baseline
      uv run python -m ingest.chunk --header --variant header        # con encabezado
      uv run python -m ingest.chunk --max-tokens 100 --variant ctrl100   # control: mismo presupuesto, sin encabezado
"""
import argparse
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
HEADER_MAX = 24          # tope de tokens del encabezado "capítulo › sección"
MIN_BODY = 60            # si el cuerpo tendría menos tokens que esto, algo va mal con el encabezado
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


# --- Encabezado y rutas ------------------------------------------------------
def _limpiar(titulo: str) -> str:
    titulo = re.sub(r"^\s*\d+\s*[.\-–]\s*", "", titulo).strip()     # "17. DIABETES" -> "DIABETES"
    return titulo.capitalize() if titulo.isupper() else titulo        # "DIABETES MELLITUS" -> "Diabetes mellitus"


def make_header(chapter: str, section: str, measure=count_tokens, max_tokens: int = HEADER_MAX) -> str:
    """'Capítulo › Sección', con tope de tokens: si no cabe se recorta la sección, y en último caso el capítulo."""
    cap, sec = _limpiar(chapter), _limpiar(section)
    if sec.lower() == cap.lower():
        sec = ""

    def join(c, s):
        return f"{c} › {s}" if s else c

    sec_words, cap_words = sec.split(), cap.split()
    while measure(join(" ".join(cap_words), " ".join(sec_words))) > max_tokens and sec_words:
        sec_words.pop()
    while measure(join(" ".join(cap_words), " ".join(sec_words))) > max_tokens and len(cap_words) > 1:
        cap_words.pop()
    return join(" ".join(cap_words), " ".join(sec_words))


def embed_text(chunk: dict) -> str:
    """El texto que ve el modelo de embeddings: encabezado (si hay) + texto. Única fuente de verdad."""
    header = chunk.get("header", "")
    return f"{header}\n{chunk['text']}" if header else chunk["text"]


def variant_paths(variant: str = "") -> tuple[Path, Path]:
    """(chunks.json, embs.npz) de una variante; sin nombre, los archivos del baseline."""
    suf = f"_{variant}" if variant else ""
    return OUT_PATH.with_name(f"greenspan_chunks{suf}.json"), OUT_PATH.with_name(f"greenspan_embs{suf}.npz")


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


def build_chunks(section_chunks: list[list[dict]], measure=count_tokens, max_tokens: int = MAX_TOKENS) -> list[dict]:
    chunks = [c for lista in section_chunks for c in lista]
    for i, c in enumerate(chunks):
        c["chunk_index"] = i                      # único en el libro: (libro, chunk_index) es la clave
        c["n_tokens"] = measure(embed_text(c))    # lo que ve el modelo (encabezado incluido), medido aparte
        c["max_tokens"] = max_tokens
        c["model_name"] = MODEL_NAME
    return chunks


def chunk_all(sections: list[dict], header: bool = False, max_tokens: int = MAX_TOKENS,
              measure=count_tokens) -> list[dict]:
    """Chunks de todo el libro. Con header=True el presupuesto del cuerpo de cada sección es
    max_tokens - tokens de SU encabezado (cada sección tiene un título de distinta longitud)."""
    lists = []
    for s in sections:
        h = make_header(s["chapter"], s["section"], measure) if header else ""
        budget = max_tokens - (measure(h) if h else 0)
        assert budget >= MIN_BODY, f"presupuesto {budget} < {MIN_BODY} en {s['chapter']!r} › {s['section']!r}"
        chunks = chunk_section(s, budget, measure)
        for c in chunks:
            c["header"] = h
        lists.append(chunks)
    return build_chunks(lists, measure, max_tokens)


# --- Reporte -----------------------------------------------------------------
def source_chars(sections: list[dict]) -> int:
    return sum(len(" ".join(p["text"].split())) for s in sections for p in s["paragraphs"])


def report(chunks: list[dict], sections: list[dict], max_tokens: int = MAX_TOKENS) -> str:
    lens = np.array([c["n_tokens"] for c in chunks])
    cobertura = sum(len(c["text"]) for c in chunks) / source_chars(sections)
    muestra = random.Random(0).sample(chunks, min(3, len(chunks)))
    lineas = [
        f"{'Total chunks':<30}: {len(chunks)}",
        f"{'Tokens mínimo (embebidos)':<30}: {lens.min()}",
        f"{'Tokens mediana':<30}: {np.median(lens):.1f}",
        f"{'90% por debajo de':<30}: {np.percentile(lens, 90):.1f}",
        f"{'Tokens máximo':<30}: {lens.max()}",
        f"{f'Chunks > {max_tokens} (presupuesto)':<30}: {(lens > max_tokens).sum()}",
        f"{f'Chunks > {MODEL_LIMIT} (TRUNCA)':<30}: {(lens > MODEL_LIMIT).sum()}",
        f"{f'Chunks < {SMALL_CHUNK} tokens':<30}: {(lens < SMALL_CHUNK).sum()}",
        f"{'Chars chunks / chars fuente':<30}: {cobertura:.2f}   (debe ser >= 1.00; el solapamiento suma ~0.1-0.3)",
        "\nMuestra:",
    ]
    for c in muestra:
        lineas.append(f"  [#{c['chunk_index']} cap {c['chapter_num']} · {c['section'][:40]} · "
                      f"p.{c['printed_page_start']}-{c['printed_page_end']} · {c['n_tokens']} tok]\n  {c['text'][:250]}")
    return "\n".join(lineas)


def main(variant: str = "", header: bool = False, max_tokens: int = MAX_TOKENS):
    sections = load_sections()
    chunks = chunk_all(sections, header, max_tokens)
    print(report(chunks, sections, max_tokens))
    out = variant_paths(variant)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nVariante {variant or 'baseline'!r} (header={header}, max_tokens={max_tokens}) guardada en {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="", help="nombre de la variante; sin él se sobrescribe el baseline")
    ap.add_argument("--header", action="store_true", help="anteponer 'capítulo › sección' al texto embebido")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()
    if (a.header or a.max_tokens != MAX_TOKENS) and not a.variant:
        ap.error("una variante distinta del baseline necesita --variant NOMBRE (para no pisar el baseline)")
    main(a.variant, a.header, a.max_tokens)