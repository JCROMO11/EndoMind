"""extract v1.1: cuerpo y títulos de sección de Greenspan, con capítulo y página.

Lista blanca: solo entra el cuerpo (Palladio 8.9) y los títulos de sección
(ProximaNova-Bold 11). Todo lo demás (tablas, figuras, referencias, índice,
encabezados) se excluye y se cuenta por (fuente, tamaño), para poder revisar
qué se dejó fuera.

Cambios de la v1.1 (a partir de lo visto en el libro real):
- Un párrafo que no termina en punto y llega al fondo de la columna continúa en
  el siguiente aunque este empiece en mayúscula ("…La" + "CT también…").
- Un párrafo que empieza con un signo de cierre ("–)", ",") continúa al anterior.
- Los ítems con letra minúscula ("a. Gliburida…") nunca se pegan por empezar en minúscula.
- Se cuentan las comparaciones encabezado vs TOC y las uniones hechas, por regla.

Uso:  uv run python -m ingest.extract "$libro"
"""
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

PAGE_OFFSET = 25                  # página impresa = página del PDF - 25 (visto en 9 páginas)
HEADER_Y = 50                     # bloques por encima de esta altura: encabezado corrido
BOTTOM_MARGIN = 50                # un bloque que termina a menos de esto del borde inferior llega al fondo
BODY = ("URWPalladioTOT", 8.9)    # (prefijo de fuente, tamaño)
HEADING = ("ProximaNova-Bold", 11.0)
CAPTION = re.compile(r"^(FIGURA|TABLA)\s")
CHAPTER = re.compile(r"^(\d+)\.\s")
HEADER_CHAPTER = re.compile(r"CAPÍTULO\s+(\d+)")
END_SENTENCE = (".", "?", "!", ":", ";")
CONTINUATION = re.compile(r"[a-záéíóúñü)\],;–-]")   # primer carácter de un párrafo que continúa al anterior
LETTERED = re.compile(r"[a-z]\.\s")                 # ítem "a. Gliburida…": no es continuación
CLOSERS = ")],;–-"                                   # se pegan sin espacio
OUT_PATH = Path("data/processed/greenspan_sections.json")


@dataclass
class Section:
    chapter_num: int
    chapter: str
    title: str
    paragraphs: list = field(default_factory=list)   # [(página del PDF, texto)]
    at_bottom: bool = False                           # el último bloque llegó al fondo de la columna


def dominant(block):
    """(fuente, tamaño) que cubre más caracteres del bloque."""
    c = Counter()
    for line in block["lines"]:
        for s in line["spans"]:
            c[(s["font"], round(s["size"], 1))] += len(s["text"])
    return c.most_common(1)[0][0] if c else ("", 0.0)


def block_text(block):
    """Une las líneas del bloque; si una línea termina en guion y la siguiente
    empieza en minúscula, une la palabra partida."""
    out = ""
    for line in block["lines"]:
        ln = "".join(s["text"] for s in line["spans"]).strip()
        if not ln:
            continue
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}".strip()
    return out


def load_chapters(doc):
    """Capítulos = entradas de nivel 2 del TOC que empiezan con '<número>. '."""
    return [(int(m.group(1)), title, page)
            for lvl, title, page in doc.get_toc()
            if lvl == 2 and (m := CHAPTER.match(title))]


def chapter_at(chapters, page):
    cur = None
    for ch in chapters:
        if ch[2] <= page:
            cur = ch
    return cur


def add_paragraph(sec, page, text, at_bottom, stats):
    """Une el párrafo con el anterior si éste quedó cortado. El anterior no terminó
    en punto y (a) llegaba al fondo de la columna, o (b) este empieza en minúscula
    o en un signo de cierre. Los ítems 'a. …' no cuentan como continuación por (b)."""
    if sec.paragraphs and not sec.paragraphs[-1][1].endswith(END_SENTENCE):
        prev_page, prev = sec.paragraphs[-1]
        by_bottom = sec.at_bottom
        by_start = bool(CONTINUATION.match(text)) and not LETTERED.match(text)
        if by_bottom or by_start:
            if prev.endswith("-") and text[:1].islower():
                merged = prev[:-1] + text
            elif text[:1] in CLOSERS:
                merged = prev + text
            else:
                merged = f"{prev} {text}"
            sec.paragraphs[-1] = (prev_page, merged)
            sec.at_bottom = at_bottom
            stats["union_por_fondo" if by_bottom else "union_por_inicio"] += 1
            return
    sec.paragraphs.append((page, text))
    sec.at_bottom = at_bottom


def extract(path):
    sections, excluded, mismatches, stats = [], Counter(), [], Counter()
    with pymupdf.open(path) as doc:
        chapters = load_chapters(doc)
        if not chapters:
            raise SystemExit("No encontré capítulos (nivel 2 del TOC con '<n>. '). Revisa load_chapters.")
        print(f"Capítulos en el TOC: {len(chapters)} "
              f"(primero: {chapters[0][1][:40]!r} p.{chapters[0][2]}, "
              f"último: {chapters[-1][1][:40]!r} p.{chapters[-1][2]})")
        cur = None
        for page in doc:
            num = page.number + 1
            ch = chapter_at(chapters, num)
            if ch is None:          # preliminares: antes del capítulo 1
                continue
            if cur is None or cur.chapter_num != ch[0]:
                cur = Section(ch[0], ch[1], ch[1])
                sections.append(cur)
            for b in page.get_text("dict")["blocks"]:
                if b["type"] != 0 or not b["lines"]:
                    continue
                font, size = dominant(b)
                text = block_text(b)
                if not text:
                    continue
                if b["bbox"][1] < HEADER_Y:                    # encabezado corrido
                    m = HEADER_CHAPTER.search(text)
                    if m:
                        stats["encabezados_comparados"] += 1
                        if int(m.group(1)) != ch[0]:
                            mismatches.append((num, int(m.group(1)), ch[0]))
                    continue
                if font.startswith(BODY[0]) and size == BODY[1]:
                    at_bottom = b["bbox"][3] > page.rect.height - BOTTOM_MARGIN
                    add_paragraph(cur, num, text, at_bottom, stats)
                elif font == HEADING[0] and size == HEADING[1] and not CAPTION.match(text):
                    cur = Section(ch[0], ch[1], text)
                    sections.append(cur)
                else:
                    excluded[(font, size)] += len(text)
    return [s for s in sections if s.paragraphs], excluded, mismatches, stats


def main():
    sections, excluded, mismatches, stats = extract(sys.argv[1])
    total = sum(len(t) for s in sections for _, t in s.paragraphs)
    print(f"\n{len(sections)} secciones | {total:,} caracteres de cuerpo "
          f"(el censo de fuentes sugiere del orden de 3,9 a 4,1 millones)")
    print(f"Desajustes encabezado 'CAPÍTULO n' vs TOC: {len(mismatches)} de "
          f"{stats['encabezados_comparados']} comparaciones  "
          f"(página, encabezado, TOC) {mismatches[:5]}")
    print(f"Uniones de párrafo: {stats['union_por_fondo']} por fondo de columna, "
          f"{stats['union_por_inicio']} por inicio en minúscula o signo de cierre")
    print("\nExcluido, por (fuente, tamaño):")
    for (font, size), n in excluded.most_common(8):
        print(f"  {n:>9,}  {font[:26]:<26} {size}")

    print("\nMuestra de 3 secciones:")
    paso = max(len(sections) // 3, 1)
    for s in sections[::paso][:3]:
        print(f"\n[cap {s.chapter_num}] {s.title[:60]}  "
              f"(PDF p.{s.paragraphs[0][0]}, {len(s.paragraphs)} párrafos)")
        print("  " + s.paragraphs[0][1][:300])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [{"chapter_num": s.chapter_num, "chapter": s.chapter, "section": s.title,
             "paragraphs": [{"page": p, "printed_page": p - PAGE_OFFSET, "text": t}
                            for p, t in s.paragraphs]} for s in sections]
    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
