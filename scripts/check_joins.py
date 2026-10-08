"""Contexto de los párrafos sospechosos: cómo termina uno y cómo empieza el siguiente.

Uso: uv run python -m scripts.check_joins
"""
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/greenspan_sections.json")
secs = json.loads(path.read_text(encoding="utf-8"))
paras = [(s["chapter_num"], s["section"], p["page"], p["text"])
         for s in secs for p in s["paragraphs"]]
FINALES = (".", "?", "!", ":", ";", ")")


def repartidos(idx, n):
    """n elementos repartidos por toda la lista (no solo los primeros)."""
    paso = max(len(idx) // n, 1)
    return idx[::paso][:n]


def donde(i):
    cap, sec, pag, _ = paras[i]
    return f"cap {cap} p.{pag} [{sec[:25]}]"


# A. Empiezan en minúscula: ¿cómo termina el párrafo anterior?
minus = [i for i in range(1, len(paras)) if paras[i][3][:1].islower()]
print(f"A. Empiezan en minúscula: {len(minus)}")
for i in repartidos(minus, 8):
    print(f"\n  {donde(i)}")
    print(f"    antes: …{paras[i - 1][3][-70:]!r}   (sección anterior: {paras[i - 1][1][:25]!r})")
    print(f"    este : {paras[i][3][:70]!r}…")

# B. Sin punto final: ¿cómo terminan y qué viene después?
sin = [i for i in range(len(paras) - 1) if not paras[i][3].rstrip().endswith(FINALES)]
seguidos = sum(1 for i in sin if paras[i + 1][3][:1].islower())
print(f"\nB. Sin punto final: {len(sin)} (seguidos por un párrafo en minúscula: {seguidos})")
for i in repartidos(sin, 10):
    print(f"\n  {donde(i)}")
    print(f"    termina: …{paras[i][3][-70:]!r}")
    print(f"    sigue  : {paras[i + 1][3][:50]!r}…")

# C. Secciones más largas: ¿tienen subtítulos con letra ('A. ', 'B. ') dentro del texto?
LETRA = re.compile(r"^[A-Z]\.\s")
grandes = sorted(secs, key=lambda s: -sum(len(p["text"]) for p in s["paragraphs"]))[:3]
print("\nC. Las 3 secciones más largas")
for s in grandes:
    total = sum(len(p["text"]) for p in s["paragraphs"])
    con = [p["text"][:50] for p in s["paragraphs"] if LETRA.match(p["text"])]
    print(f"  {total:>7,} chars, {len(s['paragraphs'])} párrafos, {len(con)} empiezan con letra: {s['section'][:40]}")
    for t in con[:4]:
        print(f"      {t!r}")
