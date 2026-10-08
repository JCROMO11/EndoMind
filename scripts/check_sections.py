"""Revisa la calidad de data/processed/greenspan_sections.json antes del chunking.

Uso: uv run python -m scripts.check_sections
"""
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/greenspan_sections.json")
secs = json.loads(path.read_text(encoding="utf-8"))
paras = [(s["chapter_num"], s["section"], p["page"], p["text"])
         for s in secs for p in s["paragraphs"]]
print(f"{len(secs)} secciones, {len(paras)} párrafos, {sum(len(p[3]) for p in paras):,} caracteres")


def muestra(titulo, items, n=5):
    print(f"\n{titulo}: {len(items)}")
    for cap, sec, pag, txt in items[:n]:
        print(f"  cap {cap}, p.{pag}, [{sec[:30]}]  {txt[:70]!r}")


# 1. Tamaño de las secciones: decide cómo se hace el chunking
tam = sorted((sum(len(p["text"]) for p in s["paragraphs"]), s["chapter_num"], s["section"])
             for s in secs)
k = len(tam)
print(f"\nCaracteres por sección: mín={tam[0][0]:,} p10={tam[k // 10][0]:,} "
      f"mediana={tam[k // 2][0]:,} p90={tam[k * 9 // 10][0]:,} máx={tam[-1][0]:,}")
print("Las 3 más largas:")
for t, cap, sec in tam[-3:][::-1]:
    print(f"  {t:>7,}  cap {cap}  {sec[:60]}")

# 2. Párrafos sospechosos
FINALES = (".", "?", "!", ":", ";", ")")
muestra("Párrafos que empiezan en minúscula", [x for x in paras if x[3][:1].islower()])
muestra("Párrafos sin punto final", [x for x in paras if not x[3].rstrip().endswith(FINALES)])
muestra("Párrafos de menos de 40 caracteres", [x for x in paras if len(x[3]) < 40])

# 3. Reparto por capítulo
por_cap = Counter()
for cap, _, _, txt in paras:
    por_cap[cap] += len(txt)
print("\nCaracteres por capítulo (miles):", {c: n // 1000 for c, n in sorted(por_cap.items())})
