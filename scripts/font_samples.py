import sys
from collections import Counter, defaultdict
import pymupdf

TOP = 12
chars = Counter()
muestras = defaultdict(list)   # (fuente, tamaño) -> [(página, texto)]

with pymupdf.open(sys.argv[1]) as doc:
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for s in line["spans"]:
                    clave = (s["font"], round(s["size"], 1))
                    chars[clave] += len(s["text"])
                    if len(s["text"].strip()) > 25:
                        muestras[clave].append((page.number + 1, s["text"].strip()))

for clave, n in chars.most_common(TOP):
    fuente, size = clave
    m = muestras[clave]
    print(f"\n{fuente} {size}  ({n:,} caracteres, {len(m)} fragmentos largos)")
    if not m:
        print("   (sin fragmentos largos)")
        continue
    for i in (len(m) // 6, len(m) // 2, len(m) * 5 // 6):
        pag, txt = m[min(i, len(m) - 1)]
        print(f"   p.{pag:<4} {txt[:70]!r}")
