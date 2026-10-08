import sys
from collections import Counter
import pymupdf

chars = Counter()   # (fuente, tamaño) -> caracteres
primera = {}        # (fuente, tamaño) -> (página, texto) de su primera aparición

with pymupdf.open(sys.argv[1]) as doc:
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for s in line["spans"]:
                    clave = (s["font"], round(s["size"], 1))
                    chars[clave] += len(s["text"])
                    if s["text"].strip():
                        primera.setdefault(clave, (page.number + 1, s["text"].strip()[:40]))

total = sum(chars.values())
print(f"Caracteres totales: {total:,}\n")
for (fuente, size), n in chars.most_common(20):
    pag, txt = primera[(fuente, size)]
    print(f"{n / total:>6.1%}  {fuente[:26]:<26} {size:>5}  p.{pag:<4} {txt!r}")
