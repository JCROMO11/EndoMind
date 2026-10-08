import re
import sys
import pymupdf

CAP = re.compile(r"^TABLA \d+[-‑–]\d+", re.M)
with pymupdf.open(sys.argv[1]) as doc:
    con = [(i + 1, CAP.findall(p.get_text())) for i, p in enumerate(doc)]
    con = [(n, c) for n, c in con if c]
    print(f"Páginas con leyenda TABLA: {len(con)} | leyendas: {sum(len(c) for _, c in con)}")
    for n, c in con[:: max(len(con) // 12, 1)]:
        print(f"p.{n}: {c}  find_tables={len(doc[n - 1].find_tables().tables)}")
