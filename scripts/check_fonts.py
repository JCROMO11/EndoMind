import sys
import pymupdf

path, num = sys.argv[1], int(sys.argv[2])
with pymupdf.open(path) as doc:
    for i, b in enumerate(doc[num - 1].get_text("dict")["blocks"]):
        if b["type"] != 0:
            continue
        spans = [s for l in b["lines"] for s in l["spans"]]
        size = max(s["size"] for s in spans)
        x0, y0 = b["bbox"][:2]
        texto = "".join(s["text"] for s in spans)
        print(f"{i:>2} size={size:>4.1f} {spans[0]['font'][:22]:<22} x={x0:>3.0f} y={y0:>3.0f}  {texto[:40]!r}")
