import sys
import pymupdf

path, num = sys.argv[1], int(sys.argv[2])

with pymupdf.open(path) as doc:
    page = doc[num - 1]
    print(f"ancho={page.rect.width:.0f} alto={page.rect.height:.0f}")
    for i, b in enumerate(page.get_text("blocks")):
        x0, y0, x1, y1, texto = b[:5]
        if b[6] == 0:  # solo bloques de texto
            print(f"{i:>2} x={x0:>3.0f}-{x1:>3.0f} y={y0:>3.0f}-{y1:>3.0f}  {texto[:50]!r}")