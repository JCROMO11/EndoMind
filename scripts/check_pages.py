import sys
import pymupdf

path = sys.argv[1]
paginas = [int(x) for x in sys.argv[2:]] or [47, 208, 277, 530]

with pymupdf.open(path) as doc:
    for num in paginas:
        page = doc[num - 1]
        print(f"\n{'=' * 70}\nPÁGINA {num}  (alto={page.rect.height:.0f})")

        # Márgenes: ¿hay header o footer reales?
        blocks = page.get_text("blocks")   # (x0, y0, x1, y1, texto, ...)
        arriba = [b[4].strip() for b in blocks if b[1] < 60 and b[4].strip()]
        abajo = [b[4].strip() for b in blocks if b[3] > page.rect.height - 50 and b[4].strip()]
        print("ARRIBA:", arriba)
        print("ABAJO :", abajo)

        # Tablas detectadas, en markdown
        tablas = page.find_tables().tables
        print(f"\nfind_tables: {len(tablas)} tabla(s)")
        for t in tablas:
            print(f"--- bbox={tuple(round(x) for x in t.bbox)}")
            print(t.to_markdown()[:1200])