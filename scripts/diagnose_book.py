import sys
from collections import Counter
import pymupdf

path = sys.argv[1]

with pymupdf.open(path) as doc:
    print(f"Páginas: {len(doc)}")

    # 1. ¿Hay índice (TOC) que sirva para capítulos y secciones?
    toc = doc.get_toc()
    print(f"\nTOC: {len(toc)} entradas")
    for lvl, title, page in toc[:15]:
        print(f"  {'  ' * (lvl - 1)}{title[:60]}  (p.{page})")

    # 2. ¿Es escaneado? Páginas sin capa de texto
    texts = [p.get_text() for p in doc]
    empty = [i + 1 for i, t in enumerate(texts) if len(t.strip()) < 20]
    print(f"\nPáginas casi sin texto: {len(empty)} {empty[:10]}")

    # 3. Headers y footers: líneas que se repiten en muchas páginas
    c = Counter()
    for t in texts:
        for line in {l.strip() for l in t.splitlines() if l.strip()}:
            c[line] += 1
    print("\nLíneas más repetidas (candidatas a header/footer):")
    for line, n in c.most_common(8):
        print(f"  {n:>4}x  {line[:70]}")

    # 4. Tablas detectadas en una muestra de páginas
    step = max(len(doc) // 40, 1)
    con_tablas = []
    for i in range(0, len(doc), step):
        n = len(doc[i].find_tables().tables)
        if n:
            con_tablas.append((i + 1, n))
    print(f"\nMuestra cada {step} páginas, con tablas: {con_tablas[:15]}")