# Eval log

Criterio principal: lax (capítulo + sección + página). `strict` = chunk_index exacto.
Las variantes (`[nombre]` en la nota) solo tienen lax: sus chunk_index no son los del YAML.
La primera fila es anterior al arreglo del criterio lax (por eso 0%) y su columna por dificultad es strict.

| fecha | modelo | norm | strict@3 | strict@5 | lax@3 | lax@5 | lax por dificultad (@3/@5) | nota |
|---|---|---|---|---|---|---|---|---|
| 2026-10-08 18:15 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | True | 50% | 60% | 0% | 0% | easy 67%/80%; hard 0%/0% |
| 2026-10-08 18:21 | paraphrase-multilingual-MiniLM-L12-v2 | True | 50% | 60% | 70% | 70% | easy 93%/93%; hard 0%/0% |  |
| 2026-10-08 19:15 | paraphrase-multilingual-MiniLM-L12-v2 | True | 50% | 60% | 70% | 70% | easy 93%/93%; hard 0%/0% | re-run baseline tras refactor |
| 2026-10-08 19:15 | paraphrase-multilingual-MiniLM-L12-v2 | True | n/a | n/a | 70% | 75% | easy 87%/93%; hard 20%/20% | [ctrl100] control sin header, max 100 |
| 2026-10-08 19:24 | paraphrase-multilingual-MiniLM-L12-v2 | True | n/a | n/a | 65% | 70% | easy 87%/87%; hard 0%/20% | [header] header cap›sec; espero flip t2d-15 |
