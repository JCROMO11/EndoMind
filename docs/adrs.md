# Architecture Decision Records

Formato: contexto → decisión → consecuencias. Estado: `aceptada`, `rechazada`, `pospuesta`.

---

## ADR-001 · Variantes de chunking aisladas del baseline

- **Fecha:** 2026-10-08 · **Estado:** aceptada

**Contexto.** Para probar cambios de chunking (encabezado, otro presupuesto de tokens) hacía falta
re-chunkear y re-embeber sin pisar el baseline, y comparar las corridas sin confundir +1/-1 con "sin cambios".

**Decisión.**
- `ingest.chunk` / `ingest.embed` / `scripts.eval_retrieval` aceptan `--variant NOMBRE`; cada variante
  escribe `greenspan_chunks_NOMBRE.json` y `greenspan_embs_NOMBRE.npz` (`variant_paths()` es la única fuente de rutas).
- `embed_text(chunk)` es la única definición de "lo que ve el modelo"; `n_tokens` se mide sobre eso.
- En las variantes solo se calcula `lax` (capítulo + sección + página), porque los `chunk_index` del YAML
  solo valen para el chunking del baseline.
- `scripts.compare_runs` lista qué consultas ganan y pierden entre dos corridas.

**Consecuencias.** Experimentar es barato y reproducible. Pero mientras el golden set tenga 20 consultas
(5 hard), una diferencia neta de una consulta (5 pp en total, 20 pp en hard) es ruido.

---

## ADR-002 · No adoptar el encabezado "capítulo › sección" en el texto embebido

- **Fecha:** 2026-10-08 · **Estado:** rechazada (por ahora)

**Contexto.** Hipótesis: anteponer `capítulo › sección` al texto embebido da contexto a chunks que lo perdieron.

**Resultado** (lax@3 / lax@5, total y hard):

| variante | total | hard |
|---|---|---|
| baseline (120 tok) | 70% / 70% | 0% / 0% |
| ctrl100 (100 tok, sin header) | 70% / 75% | 20% / 20% |
| header (120 tok, header incluido) | 65% / 70% | 0% / 20% |

**Decisión.** Mantener el baseline sin encabezado. Las diferencias son de una consulta (ruido), y hay una
razón estructural para que no ayude: los títulos de sección son demasiado genéricos. Por ejemplo,
"Hormonas pancreáticas y diabetes mellitus › Terapia específica" se repite idéntico en 88 chunks y no
los distingue entre sí, mientras que los subtítulos reales ("A. Diabetes tipo 1.", "Agentes antihiperglicémicos.")
quedaron incrustados en el texto porque el extractor no los detecta.

**Consecuencias.** El código de `--header` se conserva como variante opcional. Vale la pena reintentarlo
cuando los encabezados sean más finos (ver ADR-003, punto 6).

---

## ADR-003 · Diagnóstico de las consultas difíciles y mejoras de retrieval pospuestas

- **Fecha:** 2026-10-08 · **Estado:** pospuesta (se sigue con la siguiente fase del MVP)

### Contexto

Baseline: `paraphrase-multilingual-MiniLM-L12-v2`, chunks de ≤120 tokens, búsqueda densa sobre los
13 886 chunks del libro. Easy lax@5 93%, **hard lax@5 0%**.

Puesto del mejor chunk gold en el ranking (baseline):

| consulta | puesto | qué falla |
|---|---|---|
| t2d-03 "orino mucho, sed, bajo de peso" | 11 | vocabulario de paciente vs. clínico; la consulta no dice "diabetes" |
| t2d-09 peso y ejercicio | 15 | respuesta repartida en varias secciones |
| t2d-07 meta de glucosa/HbA1c | 98 | "meta" ≠ "criterios de control aceptable"; "HbA1c" arrastra la sección de diagnóstico |
| t2d-14 cuándo empezar insulina | 99 | "pastillas" ≠ "agentes orales"; "insulina" arrastra cientos de chunks |
| t2d-20 fármacos sin hipoglucemia | 107 | la negación se pierde: coseno("…sin causar hipoglucemia", "…y causan hipoglucemia") = 0.98 |

Los fallos no son por poco: el gold está alrededor del puesto 100, y el top-1 erróneo tiene más similitud
(0.84–0.87) que el gold (0.69–0.73).

### Causas

1. **Brecha de vocabulario paciente ↔ libro.** Reformular t2d-03 como "síntomas de diabetes: poliuria,
   polidipsia y pérdida de peso" lleva el gold del puesto 11 al **1**.
2. **El tema domina a la intención; la negación no existe.** Con *mean pooling*, "sin" es un token entre
   muchos y su efecto se diluye en el promedio.
3. **El modelo no es de retrieval.** Está entrenado en paráfrasis (tarea simétrica), no en pregunta → pasaje
   (asimétrica). Además es pequeño y su límite de 128 tokens obliga a usar chunks diminutos.
4. **Chunks sin contexto.** 1 109 chunks (8%) empiezan con pronombre o conector sin sujeto ("Ellos también
   tienen bajo riesgo de hipoglucemia…"). La sección "Terapia específica" tiene 78 chunks y abarca 5 páginas
   con temas mezclados. Hay 139 chunks con subtítulos incrustados en el texto.
5. **Competencia del libro entero** (menor). Filtrar al capítulo 17 ayuda poco (t2d-20: 107 → 51), porque la
   mayoría de los distractores ya son del capítulo 17.

### Experimento exploratorio: BM25 e híbrido (RRF, k=60)

Script fuera del repo, con un BM25 rudimentario (sin stemming). Mejor puesto de un chunk gold:

| consulta | denso | BM25 | híbrido |
|---|---|---|---|
| t2d-03 | 11 | 4 | **1** |
| t2d-07 | 98 | **4** | 22 |
| t2d-09 | 15 | 8 | **3** |
| t2d-14 | 99 | 36 | 63 |
| t2d-20 | 107 | **1** | 10 |
| t2d-18 (easy) | 1 | 725 | 10 |

Lo denso y lo léxico fallan en consultas distintas: el léxico acierta con términos exactos (HbA1c,
hipoglucemia) y falla con sinónimos ("riñones" vs. "nefropatía").

### Problemas de la evaluación

- n = 5 hard: una consulta = 20 pp. No permite distinguir variantes.
- "hard" mezcla tipos de fallo (vocabulario, negación, síntesis) que se arreglan con técnicas distintas.
- El gold está incompleto: p. ej. el chunk 10252 ("la insulina está indicada… en tipo 2 cuya hiperglucemia
  no responde a la dieta y otros medicamentos") contesta parcialmente t2d-14 y cuenta como fallo.
- hit@k no distingue entre el puesto 6 y el 1 000.

### Mejoras pospuestas, en orden de prioridad

| # | Mejora | Ataca | Evidencia / nota | Coste |
|---|---|---|---|---|
| 1 | **Evaluación más robusta**: 15–20 consultas hard etiquetadas por tipo de fallo (`vocabulario`, `negacion`, `sintesis`…); añadir MRR y recall@20/50; revisar los gold incompletos | requisito de todo lo demás | sin esto ninguna mejora es medible | bajo |
| 2 | **Búsqueda híbrida** BM25 + densa con RRF | causas 1, 2 | t2d-03 → 1, t2d-09 → 3, t2d-20 → 10. En Postgres: `tsvector` con configuración `spanish` (stemming) junto a pgvector | bajo-medio |
| 3 | **Modelo de embeddings de retrieval multilingüe**: `intfloat/multilingual-e5-base` (prefijos `query: ` / `passage: `) o `BAAI/bge-m3` | causas 3, 4 | ≥512 tokens → chunks más grandes, menos pronombres huérfanos. Re-chunkear y regenerar el YAML | medio |
| 4 | **Reescritura de consultas con LLM** (DeepSeek ya está en `config.py`): traducir del lenguaje del paciente al clínico, HyDE o multi-query | causa 1 | t2d-03: puesto 11 → 1 al reformular | medio (latencia y coste por consulta) |
| 5 | **Reranker cross-encoder** (p. ej. `BAAI/bge-reranker-v2-m3`) sobre el top-50 | causa 2 | lee consulta y pasaje juntos, así que puede captar la negación. Requiere buen recall@50 (puntos 2–4) | medio |
| 6 | **Mejor chunking**: detectar subtítulos incrustados como secciones; *contextual retrieval* (un LLM antepone una frase de contexto a cada chunk); *parent-child* (buscar con chunks pequeños, devolver el padre) | causa 4 | reabre ADR-002 con encabezados finos | medio-alto |
| 7 | **Filtro por metadatos** (capítulo/condición) cuando la consulta lo permita | causa 5 | ganancia pequeña aislada; útil combinado | bajo |

**Consecuencias.** El MVP avanza con el retrieval actual, sabiendo que falla en preguntas formuladas
como las haría un paciente. Esas son justo las del caso de uso de EndoMind, así que el punto 1 y
el punto 2 deberían entrar antes de generar contenido para pacientes reales.
