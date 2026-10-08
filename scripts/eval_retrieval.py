"""eval_retrieval: hit@k de las golden queries sobre los embeddings guardados (sin base de datos).

Dos criterios de acierto, ambos contra los items `relevant` del YAML:
- strict: el chunk recuperado es uno de los chunk_index listados.
- lax:    el chunk recuperado está en el mismo capítulo y sección que un item y su rango de
          páginas impresas incluye la página del item. Es el criterio principal: el YAML lista
          solo 5 chunks por consulta y casi siempre hay otros que contestan igual de bien.

Los embeddings de los chunks se leen del .npz (no se recalculan); solo se codifican las consultas.

Uso:  uv run python3 -m scripts.eval_retrieval ["nota de la corrida"]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PATH_CHUNKS = Path('data/processed/greenspan_chunks.json')
PATH_EMBS = Path('data/processed/greenspan_embs.npz')
PATH_QUERIES = Path('tests/queries.yaml')
DIR_EVAL = Path('data/eval')              # detalle por corrida (ignorado por git)
PATH_LOG = Path('docs/eval_log.md')       # solo números y configuración (se versiona)
KS = (3, 5)
METRICS = [f'{kind}@{k}' for kind in ('strict', 'lax') for k in KS]


# --- Carga -------------------------------------------------------------------
def load_queries(path: Path) -> list[dict]:
    queries = yaml.safe_load(Path(path).read_text(encoding='utf-8'))['queries']
    for q in queries:
        assert q.get('relevant'), f"{q.get('id')}: sin items en 'relevant'"
        for r in q['relevant']:
            missing = {'chunk_index', 'chapter_num', 'page', 'section'} - r.keys()
            assert not missing, f"{q['id']}: a un item le faltan {missing}"
    return queries


def load_chunk_meta(path: Path) -> dict[int, dict]:
    """chunk_index -> dónde está el chunk (sin texto). Falla si falta un campo: nada de .get()."""
    chunks = json.loads(Path(path).read_text(encoding='utf-8'))
    return {c['chunk_index']: {'chapter_num': c['chapter_num'], 'section': c['section'],
                               'page_start': c['printed_page_start'], 'page_end': c['printed_page_end']}
            for c in chunks}


# --- Criterios de acierto ----------------------------------------------------
def is_strict_hit(chunk_index: int, relevant: list[dict]) -> bool:
    return any(chunk_index == r['chunk_index'] for r in relevant)


def is_lax_hit(where: dict, relevant: list[dict]) -> bool:
    return any(r['chapter_num'] == where['chapter_num'] and r['section'] == where['section']
               and where['page_start'] <= r['page'] <= where['page_end'] for r in relevant)


# --- Evaluación --------------------------------------------------------------
def unit_rows(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)      # coseno exacto aunque ya venga normalizado


def evaluate(queries: list[dict], qvec: np.ndarray, embs: np.ndarray, ids: np.ndarray,
             meta: dict[int, dict], ks=KS) -> list[dict]:
    """Una fila por consulta. qvec y embs deben estar normalizados (producto punto = coseno)."""
    scores = qvec @ embs.T                                           # (consultas, chunks)
    top = np.argsort(-scores, axis=1, kind='stable')[:, :max(ks)]
    rows = []
    for q, order in zip(queries, top):
        top_ids = [int(ids[i]) for i in order]
        where = [meta[cid] for cid in top_ids]                       # KeyError si un id no existe
        row = {'id': q['id'], 'difficulty': q.get('difficulty', 'na'), 'top_ids': top_ids, 'top_where': where}
        for k in ks:
            row[f'strict@{k}'] = any(is_strict_hit(cid, q['relevant']) for cid in top_ids[:k])
            row[f'lax@{k}'] = any(is_lax_hit(w, q['relevant']) for w in where[:k])
            assert row[f'lax@{k}'] >= row[f'strict@{k}'], \
                f"{q['id']}: lax@{k} < strict@{k}; el criterio laxo debe incluir al estricto (revisa los metadatos)"
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    return {'n': len(rows), **{m: sum(r[m] for r in rows) / len(rows) for m in METRICS}}


def summarize_all(rows: list[dict]) -> tuple[dict, dict]:
    by_diff = {d: summarize([r for r in rows if r['difficulty'] == d])
               for d in sorted({r['difficulty'] for r in rows})}
    return summarize(rows), by_diff


# --- Salida ------------------------------------------------------------------
def fmt_where(w: dict) -> str:
    pages = str(w['page_start']) if w['page_start'] == w['page_end'] else f"{w['page_start']}-{w['page_end']}"
    return f"cap {w['chapter_num']} · {w['section'][:34]} · p.{pages}"


def format_table(total: dict, by_diff: dict) -> str:
    lines = [f"{'':<10}{'n':>4}" + ''.join(f'{m:>11}' for m in METRICS)]
    for name, s in [('TOTAL', total), *by_diff.items()]:
        lines.append(f"{name:<10}{s['n']:>4}" + ''.join(f'{s[m]:>11.0%}' for m in METRICS))
    return '\n'.join(lines)


def format_misses(queries: list[dict], rows: list[dict], k: int = max(KS)) -> str:
    """Para cada consulta que falla (criterio laxo): lo esperado frente a lo recuperado."""
    out = []
    for q, r in zip(queries, rows):
        if r[f'lax@{k}']:
            continue
        expected = sorted({(x['chapter_num'], x['section'][:34], x['page']) for x in q['relevant']})
        out.append(f"\n[{r['id']} · {r['difficulty']}] {q['query']}")
        out.append('  esperado : ' + '; '.join(f'cap {c} · {s} · p.{p}' for c, s, p in expected))
        out += [f'  recuperó#{n}: {fmt_where(w)}' for n, w in enumerate(r['top_where'][:3], 1)]
    return '\n'.join(out) if out else '\nSin fallos con el criterio laxo.'


def save_detail(rows, total, by_diff, model_name, normalized, note) -> Path:
    DIR_EVAL.mkdir(parents=True, exist_ok=True)
    cfg = f"{model_name.split('/')[-1]}_norm{int(normalized)}"
    path = DIR_EVAL / f"{datetime.now():%Y%m%d_%H%M%S}_{cfg}.json"
    payload = {'model': model_name, 'normalized': normalized, 'note': note, 'ks': list(KS),
               'total': total, 'by_difficulty': by_diff, 'queries': rows}          # solo ids y ubicaciones, sin texto
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def append_log(total, by_diff, model_name, normalized, note):
    PATH_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PATH_LOG.exists():
        PATH_LOG.write_text(
            '# Eval log\n\nCriterio principal: lax (capítulo + sección + página). `strict` = chunk_index exacto.\n\n'
            '| fecha | modelo | norm | strict@3 | strict@5 | lax@3 | lax@5 | lax por dificultad (@3/@5) | nota |\n'
            '|---|---|---|---|---|---|---|---|---|\n', encoding='utf-8')
    diff = '; '.join(f"{d} {s['lax@3']:.0%}/{s['lax@5']:.0%}" for d, s in by_diff.items())
    line = (f"| {datetime.now():%Y-%m-%d %H:%M} | {model_name.split('/')[-1]} | {normalized} | "
            f"{total['strict@3']:.0%} | {total['strict@5']:.0%} | {total['lax@3']:.0%} | {total['lax@5']:.0%} | "
            f"{diff} | {note} |\n")
    with open(PATH_LOG, 'a', encoding='utf-8') as f:
        f.write(line)


def main(note: str = ''):
    from sentence_transformers import SentenceTransformer     # import perezoso: los tests no lo necesitan

    queries = load_queries(PATH_QUERIES)
    meta = load_chunk_meta(PATH_CHUNKS)
    data = np.load(PATH_EMBS)
    model_name, normalized = str(data['model_name']), bool(data['normalized'])
    embs, ids = unit_rows(data['embeddings']), data['ids']

    model = SentenceTransformer(model_name)
    qvec = unit_rows(model.encode([q['query'] for q in queries], normalize_embeddings=normalized))

    rows = evaluate(queries, qvec, embs, ids, meta)
    total, by_diff = summarize_all(rows)
    print(format_table(total, by_diff))
    print(format_misses(queries, rows))
    print(f"\nDetalle: {save_detail(rows, total, by_diff, model_name, normalized, note)}")
    append_log(total, by_diff, model_name, normalized, note)
    print(f'Log: {PATH_LOG}')


if __name__ == '__main__':
    main(' '.join(sys.argv[1:]))