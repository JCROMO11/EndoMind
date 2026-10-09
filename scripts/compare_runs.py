"""compare_runs: qué consultas cambian entre dos corridas de eval_retrieval (archivos de data/eval/).

Los totales esconden los cambios: +1 y -1 dan el mismo porcentaje. Aquí se ve CUÁL consulta pasó de
fallo a acierto (+) y cuál de acierto a fallo (-), con el primer resultado recuperado en cada corrida.
Con n=20, una diferencia neta de una sola consulta es ruido.

Cada argumento es un .json, o un nombre de variante ('baseline', 'ctrl100', 'header'): se usa la
corrida más reciente de esa variante.

Uso:  uv run python -m scripts.compare_runs baseline header
"""
import argparse
import json
import re
from pathlib import Path

DIR_EVAL = Path('data/eval')
METRICS = ('lax@3', 'lax@5')


def resolve(arg: str) -> Path:
    if arg.endswith('.json'):
        return Path(arg)
    if arg == 'baseline':
        found = [p for p in DIR_EVAL.glob('*.json') if re.search(r'_norm\d\.json$', p.name)]
    else:
        found = list(DIR_EVAL.glob(f'*_{arg}.json'))
    if not found:
        raise FileNotFoundError(f"no hay corridas de '{arg}' en {DIR_EVAL}")
    return max(found, key=lambda p: p.name)          # el nombre empieza con la fecha y hora


def load(path: Path) -> dict:
    run = json.loads(Path(path).read_text(encoding='utf-8'))
    run['by_id'] = {r['id']: r for r in run['queries']}
    return run


def flips(a: dict, b: dict, metric: str) -> tuple[list[str], list[str]]:
    assert a['by_id'].keys() == b['by_id'].keys(), 'las corridas no tienen las mismas consultas'
    gained = [i for i in a['by_id'] if not a['by_id'][i][metric] and b['by_id'][i][metric]]
    lost = [i for i in a['by_id'] if a['by_id'][i][metric] and not b['by_id'][i][metric]]
    return gained, lost


def top1(run: dict, qid: str) -> str:
    w = run['by_id'][qid]['top_where'][0]
    return f"cap {w['chapter_num']} · {w['section'][:30]} · p.{w['page_start']}"


def report(a: dict, b: dict, name_a: str, name_b: str) -> str:
    out = [f'A = {name_a}  ({a.get("note", "")})', f'B = {name_b}  ({b.get("note", "")})']
    for m in METRICS:
        gained, lost = flips(a, b, m)
        out.append(f"\n{m}: A {a['total'][m]:.0%} -> B {b['total'][m]:.0%}   "
                   f"(+{len(gained)} ganadas, -{len(lost)} perdidas, neto {len(gained) - len(lost):+d})")
        for sign, ids in (('+', gained), ('-', lost)):
            for i in ids:
                out.append(f"  {sign} {i} [{a['by_id'][i]['difficulty']}]  A: {top1(a, i)}  |  B: {top1(b, i)}")
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('a', help='referencia (.json o variante)')
    ap.add_argument('b', help='corrida nueva (.json o variante)')
    args = ap.parse_args()
    pa, pb = resolve(args.a), resolve(args.b)
    print(report(load(pa), load(pb), pa.name, pb.name))