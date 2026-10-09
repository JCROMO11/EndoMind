import json

import pytest

from scripts import compare_runs as cr


def row(qid, lax3, lax5, diff='easy', chap=17, sec='S', page=1):
    return {'id': qid, 'difficulty': diff, 'lax@3': lax3, 'lax@5': lax5,
            'top_where': [{'chapter_num': chap, 'section': sec, 'page_start': page, 'page_end': page}]}


def write_run(path, rows, note=''):
    total = {m: sum(r[m] for r in rows) / len(rows) for m in cr.METRICS}
    path.write_text(json.dumps({'note': note, 'total': total, 'queries': rows}), encoding='utf-8')
    return path


def test_flips_distingue_ganadas_de_perdidas_aunque_el_neto_sea_cero(tmp_path):
    a = cr.load(write_run(tmp_path / 'a.json', [row('q1', True, True), row('q2', False, False), row('q3', True, True)]))
    b = cr.load(write_run(tmp_path / 'b.json', [row('q1', False, True), row('q2', True, True), row('q3', True, True)]))
    assert cr.flips(a, b, 'lax@3') == (['q2'], ['q1'])           # neto 0, pero cambiaron dos consultas
    assert cr.flips(a, b, 'lax@5') == (['q2'], [])
    txt = cr.report(a, b, 'a', 'b')
    assert '+ q2' in txt and '- q1' in txt and 'neto +0' in txt and 'neto +1' in txt


def test_consultas_distintas_fallan_en_voz_alta(tmp_path):
    a = cr.load(write_run(tmp_path / 'a.json', [row('q1', True, True)]))
    b = cr.load(write_run(tmp_path / 'b.json', [row('q9', True, True)]))
    with pytest.raises(AssertionError):
        cr.flips(a, b, 'lax@3')


def test_resolve_elige_la_corrida_mas_reciente_de_cada_variante(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, 'DIR_EVAL', tmp_path)
    for name in ['20261008_100000_m_norm1.json', '20261008_120000_m_norm1.json',
                 '20261008_110000_m_norm1_header.json', '20261008_130000_m_norm1_header.json',
                 '20261008_140000_m_norm1_ctrl100.json']:
        (tmp_path / name).write_text('{}')
    assert cr.resolve('baseline').name == '20261008_120000_m_norm1.json'      # ignora las variantes
    assert cr.resolve('header').name == '20261008_130000_m_norm1_header.json'
    assert cr.resolve('x/y.json').as_posix() == 'x/y.json'
    with pytest.raises(FileNotFoundError):
        cr.resolve('nope')