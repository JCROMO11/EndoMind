"""Tests de scripts/eval_retrieval.py con datos sintéticos y un modelo falso: no descargan nada."""
import json
import sys
import types

import numpy as np
import pytest
import yaml

from scripts import eval_retrieval as ev


def where(chap, sec, p0, p1):
    return {'chapter_num': chap, 'section': sec, 'page_start': p0, 'page_end': p1}


def test_lax_exige_capitulo_seccion_y_pagina_dentro_del_rango():
    rel = [{'chunk_index': 1, 'chapter_num': 17, 'section': 'Metformina', 'page': 650}]
    assert ev.is_lax_hit(where(17, 'Metformina', 650, 651), rel)
    assert ev.is_lax_hit(where(17, 'Metformina', 649, 650), rel)          # el chunk cruza páginas
    assert not ev.is_lax_hit(where(17, 'Metformina', 651, 651), rel)      # página fuera del rango
    assert not ev.is_lax_hit(where(18, 'Metformina', 650, 650), rel)      # otro capítulo, misma sección
    assert not ev.is_lax_hit(where(17, 'Sulfonilureas', 650, 650), rel)


def make_world(n=12):
    """n chunks con vectores ortogonales. Los chunks 2 y 3 están en la misma sección y página."""
    embs = np.eye(n)
    ids = np.arange(n)
    meta = {0: where(17, 'A', 600, 600), 1: where(17, 'B', 610, 610), 2: where(17, 'Metformina', 650, 650),
            3: where(17, 'Metformina', 650, 651), 4: where(17, 'C', 660, 660), 5: where(18, 'Metformina', 650, 650)}
    meta.update({i: where(19, f'Relleno {i}', 700 + i, 700 + i) for i in range(6, n)})
    return embs, ids, meta


def query(qid, relevant_idx, difficulty='easy'):
    return {'id': qid, 'query': qid, 'difficulty': difficulty,
            'relevant': [{'chunk_index': 2, 'chapter_num': 17, 'section': 'Metformina', 'page': 650}]
            if relevant_idx else []}


def test_strict_falla_pero_lax_acierta_con_un_vecino_no_listado():
    embs, ids, meta = make_world()
    qvec = embs[[3]]                                  # la consulta se parece al chunk 3, que no está listado
    rows = ev.evaluate([query('q', [2])], qvec, embs, ids, meta)
    assert rows[0]['strict@3'] is False and rows[0]['lax@3'] is True


def test_el_acierto_estricto_implica_el_laxo_y_respeta_el_orden_de_k():
    embs, ids, meta = make_world()
    qvec = embs[[2]]
    r = ev.evaluate([query('q', [2])], qvec, embs, ids, meta)[0]
    assert r['strict@3'] and r['lax@3'] and r['strict@5'] and r['lax@5']
    assert r['top_ids'][0] == 2

    # relevante recuperado en la posición 4: cuenta para @5 pero no para @3
    qvec = embs[[4]] + 0.4 * embs[[0]] + 0.3 * embs[[1]] + 0.2 * embs[[5]] + 0.1 * embs[[2]]
    qvec = qvec / np.linalg.norm(qvec)
    r = ev.evaluate([query('q', [2])], qvec, embs, ids, meta)[0]
    assert r['top_ids'][:5] == [4, 0, 1, 5, 2]
    assert not r['strict@3'] and r['strict@5'] and r['lax@5'] and not r['lax@3']


def test_un_id_sin_metadatos_falla_en_voz_alta():
    embs, ids, meta = make_world()
    del meta[4]
    with pytest.raises(KeyError):
        ev.evaluate([query('q', [2])], embs[[4]], embs, ids, meta)


def test_summarize_por_dificultad():
    rows = [{'difficulty': 'easy', **{m: True for m in ev.METRICS}},
            {'difficulty': 'easy', **{m: False for m in ev.METRICS}},
            {'difficulty': 'hard', **{m: False for m in ev.METRICS}}]
    total, by_diff = ev.summarize_all(rows)
    assert total['n'] == 3 and total['lax@3'] == pytest.approx(1 / 3)
    assert by_diff['easy']['strict@5'] == 0.5 and by_diff['hard']['lax@5'] == 0.0


def test_load_queries_rechaza_items_incompletos(tmp_path):
    p = tmp_path / 'q.yaml'
    p.write_text(yaml.safe_dump({'queries': [{'id': 'x', 'query': '?', 'relevant': [{'chunk_index': 1}]}]}))
    with pytest.raises(AssertionError):
        ev.load_queries(p)


def test_main_de_punta_a_punta_con_modelo_falso(tmp_path, monkeypatch, capsys):
    embs, ids, meta = make_world()
    chunks = [{'chunk_index': i, 'chapter_num': m['chapter_num'], 'section': m['section'],
               'printed_page_start': m['page_start'], 'printed_page_end': m['page_end']} for i, m in meta.items()]
    (tmp_path / 'chunks.json').write_text(json.dumps(chunks), encoding='utf-8')
    np.savez(tmp_path / 'embs.npz', embeddings=embs, ids=ids, model_name='org/fake-model', normalized=True)
    (tmp_path / 'queries.yaml').write_text(yaml.safe_dump({'queries': [
        {'id': 'q1', 'query': 'uno', 'difficulty': 'easy',
         'relevant': [{'chunk_index': 2, 'chapter_num': 17, 'section': 'Metformina', 'page': 650}]},
        {'id': 'q2', 'query': 'dos', 'difficulty': 'hard',
         'relevant': [{'chunk_index': 2, 'chapter_num': 17, 'section': 'Metformina', 'page': 650}]},
    ]}, allow_unicode=True), encoding='utf-8')

    class FakeModel:
        def __init__(self, name): assert name == 'org/fake-model'
        def encode(self, texts, normalize_embeddings=True):
            far = embs[6:11].sum(axis=0) / np.sqrt(5)                       # se parece solo a los chunks de relleno
            return np.stack([embs[3] if t == 'uno' else far for t in texts])   # q1 acierta (lax), q2 falla

    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=FakeModel))
    monkeypatch.setattr(ev, 'PATH_CHUNKS', tmp_path / 'chunks.json')
    monkeypatch.setattr(ev, 'PATH_EMBS', tmp_path / 'embs.npz')
    monkeypatch.setattr(ev, 'PATH_QUERIES', tmp_path / 'queries.yaml')
    monkeypatch.setattr(ev, 'DIR_EVAL', tmp_path / 'eval')
    monkeypatch.setattr(ev, 'PATH_LOG', tmp_path / 'docs' / 'eval_log.md')

    ev.main('baseline')
    out = capsys.readouterr().out
    assert 'TOTAL' in out and '[q2 · hard]' in out and '[q1' not in out          # solo q2 aparece como fallo
    detail = json.loads(next((tmp_path / 'eval').glob('*.json')).read_text(encoding='utf-8'))
    assert detail['note'] == 'baseline' and detail['total']['lax@3'] == 0.5
    assert 'text' not in json.dumps(detail['queries'])                            # el detalle no lleva texto del libro
    log = (tmp_path / 'docs' / 'eval_log.md').read_text(encoding='utf-8')
    assert log.count('\n| 20') == 1 and 'baseline' in log
