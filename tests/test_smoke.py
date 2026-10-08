import numpy as np

from scripts.smoke import top_k


def unit(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_top_k_encuentra_la_fila_identica_primero():
    M = unit(np.random.default_rng(0).normal(size=(50, 8)))
    idx, scores = top_k(M, M[7], k=3)
    assert idx[0] == 7 and np.isclose(scores[0], 1.0)
    assert list(scores) == sorted(scores, reverse=True)


def test_top_k_sin_normalizar_da_el_mismo_orden_que_normalizado():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(40, 6))
    q = rng.normal(size=6)
    esperado, _ = top_k(unit(base), q / np.linalg.norm(q), k=5, normalized=True)
    obtenido, scores = top_k(base * rng.uniform(0.5, 3, size=(40, 1)), q * 4, k=5, normalized=False)
    assert list(esperado) == list(obtenido) and scores.max() <= 1.0 + 1e-9
