"""Tests de chunk.py con un medidor falso (1 palabra = 1 token): no descargan el modelo."""
from ingest.chunk import build_chunks, chunk_section, split_long, split_sentences


def words(text: str) -> int:
    return len(text.split())


def section(texts, name="S", chapter_num=1):
    return {
        "chapter_num": chapter_num,
        "chapter": f"{chapter_num}. CAP",
        "section": name,
        "paragraphs": [{"page": 30 + i, "printed_page": 5 + i, "text": t} for i, t in enumerate(texts)],
    }


# --- split_sentences ---------------------------------------------------------
def test_no_corta_abreviaturas():
    t = "Se usa p. ej. la metformina. Ver Fig. 3-2 para detalles. Aumentan aprox. 10 veces."
    assert split_sentences(t) == [
        "Se usa p. ej. la metformina.",
        "Ver Fig. 3-2 para detalles.",
        "Aumentan aprox. 10 veces.",
    ]


def test_no_corta_items_con_letra_ni_decimales():
    t = "Opciones: a. Gliburida b. Metformina. El pH fue 7.4. Es normal."
    assert split_sentences(t) == ["Opciones: a. Gliburida b. Metformina.", "El pH fue 7.4.", "Es normal."]


def test_corta_tras_punto_y_parentesis_y_signos_de_apertura():
    t = "Dato clave (ver tabla.) Luego sigue. ¿Qué pasa? ¡Nada! Fin."
    assert split_sentences(t) == ["Dato clave (ver tabla.)", "Luego sigue.", "¿Qué pasa?", "¡Nada!", "Fin."]


def test_conserva_el_simbolo_paragrafo():
    assert split_sentences("Ver § 3 del texto. Luego sigue.") == ["Ver § 3 del texto.", "Luego sigue."]


# --- split_long --------------------------------------------------------------
def test_split_long_respeta_el_limite_y_no_pierde_palabras():
    t = "uno dos tres, cuatro cinco seis; siete ocho nueve diez once doce, trece catorce"
    piezas = split_long(t, 5, words)
    assert all(words(p) <= 5 for p in piezas)
    assert " ".join(piezas).split() == t.split()


def test_split_long_por_palabras_cuando_no_hay_separadores():
    t = " ".join(f"w{i}" for i in range(23))
    piezas = split_long(t, 10, words)
    assert [words(p) for p in piezas] == [10, 10, 3]


def test_split_long_palabra_enorme_no_entra_en_bucle():
    assert split_long("x" * 500, 5, lambda s: 99) == ["x" * 500]


# --- chunk_section -----------------------------------------------------------
def test_chunks_no_superan_el_maximo_y_llevan_metadatos():
    s = section(["Uno dos tres cuatro. Cinco seis siete. Ocho nueve diez once.", "Doce trece. Catorce quince dieciséis."],
                name="Mi sección", chapter_num=7)
    chunks = chunk_section(s, max_size=8, measure=words)
    assert chunks and all(words(c["text"]) <= 8 for c in chunks)
    for c in chunks:
        assert (c["chapter_num"], c["section"]) == (7, "Mi sección")
        assert c["page_start"] <= c["page_end"] and c["printed_page_start"] == c["page_start"] - 25


def test_solapamiento_repite_la_ultima_oracion():
    s = section(["Aaa bbb ccc. Ddd eee fff. Ggg hhh iii."])
    chunks = chunk_section(s, max_size=7, measure=words)
    assert [c["text"] for c in chunks] == ["Aaa bbb ccc. Ddd eee fff.", "Ddd eee fff. Ggg hhh iii."]


def test_oracion_larga_se_parte_y_ningun_chunk_pasa_el_limite():
    larga = ", ".join(f"cláusula{i} con tres palabras" for i in range(12)) + "."
    chunks = chunk_section(section([larga]), max_size=10, measure=words)
    assert len(chunks) > 1 and all(words(c["text"]) <= 10 for c in chunks)


def test_no_pierde_texto_de_la_seccion():
    textos = ["Alfa beta gamma. Delta epsilon zeta eta. Theta iota.", "Kappa lambda mu nu xi omicron pi."]
    chunks = chunk_section(section(textos), max_size=6, measure=words)
    en_chunks = set(" ".join(c["text"] for c in chunks).split())
    assert en_chunks == set(" ".join(textos).split())


def test_las_secciones_no_se_mezclan_y_el_indice_es_global():
    a = chunk_section(section(["Aaa bbb ccc. Ddd eee fff."], name="A"), max_size=4, measure=words)
    b = chunk_section(section(["Ggg hhh iii. Jjj kkk lll."], name="B"), max_size=4, measure=words)
    todos = build_chunks([a, b], measure=words)
    assert [c["chunk_index"] for c in todos] == list(range(len(todos)))
    assert all(c["section"] == "A" for c in todos[:len(a)]) and all(c["section"] == "B" for c in todos[len(a):])
    assert all("Aaa" not in c["text"] for c in todos[len(a):])


# --- encabezado --------------------------------------------------------------
from ingest.chunk import chunk_all, embed_text, make_header, variant_paths  # noqa: E402


def test_header_quita_numero_y_mayusculas_y_une_con_flecha():
    assert make_header("17. DIABETES MELLITUS", "Metformina", words, 24) == "Diabetes mellitus › Metformina"


def test_header_no_repite_cuando_seccion_y_capitulo_son_iguales():
    assert make_header("5. TIROIDES", "TIROIDES", words, 24) == "Tiroides"


def test_header_respeta_el_tope_recortando_la_seccion():
    h = make_header("17. DIABETES", "uno dos tres cuatro cinco seis siete", words, 5)
    assert words(h.replace("›", "")) <= 5 and h.startswith("Diabetes")


def test_embed_text_solo_antepone_si_hay_encabezado():
    assert embed_text({"text": "abc"}) == "abc"
    assert embed_text({"text": "abc", "header": ""}) == "abc"
    assert embed_text({"text": "abc", "header": "Cap › Sec"}) == "Cap › Sec\nabc"


def test_chunk_all_descuenta_el_encabezado_del_presupuesto():
    texto = " ".join(f"p{i}." for i in range(300))                     # 300 oraciones de 1 palabra
    sec = section([texto], "Metformina", 17)
    sin = chunk_all([sec], header=False, max_tokens=100, measure=words)
    con = chunk_all([sec], header=True, max_tokens=100, measure=words)
    assert all(c["n_tokens"] <= 100 for c in sin + con)                  # n_tokens ya incluye el encabezado
    assert max(c["n_tokens"] for c in con) <= 100
    assert all(c["header"] and embed_text(c).startswith(c["header"]) for c in con)
    assert all(c["header"] == "" for c in sin)
    assert len(con) >= len(sin)                                          # menos cuerpo por chunk -> no menos chunks
    assert [c["chunk_index"] for c in con] == list(range(len(con)))


def test_variant_paths():
    base = variant_paths()
    h = variant_paths("header")
    assert base[0].name == "greenspan_chunks.json" and base[1].name == "greenspan_embs.npz"
    assert h[0].name == "greenspan_chunks_header.json" and h[1].name == "greenspan_embs_header.npz"


def test_report_cuenta_contra_el_presupuesto_de_la_variante():
    from ingest.chunk import report
    sec = section(["uno dos tres."])
    chunks = [{"chunk_index": 0, "chapter_num": 1, "section": "S", "printed_page_start": 5,
               "printed_page_end": 5, "n_tokens": 110, "text": "uno dos tres."}]
    assert "Chunks > 100 (presupuesto)    : 1" in report(chunks, [sec], max_tokens=100)
