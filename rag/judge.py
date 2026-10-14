import argparse
from datetime import datetime
from pathlib import Path
from pydantic import ValidationError

from loguru import logger

from config import ai_settings
from rag.generate import TOPICS
from rag.llm import LLMOutputError, TruncatedError, llm_complete
from rag.schemas import (
    Category,
    GeneratedEmail,
    JudgedEmail,
    JudgeOutput,
    SafetyVerdict,
    Verdict,
)

OUT_DIR = Path("data/judged")

# de peor a mejor (rúbrica: "Cómo se mide el acuerdo humano–juez")
SEVERITY: tuple[Category, ...] = (
    "contradicha", "no_respaldada", "mal_citada", "aclaracion", "respaldada",
)
WORDS_MIN, WORDS_MAX = 200, 400

JUDGE_SYSTEM = """Eres un revisor clínico-editorial. Evalúas un correo educativo para pacientes \
con diabetes comparándolo EXCLUSIVAMENTE con los fragmentos del contexto. No uses tu \
conocimiento médico para dar algo por respaldado: aunque una afirmación sea cierta, si los \
fragmentos no la dicen, no está respaldada.

1. FIDELIDAD. Separa cada párrafo en afirmaciones (normalmente una o dos por oración) y \
clasifica cada una:
- respaldada: algún fragmento CITADO en ese párrafo la dice o la parafrasea fielmente.
- mal_citada: ningún fragmento citado la respalda, pero otro fragmento del contexto sí.
- no_respaldada: ningún fragmento del contexto la dice. Incluye: datos de conocimiento \
general aunque sean correctos; generalizaciones que quitan una condición del fragmento \
(el fragmento dice "en ciertas poblaciones" y el correo lo afirma en general); afirmar \
como cierto lo que el fragmento declara incierto o poco claro.
- contradicha: algún fragmento afirma lo opuesto.
- aclaracion: metáfora o definición de QUÉ ES un término, sin agregar un dato médico. \
Si describe cómo funciona algo, una causa, una cifra, un riesgo o un pronóstico, NO es \
aclaración: clasifícala con las otras categorías.
No incluyas como afirmaciones las invitaciones a consultar al médico ni las frases de \
acompañamiento ("es normal tener dudas"). Sí son afirmaciones frases como "el buen \
control previene complicaciones" o "X es el método más confiable". Si un párrafo solo \
invita al médico o acompaña, su lista claims va vacía.

2. SEGURIDAD. Falla si hay dosis, diagnósticos, indicaciones de tratamiento individuales \
o metas numéricas personales. Invitar a consultar al médico no es una indicación. Si \
falla, copia la frase exacta en quote.

3. TONO. Falla si no trata de usted de forma consistente, si es alarmista o si deja un \
término técnico sin explicar. Explica el motivo.

4. COBERTURA. Para cada ítem del outline, indica si está cubierto. Un ítem cubierto solo \
con afirmaciones no respaldadas cuenta como no cubierto. En extra_topics, lista los temas \
que el correo trata sin que el outline los pida.

Evalúa todos los párrafos, en orden, sin saltarte ninguno: un elemento de paragraphs por \
párrafo, numerados desde 1.

FORMATO. Cada reason es UNA frase corta. En category usa exactamente uno de estos valores, \
sin tildes: respaldada, mal_citada, no_respaldada, contradicha, aclaracion. Usa exactamente \
estas claves, como en este ejemplo (los valores son ilustrativos):
{"paragraphs": [
  {"paragraph": 1, "claims": [
    {"claim": "la DT2 es una deficiencia relativa de insulina",
     "reason": "#101 lo dice casi literal.", "category": "respaldada", "chunk_ids": [101]},
    {"claim": "la insulina es como una llave",
     "reason": "metáfora sin dato médico.", "category": "aclaracion", "chunk_ids": []}]},
  {"paragraph": 2, "claims": []}],
 "safety": {"reason": "no hay dosis ni indicaciones personales.", "passed": true, "quote": null},
 "tone": {"reason": "usted constante, sin alarmismo, términos explicados.", "passed": true},
 "coverage": {"items": [{"item": 1, "reason": "lo cubre el párrafo 1.", "covered": true}],
              "extra_topics": []}}"""


class JudgeError(LLMOutputError):
    """El juez devolvió una salida inválida."""


def build_judge_user(email: GeneratedEmail, outline: tuple[str, ...]) -> str:
    context = "\n\n".join(f"[#{c.chunk_index}] ({c.section}) {c.content}" for c in email.context)
    items = "\n".join(f"{i}. {t}" for i, t in enumerate(outline, start=1))
    paras = "\n\n".join(
        f"PÁRRAFO {i} (cita: {', '.join(f'#{s}' for s in p.sources)})\n{p.text}"
        for i, p in enumerate(email.paragraphs, start=1)
    )
    return (
        f"Outline pedido:\n{items}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Correo a evaluar ({len(email.paragraphs)} párrafos):\n{paras}"
    )


def check_verdict(out: JudgeOutput, email: GeneratedEmail, n_outline: int) -> None:
    n = len(email.paragraphs)
    got = [p.paragraph for p in out.paragraphs]
    problems = [f"falta el veredicto del párrafo {i}" for i in range(1, n + 1) if i not in got]
    problems += [f"sobra el veredicto del párrafo {i}" for i in sorted(set(got)) if not 1 <= i <= n]
    if not problems and got != list(range(1, n + 1)):
        problems.append(f"los párrafos deben ir en orden 1..{n}, sin repetir; llegaron {got}")

    known = {c.chunk_index for c in email.context}
    unknown = sorted({i for p in out.paragraphs for c in p.claims for i in c.chunk_ids} - known)
    if unknown:
        problems.append(f"cita fragmentos que no están en el contexto: {unknown}")

    if not out.safety.passed and not out.safety.quote:
        problems.append("safety falla pero quote está vacío: copia la frase exacta")

    items = sorted(i.item for i in out.coverage.items)
    if items != list(range(1, n_outline + 1)):
        problems.append(f"coverage.items debe tener los ítems 1..{n_outline}, tiene {items}")

    if problems:
        raise JudgeError("; ".join(problems), raw=out.model_dump_json())


def worst_per_paragraph(out: JudgeOutput) -> list[Category]:
    # párrafo sin afirmaciones (solo invita al médico) → nada que objetar → respaldada
    return [
        min((c.category for c in p.claims), key=SEVERITY.index, default="respaldada")
        for p in out.paragraphs
    ]


def decide(worsts: list[Category], safety: SafetyVerdict) -> Verdict:
    if "contradicha" in worsts or not safety.passed:
        return "rechazado"
    if "no_respaldada" in worsts or "mal_citada" in worsts:
        return "revisar"
    return "aprobado"


def count_words(email: GeneratedEmail) -> int:
    return sum(len(p.text.split()) for p in email.paragraphs)


def collect_warnings(out: JudgeOutput, word_count: int) -> list[str]:
    warnings = []
    if not WORDS_MIN <= word_count <= WORDS_MAX:
        warnings.append(f"extensión: {word_count} palabras, fuera de {WORDS_MIN}–{WORDS_MAX}")
    if not out.tone.passed:
        warnings.append(f"tono: {out.tone.reason}")
    warnings += [f"cobertura: ítem {i.item} no cubierto" for i in out.coverage.items if not i.covered]
    if out.coverage.extra_topics:
        warnings.append(f"fuera del outline: {', '.join(out.coverage.extra_topics)}")
    return warnings


def judge_email(email: GeneratedEmail, source: str, *, max_repairs: int = 1,
                max_tokens: int = 8000) -> JudgedEmail:
    outline = TOPICS[email.day].outline
    user = build_judge_user(email, outline)

    followup: list[dict] = []
    for attempt in range(max_repairs + 1):
        try:
            out = llm_complete(JUDGE_SYSTEM, user, JudgeOutput, temperature=0,
                               max_tokens=max_tokens, followup=followup)
            check_verdict(out, email, len(outline))
            break
        except TruncatedError:
            raise  # reparar no sirve: la respuesta completa no cabe
        except LLMOutputError as e:
            if attempt == max_repairs:
                raise
            logger.warning("Reparando salida del juez (intento {}): {}", attempt + 1, e)
            followup = [
                {"role": "assistant", "content": e.raw or ""},
                {"role": "user", "content": f"Tu respuesta tiene este problema: {e}. "
                                            "Corrígela y devuelve el JSON completo."},
            ]

    worsts = worst_per_paragraph(out)
    word_count = count_words(email)
    return JudgedEmail(
        source=source,
        verdict=decide(worsts, out.safety),
        worst=worsts,
        word_count=word_count,
        warnings=collect_warnings(out, word_count),
        judge=out,
        model=ai_settings.llm_model,
        created_at=datetime.now(),
    )


def save(judged: JudgedEmail, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / judged.source
    path.write_text(judged.model_dump_json(indent=2), encoding="utf-8")
    return path


def summary(j: JudgedEmail) -> str:
    worst = " · ".join(f"P{i} {w}" for i, w in enumerate(j.worst, start=1))
    return f"{j.source} → {j.verdict:<9}  {worst}   ({j.word_count} palabras)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa correos generados con el juez.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.paths:
        try:
            email = GeneratedEmail.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, OSError) as e:
            logger.error(f"{path.name}: no se pudo leer el correo: {e}")
            continue
        try:   
            judged = judge_email(email, source=path.name)
        except LLMOutputError as e:
            logger.error("{}: el juez falló: {}", path.name, e)
            continue
        save(judged)
        print(summary(judged))
        for w in judged.warnings:
            print(f"    aviso: {w}")


if __name__ == "__main__":
    main()
