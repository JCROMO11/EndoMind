from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse

from loguru import logger

from config import ai_settings
from rag.llm import LLMOutputError, llm_complete
from rag.retrieve import Hit, retrieve
from rag.schemas import EmailDraft, GeneratedEmail, GeneratedParagraph, ContextChunk

MAX_CONTEXT_CHUNKS = 10
BOOK_SHORT = "Greenspan"  # mientras Hit no traiga el libro
OUT_DIR = Path("data/generated")

SYSTEM = """Eres el redactor de correos educativos del consultorio de un médico \
endocrinólogo. Escribes para pacientes recién diagnosticados, sin formación médica.

Reglas:
1. Usa ÚNICAMENTE la información de los fragmentos del contexto. No agregues datos \
de tu conocimiento general, aunque sean correctos.
2. En cada párrafo, indica en "sources" solo los números de los fragmentos que \
respaldan ese párrafo (por ejemplo, 9582 para [#9582]).
3. Si el contexto no cubre algún subtema pedido, omítelo. No lo completes.
4. No des dosis, indicaciones de tratamiento individuales ni diagnósticos. Cuando sea \
pertinente, invita al paciente a resolver sus dudas personales con su médico.
5. Escribe con tus propias palabras; no copies frases del texto fuente. Explica cada \
término técnico la primera vez que aparezca.
6. Tono cálido, respetuoso y sin alarmismo. Trata al paciente de usted. Escribes en \
nombre del equipo del consultorio, no como el médico.
7. Extensión: entre 3 y 5 párrafos, unas 300 palabras en total. Escribe en español.
8. No escribas saludo, bienvenida ni despedida: solo el contenido educativo. \
Esos textos se agregan aparte."""


@dataclass(frozen=True)
class Topic:
    day: int
    title: str
    chapter: int
    queries: tuple[str, ...]   # para el buscador, en lenguaje clínico
    outline: tuple[str, ...]   # para el lector: lo que el correo debe cubrir


DAY1 = Topic(
    day=1,
    title="Qué es la diabetes tipo 2",
    chapter=17,
    queries=(
        "definición y fisiopatología de la diabetes tipo 2",
        "resistencia a la insulina y disfunción de las células beta",
        "factores de riesgo de la diabetes tipo 2",
    ),
    outline=(
        "Qué es la diabetes tipo 2.",
        "El papel de la insulina y de las células beta.",
        "Qué factores aumentan el riesgo de desarrollarla.",
    ),
)

DAY7 = Topic(
    day=7,
    title="Cómo se mide y se controla la diabetes",
    chapter=17,
    queries=(
        "hemoglobina glucosilada A1c como medida del control glucémico",
        "automonitoreo de la glucosa capilar",
        "glucosa plasmática en ayunas y posprandial",
    ),
    outline=(
        "Qué es la hemoglobina glucosilada (A1c) y qué refleja.",
        "Cómo se mide la glucosa en casa y para qué sirve.",
        "Por qué es útil hacer controles periódicos con el médico.",
    ),
)

DAY30 = Topic(
    day=30,
    title="Complicaciones de la diabetes y cómo prevenirlas",
    chapter=17,
    queries=(
        "retinopatía diabética",
        "nefropatía diabética",
        "neuropatía diabética y pie diabético",
        "enfermedad cardiovascular y aterosclerosis en diabetes",
    ),
    outline=(
        "Qué son las complicaciones a largo plazo de la diabetes.",
        "Cómo puede afectar los ojos, los riñones y los nervios.",
        "Su relación con el corazón y los vasos sanguíneos.",
        "Que el buen control y las revisiones periódicas ayudan a prevenirlas.",
    ),
)

TOPICS = {t.day: t for t in (DAY1, DAY7, DAY30)}


class CitationError(LLMOutputError):
    """El JSON es válido, pero cita fragmentos que no estaban en el contexto."""


def gather_hits(topic: Topic, k_per_query: int = 5) -> list[Hit]:
    hits = [h for q in topic.queries for h in retrieve(q, k=k_per_query, chapter=topic.chapter)]
    # orden ascendente: al armar el dict, el de mayor score es el último en escribirse
    unique = {h.chunk_index: h for h in sorted(hits, key=lambda h: h.score)}.values()
    top = sorted(unique, key=lambda h: h.score, reverse=True)[:MAX_CONTEXT_CHUNKS]
    return sorted(top, key=lambda h: h.chunk_index)  # orden de lectura del libro


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[#{h.chunk_index}] ({h.section}) {h.content}" for h in hits)


def build_user(topic: Topic, context: str) -> str:
    outline = "\n".join(f"- {item}" for item in topic.outline)
    return (
        f'Escribe el correo del día {topic.day} de la secuencia educativa: "{topic.title}".\n\n'
        f"Debe cubrir, en este orden:\n{outline}\n\n"
        f"Contexto:\n{context}"
    )


def check_citations(draft: EmailDraft, allowed: set[int]) -> None:
    bad = [
        f"párrafo {i}: #{s}"
        for i, p in enumerate(draft.paragraphs, start=1)
        for s in p.sources
        if s not in allowed
    ]
    if bad:
        raise CitationError(
            f"citas que no están en el contexto: {', '.join(bad)}",
            raw=draft.model_dump_json(),
        )


def format_citation(sources: list[int], by_id: dict[int, Hit]) -> str:
    pages: dict[int, set[int]] = {}
    for s in sources:
        h = by_id[s]
        pages.setdefault(h.chapter_num, set()).update(
            range(h.printed_page_start, h.printed_page_end + 1)
        )
    parts = [
        f"cap. {ch}, p. {', '.join(str(p) for p in sorted(ps))}"
        for ch, ps in sorted(pages.items())
    ]
    return f"{BOOK_SHORT}, " + "; ".join(parts)


def enrich(topic: Topic, draft: EmailDraft, hits: list[Hit]) -> GeneratedEmail:
    by_id = {h.chunk_index: h for h in hits}
    return GeneratedEmail(
        day=topic.day,
        title=topic.title,
        subject=draft.subject,
        paragraphs=[
            GeneratedParagraph(
                text=p.text,
                sources=p.sources,
                citation=format_citation(p.sources, by_id),
            )
            for p in draft.paragraphs
        ],
        context=[
            ContextChunk(
                chunk_index=h.chunk_index, section=h.section, chapter=h.chapter_num,
                page_start=h.printed_page_start, page_end=h.printed_page_end,
                content=h.content, score=h.score,
            )
            for h in hits
        ],
        model=ai_settings.llm_model,
        created_at=datetime.now(),
    )


def generate_email(topic: Topic, *, max_repairs: int = 1) -> GeneratedEmail:
    hits = gather_hits(topic)
    allowed = {h.chunk_index for h in hits}
    user = build_user(topic, build_context(hits))

    followup: list[dict] = []
    for attempt in range(max_repairs + 1):
        try:
            draft = llm_complete(SYSTEM, user, EmailDraft, temperature=0.3, followup=followup)
            check_citations(draft, allowed)
            return enrich(topic, draft, hits)
        except LLMOutputError as e:
            if attempt == max_repairs:
                raise
            logger.warning("Reparando salida del LLM (intento {}): {}", attempt + 1, e)
            followup = [
                {"role": "assistant", "content": e.raw or ""},
                {"role": "user", "content": f"Tu respuesta tiene este problema: {e}. "
                                            "Corrígela y devuelve el JSON completo."},
            ]


def save(email: GeneratedEmail, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"day{email.day}_{email.created_at:%Y%m%d_%H%M%S}.json"
    path.write_text(email.model_dump_json(indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera correos educativos.")
    parser.add_argument("day", type=int, choices=sorted(TOPICS))
    parser.add_argument("--context", action="store_true",
                        help="muestra los chunks recuperados sin llamar al LLM")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    topic = TOPICS[args.day]

    if args.context:
        for h in gather_hits(topic):
            print(f"[#{h.chunk_index}] score={h.score:.3f} p.{h.printed_page_start} "
                  f"({h.section})\n{h.content}\n")
        return

    for i in range(1, args.runs + 1):
        try:
            email = generate_email(topic)
        except LLMOutputError as e:
            logger.error("Corrida {}/{} falló tras la reparación: {}", i, args.runs, e)
            continue
        print(f"--- corrida {i}/{args.runs} · ASUNTO: {email.subject}\n")
        for p in email.paragraphs:
            print(p.text)
            print(f"   [{p.citation}]  {p.sources}\n")
        print("guardado en", save(email), "\n")


if __name__ == "__main__":
    main()