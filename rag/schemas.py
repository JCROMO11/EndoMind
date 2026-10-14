# rag/schemas.py
import unicodedata
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field


def _normalize_category(v: object) -> object:
    if not isinstance(v, str):
        return v
    v = unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode()
    return v.strip().lower().replace(" ", "_").replace("-", "_")


Category = Annotated[
    Literal["respaldada", "mal_citada", "no_respaldada", "contradicha", "aclaracion"],
    BeforeValidator(_normalize_category),
]


class Paragraph(BaseModel):
    text: str = Field(
        min_length=20,
        description="Párrafo del correo, en lenguaje claro para pacientes.",
    )
    sources: list[int] = Field(
        min_length=1,
        description="Números de los fragmentos del contexto que respaldan este párrafo "
                    "(el número entre corchetes, p. ej. 10479 para [#10479]).",
    )


class EmailDraft(BaseModel):
    subject: str = Field(
        min_length=5,
        max_length=120,
        description="Asunto del correo: corto, claro y sin alarmismo.",
    )
    paragraphs: list[Paragraph] = Field(
        min_length=2,
        max_length=8,
        description="Cuerpo del correo, en párrafos. Cada uno cita sus fuentes.",
    )
    
class ContextChunk(BaseModel):
    chunk_index: int
    section: str
    chapter: int
    page_start: int
    page_end: int
    content: str
    score: float | None  # None en correos reconstruidos: el score depende de la query, no está en la BD


class GeneratedParagraph(BaseModel):
    text: str
    sources: list[int]
    citation: str


class GeneratedEmail(BaseModel):
    day: int
    title: str
    subject: str
    paragraphs: list[GeneratedParagraph]
    context: list[ContextChunk]
    model: str
    created_at: datetime


class ClaimVerdict(BaseModel):
    claim: str = Field(description="Una afirmación del párrafo, en pocas palabras.")
    reason: str = Field(
        description="Una frase corta: qué dice o no dice el fragmento sobre esta afirmación. "
                    "Escríbelo antes de decidir la categoría."
    )
    category: Category
    chunk_ids: list[int] = Field(
        description="Fragmentos que la respaldan (respaldada, mal_citada) o la contradicen "
                    "(contradicha). Vacío en los demás casos."
    )


class ParagraphVerdict(BaseModel):
    paragraph: int = Field(ge=1, description="Número del párrafo, desde 1.")
    claims: list[ClaimVerdict] = Field(
        description="Vacía si el párrafo solo invita a consultar al médico o acompaña."
    )


class SafetyVerdict(BaseModel):
    reason: str
    passed: bool
    quote: str | None = Field(description="Frase exacta que falla; null si pasa.")


class ToneVerdict(BaseModel):
    reason: str
    passed: bool


class OutlineItemVerdict(BaseModel):
    item: int = Field(ge=1, description="Número del ítem del outline, desde 1.")
    reason: str
    covered: bool


class CoverageVerdict(BaseModel):
    items: list[OutlineItemVerdict]
    extra_topics: list[str] = Field(description="Temas tratados que el outline no pide.")


class JudgeOutput(BaseModel):
    paragraphs: list[ParagraphVerdict]
    safety: SafetyVerdict
    tone: ToneVerdict
    coverage: CoverageVerdict


Verdict = Literal["aprobado", "revisar", "rechazado"]


class JudgedEmail(BaseModel):
    source: str  # nombre del archivo en data/generated/
    verdict: Verdict
    worst: list[Category]  # peor categoría de cada párrafo, en orden
    word_count: int
    warnings: list[str]  # avisos que no cambian el veredicto: extensión, tono, cobertura
    judge: JudgeOutput
    model: str
    created_at: datetime
