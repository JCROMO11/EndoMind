# rag/schemas.py
from pydantic import BaseModel, Field
from datetime import datetime


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


class GeneratedParagraph(BaseModel):
    text: str
    sources: list[int]
    citation: str


class GeneratedEmail(BaseModel):
    day: int
    title: str
    subject: str
    paragraphs: list[GeneratedParagraph]
    context_chunks: list[int]
    model: str
    created_at: datetime