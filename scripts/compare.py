"""Compara el juez contra las etiquetas asistidas (por ahora, un correo).

Uso: python -m scripts.compare
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.schemas import JudgedEmail

JUDGED_DIR = Path("data/judged_prompt_v1")
LABELS_PATH = Path("data/labels/assisted_v2.yaml")
NOTE_KEY = "nota"  # verifica cómo se llama la llave de la nota en tu YAML


def pass_label(passed: bool) -> str:
    return "pasa" if passed else "falla"


def normalize_judged(judged: JudgedEmail) -> dict:
    """Lleva la salida del juez a la misma forma que una entrada del YAML."""
    return {
        "seguridad": pass_label(judged.judge.safety.passed),
        "tono": pass_label(judged.judge.tone.passed),
        "paragraphs": [{"worst": w} for w in judged.worst],
    }


@dataclass
class EmailResult:
    name: str
    checks: dict[str, tuple[str, str]]  # campo -> (juez, etiqueta)
    paragraphs: list[tuple[int, str, str, str]] = field(default_factory=list)  # (n, juez, etiqueta, nota)
    error: str | None = None


def compare_email(name: str, judged: dict, label: dict) -> EmailResult:
    checks = {k: (judged[k], label[k]) for k in ("seguridad", "tono")}
    j_pars, l_pars = judged["paragraphs"], label["paragraphs"]
    if len(j_pars) != len(l_pars):
        return EmailResult(name, checks, error=f"el juez tiene {len(j_pars)} párrafos y la etiqueta {len(l_pars)}")
    paragraphs = [
        (n, j["worst"], l["worst"], l.get(NOTE_KEY, ""))
        for n, (j, l) in enumerate(zip(j_pars, l_pars), start=1)
    ]
    return EmailResult(name, checks, paragraphs)


def pct(ok: int, total: int) -> str:
    return f"{ok / total:.0%} ({ok}/{total})" if total else "sin datos"


def print_report(r: EmailResult) -> None:
    print(f"\n{'═' * 64}\n  {r.name}\n{'═' * 64}")
    for name, (j, l) in r.checks.items():
        mark = "✓" if j == l else "✗"
        print(f"  {mark} {name:<10} juez: {j:<6}  etiqueta: {l}")

    if r.error:
        print(f"\n  ! {r.error}: párrafos sin comparar")
        return

    agree = sum(j == l for _, j, l, _ in r.paragraphs)
    print(f"\n  Párrafos (peor categoría): acuerdo {pct(agree, len(r.paragraphs))}")
    for n, j, l, note in r.paragraphs:
        mark = "✓" if j == l else "✗"
        print(f"  {mark} P{n}  juez: {j:<14} etiqueta: {l}")
        if j != l and note:
            print(f"        nota: {note}")


def main() -> None:
    name = "day1_20261012_215919.json"
    labels = yaml.safe_load(LABELS_PATH.read_text(encoding="utf-8"))
    judged = JudgedEmail.model_validate_json((JUDGED_DIR / name).read_text(encoding="utf-8"))
    print_report(compare_email(name, normalize_judged(judged), labels[name]))


if __name__ == "__main__":
    main()