"""GSM8K loading and prompt construction."""

from __future__ import annotations

import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_gsm8k(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    """Load GSM8K examples as {"question", "ground_truth"} dicts."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            examples.append(
                {
                    "question": example["question"],
                    "ground_truth": extract_gsm8k_answer(example["answer"]),
                }
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def extract_gsm8k_answer(answer: str) -> str:
    """GSM8K answers look like "{rationale} #### {answer}"; keep the answer."""
    return answer.split("####")[-1].strip()


def load_prompt_template(name: str) -> str:
    """Read one of the templates in lm_alignment/prompts/."""
    return (PROMPTS_DIR / f"{name}.prompt").read_text(encoding="utf-8")


def build_prompts(template: str, questions: list[str]) -> list[str]:
    return [template.format(question=question) for question in questions]
