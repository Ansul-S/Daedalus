"""The judging contract: what the judge is shown, asked, and allowed to see.

The judge is measured against the human labels, so it is given the same
instrument and the same evidence. Three properties follow, and none of them is a
free choice.

**The judge sees exactly what the labeller saw.** The question and the chunks it
cites, and nothing else. The requested type, the requested difficulty, the
selection rank and the grounding quote are withheld here for the same reasons
`docs/PHASE-6-RUBRICS.md` section 2 withholds them from the labeller: each would
anchor the judgement on something other than the question and its material. A
judge shown more than the human was would not be disagreeing about the same
thing.

**The rubric text is imported, not restated.** It comes from
`daedalus.labelling`, which is where the labeller read it. Copying it here would
create two versions of a frozen document that could drift apart silently, and a
drift would invalidate the comparison without failing a test.

**One rubric per call.** The labeller made three separate passes so that a
judgement on one rubric could not anchor a judgement on another. Asking for all
three in one response would put a halo on one side of a comparison that exists to
measure the other, so the judge pays the same cost the human did.

The response schema requires a JSON object with a string ``grade``, but does
**not** constrain that string to the rubric's scale. That is deliberate. Protocol
section 7 step 7 records a judge answer off its own scale as a finding about the
judge — one of the things this phase sets out to measure — so an enum here would
suppress the evidence rather than collect it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from daedalus.labelling import RUBRIC_REMINDERS

#: Identifies the judging contract a score was produced under. Every setting
#: below is part of it: change any of them and this must change too, or scores
#: produced under different conditions will pool silently.
JUDGE_VERSION = "j6-v1"

#: The judge model, and the decoding settings that make a run reproducible.
#: Temperature 0 because reproducibility is worth more here than a sample of the
#: judge's variability; the ``run`` column exists to add that later without
#: redoing this.
JUDGE_MODEL = "qwen3:8b"
TEMPERATURE = 0.0
NUM_PREDICT = 200
KEEP_ALIVE = "30m"
THINK = False

#: The rubrics, in the order the labeller worked through them.
RUBRICS = ("groundedness", "relevance", "difficulty")

#: What a valid answer looks like on each scale, stated to the judge so that an
#: off-scale answer is a failure to follow an instruction it was given rather
#: than a failure to guess an unstated convention.
ANSWER_FORMATS = {
    "groundedness": 'exactly one of "0", "1" or "2"',
    "relevance": 'exactly one of "0", "1" or "2"',
    "difficulty": 'exactly one of "easy", "medium", "hard" or "unusable"',
}

#: Ollama structured-output schema. A string, not an enum: see the module
#: docstring.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"grade": {"type": "string"}},
    "required": ["grade"],
}

SYSTEM_PROMPT = """\
You are grading interview questions against a fixed rubric. You are not \
answering the questions, improving them, or explaining them.

Rules you must follow:

1. Apply the rubric exactly as written. Do not add criteria of your own.
2. Judge only the question and the material shown. You have no access to any \
other part of the source, and you must not use knowledge from outside it.
3. Return one grade and nothing else.
"""


def render_question(text: str, cited: list[tuple[int, str, str]]) -> str:
    """Render one question and its cited chunks as the labeller saw them.

    ``cited`` is (ordinal, kind, chunk text) in the order the chunks were cited.
    The chunk text is passed in full: the labelling display truncates long
    chunks but offers the whole of one on a keypress, so full text is what the
    labeller could see, and a judge shown less would be judging less evidence.
    """
    blocks = "\n".join(
        f"\n[chunk {ordinal}] {kind}\n{body}" for ordinal, kind, body in cited
    )
    return f"""\
QUESTION

{text.strip()}

CITED MATERIAL — judge against this only
{blocks}
"""


def build_messages(
    rubric: str, text: str, cited: list[tuple[int, str, str]]
) -> list[dict[str, str]]:
    """Build the chat messages that ask for one rubric's grade on one question."""
    if rubric not in RUBRICS:
        raise ValueError(f"unknown rubric {rubric!r}")

    user = f"""\
RUBRIC

{RUBRIC_REMINDERS[rubric].strip()}

{render_question(text, cited)}
Return JSON: {{"grade": <{ANSWER_FORMATS[rubric]}>}}
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def judging_seed(question_id: int, rubric: str) -> int:
    """Return the decoding seed for one question on one rubric.

    Derived from the question, the rubric and the contract version, so a rerun
    reproduces the same score and the seed is not a free parameter anyone could
    tune towards a better agreement figure.
    """
    key = f"{question_id}:{rubric}:{JUDGE_VERSION}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def judging_options(question_id: int, rubric: str) -> dict[str, Any]:
    """Return the Ollama decoding options for one judgement."""
    return {
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "seed": judging_seed(question_id, rubric),
    }


def params_hash() -> str:
    """Return a hash of everything that fixes how judging behaves."""
    payload = json.dumps(
        {
            "judge_version": JUDGE_VERSION,
            "model": JUDGE_MODEL,
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "think": THINK,
            "system": SYSTEM_PROMPT,
            "rubrics": {rubric: RUBRIC_REMINDERS[rubric] for rubric in RUBRICS},
            "answer_formats": ANSWER_FORMATS,
            "schema": RESPONSE_SCHEMA,
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()
