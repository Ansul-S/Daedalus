"""Measure search against the questions' own passages and write the retrieval report.

Run from the repo root:
    make eval-retrieval

Every question with saved passages is searched in each mode (vector, full-text, hybrid), and
the report scores how near the top its passages come, for the asked, passage and hand-written
sets apart (see app/evaluation/retrieval.py). Two files in data/eval/ add what the database does
not hold: hand-questions.toml, the hand-written questions and their passages, and
also-answers.toml, passages confirmed to answer a question beyond its own. Without them the
report has no hand-written set and no lenient score, and says so.

The report is printed and saved to data/reports/retrieval-<date>.md. It needs the local
embedding model for the vector and hybrid modes; without Ollama only full-text is measured.
No model is asked anything else, so it spends no quota.
"""

import argparse
import asyncio
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import SessionFactory, engine
from app.evaluation.retrieval import (
    ASKED,
    CURRENT,
    HAND,
    PASSAGE,
    TOP,
    FusionVerdict,
    LabelFileError,
    Query,
    Scores,
    also_answers,
    first_relevant,
    fuse,
    generated_queries,
    hand_queries,
    judge_fusion,
    moved,
    score,
    unsearchable,
)
from app.llm.embeddings import Embedder, EmbeddingError
from app.llm.models import embedding_model
from app.retrieval.fusion import RRF_K
from app.retrieval.search import CANDIDATES, keyword_ranking, vector_ranking

SET_NAMES = {
    ASKED: "Asked: the questions practice serves",
    PASSAGE: "About their passage: asked, plus those turned down but still about their passage",
    HAND: "Hand-written: the Phase 1 questions, labelled by hand",
}


@dataclass
class Measurement:
    day: date
    chunks: int
    documents: int
    embedding_model: str | None  # None when the embedder could not be reached
    queries: list[Query]
    left_out: int
    also: Mapping[str, frozenset[int]]
    keyword: dict[str, list[int]]
    vector: dict[str, list[int]] | None
    document_of: dict[int, str]  # chunk id -> document title
    unsearchable: set[int]
    notes: list[str] = field(default_factory=list)

    def relevant(self, query: Query, lenient: bool) -> frozenset[int]:
        return query.gold | self.also.get(query.id, frozenset()) if lenient else query.gold

    def modes(self) -> dict[str, Callable[[Query], list[int]]]:
        modes: dict[str, Callable[[Query], list[int]]] = {}
        if self.vector is not None:
            vector = self.vector
            modes["Vector"] = lambda query: vector[query.id]
        modes["Full-text"] = lambda query: self.keyword[query.id]
        if self.vector is not None:
            vector = self.vector
            modes["Hybrid"] = lambda query: fuse(vector[query.id], self.keyword[query.id])
        return modes


def load_labels(folder: Path, notes: list[str]) -> tuple[list[Query], dict[str, frozenset[int]]]:
    hand: list[Query] = []
    also: dict[str, frozenset[int]] = {}
    hand_path, also_path = folder / "hand-questions.toml", folder / "also-answers.toml"
    if hand_path.exists():
        try:
            hand = hand_queries(hand_path.read_text(encoding="utf-8"))
        except LabelFileError as exc:
            raise LabelFileError(f"{hand_path}: {exc}") from exc
    else:
        notes.append(f"No {hand_path.name}: the hand-written set is not measured.")
    if also_path.exists():
        try:
            also = also_answers(also_path.read_text(encoding="utf-8"))
        except LabelFileError as exc:
            raise LabelFileError(f"{also_path}: {exc}") from exc
    else:
        notes.append(f"No {also_path.name}: lenient scores equal strict ones.")
    return hand, also


async def measure(
    session: AsyncSession,
    embedder: Embedder | None,
    model_name: str,
    hand: list[Query],
    also: dict[str, frozenset[int]],
    notes: list[str],
    day: date,
) -> Measurement:
    generated, left_out = await generated_queries(session)
    queries = generated + hand
    current = Chunk.superseded_at.is_(None)
    chunks = await session.scalar(select(func.count()).select_from(Chunk).where(current))
    documents = await session.scalar(
        select(func.count(func.distinct(Chunk.document_id))).where(current)
    )
    titles = await session.execute(
        select(Chunk.id, Document.title).join(Document, Chunk.document_id == Document.id)
    )
    labelled = {chunk for query in queries for chunk in query.gold}
    labelled |= {chunk for chunks in also.values() for chunk in chunks}

    keyword: dict[str, list[int]] = {}
    vector: dict[str, list[int]] | None = {} if embedder is not None else None
    for query in queries:
        keyword[query.id] = await keyword_ranking(session, query.text, CANDIDATES)
        if vector is not None and embedder is not None:
            try:
                embedded = await embedder.embed_query(query.text)
            except EmbeddingError as exc:
                notes.append(f"Vector and hybrid search not measured: {exc}.")
                vector = None
                continue
            vector[query.id] = await vector_ranking(session, embedded, CANDIDATES)
    return Measurement(
        day=day,
        chunks=chunks or 0,
        documents=documents or 0,
        embedding_model=model_name if vector is not None else None,
        queries=queries,
        left_out=left_out,
        also=also,
        keyword=keyword,
        vector=vector,
        document_of=dict(titles.tuples().all()),
        unsearchable=await unsearchable(session, labelled),
        notes=notes,
    )


def scores_row(name: str, strict: Scores, lenient: Scores) -> str:
    return (
        f"| {name} | {strict.hit1} | {strict.hit5} | {strict.recall5:.2f} | {strict.mrr5:.2f} "
        f"| {lenient.hit1} | {lenient.hit5} | {lenient.mrr5:.2f} |"
    )


def moved_line(
    found: Measurement, subset: Sequence[Query], modes: Mapping[str, Callable], a: str, b: str
) -> str:
    def firsts(mode: str) -> list[int | None]:
        return [first_relevant(modes[mode](query), found.relevant(query, True)) for query in subset]

    up, same, down = moved(firsts(a), firsts(b))
    return f"{b} against {a}: {up} up, {same} the same, {down} down."


def render(found: Measurement, verdict: FusionVerdict | None) -> str:
    modes = found.modes()
    lines = [
        f"# Retrieval report, {found.day.isoformat()}",
        "",
        f"Search over {found.chunks} passages from {found.documents} documents. "
        + (
            f"Vector search uses `{found.embedding_model}`; "
            if found.embedding_model
            else "Vector and hybrid search were not measured; "
        )
        + f"each retriever offers {CANDIDATES} candidates; hybrid fuses the vector ranking with "
        f"full-text's top {CURRENT.keyword_candidates} at weight {CURRENT.keyword_weight:g} "
        f"(reciprocal rank fusion, k {RRF_K}). Scores look at the top {TOP}.",
        "",
        "A question's own passages are its answers (strict). Lenient also counts the passages "
        "confirmed by hand to answer it. hit@1 and hit@5 count questions; Recall@5 is the share "
        "of a question's own passages in its top five; MRR@5 is the mean of 1 / the rank of the "
        "first answer, 0 when none is in the top five.",
    ]
    for name in (ASKED, PASSAGE, HAND):
        subset = [query for query in found.queries if name in query.sets]
        if not subset:
            continue
        lines += [
            "",
            f"## {SET_NAMES[name]} ({len(subset)} questions)",
            "",
            "| Mode | hit@1 | hit@5 | Recall@5 | MRR@5 | Lenient hit@1 | hit@5 | MRR@5 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for mode, rank in modes.items():
            rankings = [rank(query) for query in subset]
            strict = score(rankings, [found.relevant(query, False) for query in subset])
            lenient = score(rankings, [found.relevant(query, True) for query in subset])
            lines.append(scores_row(mode, strict, lenient))
        if "Hybrid" in modes:
            lines += [
                "",
                "Where hybrid puts the first answer, lenient: "
                + moved_line(found, subset, modes, "Vector", "Hybrid")
                + " "
                + moved_line(found, subset, modes, "Full-text", "Hybrid"),
            ]
    lines += by_source(found, modes) + misses(found, modes) + fusion_section(verdict) + notes(found)
    return "\n".join(lines) + "\n"


def by_source(found: Measurement, modes: Mapping[str, Callable]) -> list[str]:
    best = "Hybrid" if "Hybrid" in modes else "Full-text"
    groups: dict[str, list[Query]] = {}
    for query in found.queries:
        sources = {found.document_of.get(chunk, "?") for chunk in query.gold}
        if len(sources) == 1:
            groups.setdefault(sources.pop(), []).append(query)
    lines = [
        "",
        f"## By source ({best.lower()}, lenient, questions from one document, every set)",
        "",
        "| Document | Questions | hit@1 | hit@5 | MRR@5 |",
        "|---|---|---|---|---|",
    ]
    for title, subset in sorted(groups.items()):
        found_scores = score(
            [modes[best](query) for query in subset],
            [found.relevant(query, True) for query in subset],
        )
        lines.append(
            f"| {title[:60]} | {len(subset)} | {found_scores.hit1} | {found_scores.hit5} "
            f"| {found_scores.mrr5:.2f} |"
        )
    return lines


def misses(found: Measurement, modes: Mapping[str, Callable]) -> list[str]:
    best = "Hybrid" if "Hybrid" in modes else "Full-text"
    rows = []
    for query in found.queries:
        if not query.sets & {ASKED, HAND}:
            continue
        ranks = {
            mode: first_relevant(rank(query), found.relevant(query, True))
            for mode, rank in modes.items()
        }
        if ranks[best] != 1:
            cells = " | ".join(str(rank) if rank else "—" for rank in ranks.values())
            rows.append(f"| {query.id} | {cells} | {query.text[:90]} |")
    if not rows:
        return []
    header = " | ".join(modes)
    return [
        "",
        "## Asked and hand-written questions whose first answer is not first "
        f"({best.lower()}, lenient)",
        "",
        "Rank of the first answer in each mode; — when it is not in the 50 candidates.",
        "",
        f"| Question | {header} | Question text |",
        "|---" * (len(modes) + 2) + "|",
        *rows,
    ]


def fusion_section(verdict: FusionVerdict | None) -> list[str]:
    if verdict is None:
        return []
    lines = [
        "",
        "## Fusion variants",
        "",
        "Tuned on the passage questions that are not asked, by lenient MRR@5; the choice is then "
        "judged once on the asked and hand-written questions. It replaces the current setting "
        "only if it raises MRR@5 on the asked questions with more of them up than down, and does "
        "not lower it on the hand-written ones.",
        "",
        "| Variant | MRR@5 on the tuning questions |",
        "|---|---|",
    ]
    for fusion, value in verdict.tuning:
        current = " (current)" if fusion == CURRENT else ""
        lines.append(f"| {fusion.label}{current} | {value:.3f} |")
    up, same, down = verdict.asked_moved
    lines += [
        "",
        f"Chosen: {verdict.chosen.label}. "
        f"Asked MRR@5 {verdict.asked[0]:.3f} → {verdict.asked[1]:.3f} "
        f"({up} up, {same} the same, {down} down); hand-written {verdict.hand[0]:.3f} → "
        f"{verdict.hand[1]:.3f}.",
        "",
        "**Verdict:** "
        + (
            f"adopt {verdict.chosen.label}."
            if verdict.adopted
            else "keep the current setting."
            if verdict.chosen == CURRENT
            else "keep the current setting; the choice does not pass on the asked and "
            "hand-written questions."
        ),
    ]
    return lines


def notes(found: Measurement) -> list[str]:
    lines = [
        f"- {found.left_out} questions turned down because the checker could not answer them "
        "from their passage are left out: they may reach beyond it.",
        f"- {sum(len(chunks) for chunks in found.also.values())} passages confirmed to also "
        f"answer {len(found.also)} questions.",
    ]
    if found.unsearchable:
        ids = ", ".join(str(chunk) for chunk in sorted(found.unsearchable))
        lines.append(
            f"- Labelled passages search can no longer return (re-ingested or deleted): {ids}. "
            "Questions that rely on them score lower until they are re-labelled."
        )
    lines += [f"- {note}" for note in found.notes]
    return ["", "## Notes", "", *lines]


async def main(args: argparse.Namespace) -> int:
    try:
        return await run(args)
    finally:
        await engine.dispose()


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.fake_models:
        print("The report measures the real embedding model: unset FAKE_MODELS.")
        return 1
    notes_found: list[str] = []
    try:
        hand, also = load_labels(settings.data_dir / "eval", notes_found)
    except LabelFileError as exc:
        print(exc)
        return 1
    day = date.today()
    async with SessionFactory() as session, embedding_model(settings) as embedder:
        found = await measure(
            session, embedder, settings.embedding_model, hand, also, notes_found, day
        )
    if not found.queries:
        print("No questions with passages yet: generate some with `make generate`.")
        return 1
    verdict = None
    if found.vector is not None:
        relevant = {query.id: found.relevant(query, True) for query in found.queries}
        verdict = judge_fusion(found.queries, found.vector, found.keyword, relevant)
    report = render(found, verdict)
    print(report)
    path = settings.data_dir / "reports" / f"retrieval-{day.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure search and write the retrieval report.")
    sys.exit(asyncio.run(main(parser.parse_args())))
