# Daedalus — Project Definition

Reference document. Read when working on architecture or direction; skip for
routine implementation.

Settled scope and constraints live in `docs/PHASE-0.md`; the intended end-state
product lives in `docs/FEATURES.md`; phase order lives in `docs/ROADMAP.md`.
Where this document describes something as open that those have since settled,
they win.

## Problem

People preparing for AI/ML interviews study from material they've already
collected — course PDFs, personal notes, Jupyter notebooks, screenshots of
diagrams. Generic interview-question banks don't match that material, and
generic chatbots will happily generate questions about things the person never
studied and answers the material never supported.

Daedalus generates interview questions grounded in the user's own documents, and
evaluates the user's answers against that same material.

## Pipeline (starting hypothesis, not fixed)

```
Learning material
      ↓
Document ingestion
      ↓
Content extraction
      ↓
Cleaning / normalization
      ↓
Chunking
      ↓
Embedding / indexing
      ↓
Retrieval
      ↓
Interview question generation
      ↓
User answers
      ↓
Answer evaluation
      ↓
Feedback / follow-up questions
```

This may change as the project develops. Nothing here is settled architecture.

## What "grounded" means here

The system distinguishes three kinds of content and never blurs them:

| Kind | Definition |
|---|---|
| **Supported** | Directly backed by retrieved material |
| **Inferred** | A reasonable conclusion drawn from retrieved material |
| **External** | General knowledge not present in the user's documents |

External knowledge isn't forbidden, but it must never be presented as coming
from the uploaded documents. Where practical, generated questions and
evaluations keep a reference back to the source chunks they came from.

Open question: `docs/FEATURES.md` calls for the system to detect wrong or
outdated source material and supply the correct answer instead. That requires
adjudicating between the documents and the model's own knowledge, which is not
the same operation as labelling content external, and it is not yet decided how
that judgement would be made or distinguished from a hallucination.

Example — material containing "Random Forest is an ensemble learning method that
combines multiple decision trees" should be able to produce:

- What is Random Forest?
- Why does Random Forest generally reduce overfitting compared with a single
  decision tree?
- What is bagging and how is it used in Random Forest?
- When would you choose Random Forest over a linear model?
- Here's a scenario — would Random Forest be appropriate, and why?

## Ingestion

The architecture must stay extensible so new formats can be added later.

**PDF** — text, metadata, page information where available.

**Markdown** — headings, paragraphs, lists, code blocks, useful metadata.

**Jupyter notebooks** — markdown cells, code cells, outputs where useful,
notebook metadata where useful.

**Images** — OCR where appropriate. Images should not be flattened to plain text
when their visual structure carries meaning (architecture diagrams, plots,
tables).

**Standalone code files** — `.py` and similar, where structure is carried by the
code rather than by prose or headings.

Across all formats, preserve enough structure to keep this relationship intact:

```
Heading → Concept → Explanation → Code → Output
```

## Retrieval

May eventually combine lexical retrieval, semantic retrieval, metadata
filtering, and reranking.

Decided as of 2026-08-29, for local zero-cost operation: embeddings from
`BAAI/bge-m3`, with `all-MiniLM-L6-v2` retained as an ablation baseline.
Generation and judging run on `qwen3:8b` locally; `llama3.2:3b` is a smoke-test
model for pipeline debugging only. Measured settings are in `docs/PHASE-0.md`.

Still open: the index and lexical layer (SQLite, FTS5, sqlite-vec and
alternatives), whether a reranker earns its cost, and whether hybrid retrieval
is justified by measured failure modes.

Evaluate any proposed technology against: simplicity, performance,
maintainability, explainability, local development constraints, deployment
constraints, and actual project requirements. Popularity is not a reason.

## Question types

Not all questions should be simple recall.

- **Conceptual** — "What is gradient descent?"
- **Explanation** — "Explain why learning rate affects convergence."
- **Comparison** — "Compare batch and stochastic gradient descent."
- **Applied** — "You have a highly imbalanced classification dataset. What would
  you do?"
- **Debugging** — "Validation loss increases while training loss decreases. What
  could be happening?"
- **System design** — "Design a production pipeline for serving this model."
- **Code reasoning** — "Explain what this Python code is doing."
- **Scenario-based** — "95% accuracy but poor recall on the minority class. How
  would you investigate?"

## Answer evaluation

Criteria: factual correctness, relevance, completeness, conceptual
understanding, important missing points, unsupported claims, clarity.

The methodology needs care. An LLM judge is an estimator, not ground truth.
Areas to investigate before trusting scores: reliability, consistency across
runs, judge bias, agreement with human evaluation, false positive and false
negative rates.

## System evaluation

Evaluation is a first-class component of this project, not an afterthought.
Metrics worth measuring once the relevant components exist:

**Retrieval** — Recall@K, Precision@K, MRR, latency.

**Generation** — groundedness, relevance, difficulty, diversity, hallucination
rate.

**Answer evaluation** — agreement with human assessment, consistency, scoring
reliability.

**System** — ingestion latency, query latency, memory usage, storage
requirements.

Every reported number must come from an experiment that actually ran.

## Data model

Candidate entities. These are *suggestions* — discuss whether each is actually
needed before creating it.

```
Document
DocumentVersion
DocumentSection
Chunk
Embedding
Question
QuestionSource
Answer
Evaluation
Session
```

## Defensibility bar

For every non-trivial component, I should be able to answer:

- Why did we build it this way?
- Why this technology?
- What alternatives did we consider?
- What are the limitations?
- How do we evaluate it?
- What could fail?
- How would we improve it?

If a piece of work can't survive those seven questions, it isn't finished.
