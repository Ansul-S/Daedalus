# Evaluation Engine

How Daedalus measures whether its retrieval and generation actually work.

This is developer tooling, not a product feature. Nothing described here runs when a user studies. It exists so that every claim about the system — "hybrid retrieval beats dense-only", "1000-character chunks beat 500" — is backed by a number rather than an impression.

---

## Principles

**The benchmark is fixed.** Queries, labels, and corpus are frozen and committed. If the measuring instrument changes between runs, a difference in results is uninterpretable — improvement and easier questions look identical.

**Labels are human-verified.** A language model drafts candidates; a human accepts, corrects, or rejects each one. Rejecting bad candidates is as much of the work as keeping good ones.

**Negative results are reported.** If hybrid retrieval fails to beat dense-only on some slice, that is a finding. An ablation table where every row improves is evidence of a badly designed experiment, not a good system.

---

## Three Datasets

They answer different questions and fail independently. Keeping them separate is what makes failures diagnosable — generation metrics computed over broken retrieval measure only the generator's willingness to invent.

| | Question | Size | Depends on |
|---|---|---|---|
| **A. Retrieval** | Did we fetch the right chunks? | ~120 queries | Corpus, chunker, retriever |
| **B. Groundedness** | Is the answer supported by what we fetched? | ~50 queries | A working retriever |
| **C. Judge calibration** | Does the answer-grading model agree with a human? | ~50 answers | Nothing else |

Build A first. B is meaningless until A is healthy. C is independent and can be built at any time.

---

## The Corpus

Twelve documents in `corpus/`, committed, spanning all four ingestion paths.

| Type | Files | Source chars | ≈ chunks | Share |
|---|---|---|---|---|
| Notebooks | 3 | 408,025 | ~510 | 62% |
| arXiv PDFs | 3 | 172,643 | ~215 | 26% |
| Course notes | 3 | 66,602 | ~83 | 10% |
| Images | 2 | ~7,000 | ~9 | ~1% |

Two topical clusters: RAG/NLP (~88%) and computer vision (~10%). The concentration is useful — many near-duplicate chunks compete for the same query, which makes retrieval genuinely hard and the benchmark discriminative. It also means a single query often has legitimately relevant spans in a paper, a notebook, *and* a diagram. Label all of them.

### Parsed text is frozen

Extraction output is committed to `backend/eval/parsed/`, and **all labels anchor to those files**, not to the source documents.

This is mandatory, not convenient. Vision-model output is nondeterministic (ADR-009), so re-parsing an image shifts every character offset downstream of it and silently invalidates every label in that document. Freezing also means:

- Re-parsing becomes a deliberate, versioned migration rather than silent corruption.
- CI runs the retrieval evaluation in seconds without OCR, vision models, or GPU — it reads committed text.

Changing the parser requires a dataset version bump and a re-anchoring pass.

---

## Label Anchoring

**Labels never reference chunk IDs.**

The entire purpose of dataset A is ablation — chunk 500 vs 1000 vs 1500, dense vs hybrid, with and without reranking. Chunk IDs change with every chunking configuration, so chunk-keyed labels would need re-labeling for every experiment, which in practice means the experiments stop happening.

Labels reference **character offsets into the frozen parsed text**:

```
doc_id + char_start + char_end
```

These are stable across every chunking configuration, forever.

### Contract on the chunker

Every chunk must record where it came from:

```sql
source_start INTEGER NOT NULL,   -- offset into parsed/<doc_id>.md
source_end   INTEGER NOT NULL,
extraction   TEXT NOT NULL       -- 'text' | 'ocr' | 'vision' | 'notebook'
```

Without these columns the evaluation harness cannot resolve a label to a chunk. This is a hard requirement on the ingestion pipeline, not an optional extra.

`extraction` is what enables metrics sliced by ingestion path — the mechanism that holds the OCR and vision paths accountable instead of assuming they work.

### Resolving a span to chunks

A chunk is **relevant** to a labeled span when their character ranges overlap by at least 50% of the span's length. If no chunk reaches 50%, the single chunk with the largest overlap is used, so every label always resolves to at least one chunk.

Labeled spans should be tight — one to three sentences, roughly 50–500 characters. A span covering half a page makes the 50% rule meaningless.

A short human-readable `quote` is stored alongside the offsets for eyeballing labels during review. **It is never used for matching**; quotes straddle chunk boundaries, offsets do not.

---

## Dataset A — Retrieval

`backend/eval/datasets/retrieval.jsonl`, one JSON object per line.

```json
{
  "id": "ret-0042",
  "query": "Why is the dot product scaled before the softmax in attention?",
  "query_type": "conceptual",
  "source_type": "arxiv",
  "answerable": true,
  "paraphrased": true,
  "split": "dev",
  "relevant_spans": [
    {
      "doc_id": "attention-is-all-you-need",
      "char_start": 14832,
      "char_end": 15104,
      "quote": "for large values of d_k, the dot products grow large in magnitude",
      "grade": 2
    }
  ],
  "reference_answer": "Large d_k pushes dot products into regions where softmax gradients vanish; scaling by 1/sqrt(d_k) keeps variance stable.",
  "verified_by": "human",
  "notes": ""
}
```

`grade`: `2` = essential (the question cannot be answered without this span), `1` = supporting. Two levels cost almost nothing extra to label and unlock nDCG.

### Query stratification

Random questions produce a dataset that discriminates nothing. Each type stresses a different half of hybrid retrieval:

| `query_type` | Share | Stresses | Example |
|---|---|---|---|
| `exact_term` | 20% | BM25 / FTS5 | "What does RMSNorm normalize over?" |
| `conceptual` | 25% | Dense embeddings | "Why does training destabilize without normalization?" |
| `definitional` | 20% | Both | "What is layer normalization?" |
| `multi_hop` | 15% | Top-k depth | "How do attention and positional encoding interact?" |
| `comparative` | 5% | Multi-chunk | "Batch norm vs layer norm?" |
| `unanswerable` | 15% | Refusal behaviour | "How does LoRA reduce fine-tuning memory?" |

If hybrid retrieval does not beat dense-only on the `exact_term` slice, RRF is not earning its complexity — and that is a result worth publishing in the README, not a failure to hide.

`source_type` (`arxiv`, `course_notes`, `notebook`, `image`) enables the per-ingestion-path breakdown. The `image` slice is a **smoke test only** — roughly 9 chunks cannot support a meaningful Recall@5, and any number computed over it should be reported qualitatively.

### The unanswerable set

Fifteen percent of queries have `answerable: false` and no relevant spans. This is the most important slice in the dataset and the one most portfolio projects omit entirely. It measures the failure mode that actually kills retrieval systems in production: **answering confidently from nothing**.

Two kinds, both needed:

- **Out-of-corpus** — real AI/ML questions the material genuinely does not cover. Avoid trivially off-topic questions; "What is the capital of France?" tests nothing. "How does LoRA reduce fine-tuning memory?" is a question a student would plausibly ask that this corpus cannot answer.
- **Near-miss** — questions that retrieve plausible-looking chunks which do not actually answer them. These are the hard cases and where real systems fail.

The system is expected to refuse. Measured as **false-answer rate**; target near zero, and report it prominently.

### Dev / held-out split

`split: "dev"` (~90 queries) or `"test"` (~30).

There is no training here, so there is no classic leakage. There is a subtler and very real problem: chunk size, `top_k`, `RRF_K`, and prompts will be tuned against these queries hundreds of times, and every tuning decision fits the system to the specific queries it was tuned on. That is overfitting without training.

Tune freely against `dev`. Touch `test` a handful of times total, ideally only when reporting final numbers.

Both splits are stratified across every `query_type` and `source_type`. Be honest in reporting about what 30 queries buys: it is a check against gross overfitting, not a precise measurement.

### Metrics

Let `R` be the set of chunks resolved from a query's relevant spans, and `K` the top-k retrieved chunks.

| Metric | Definition |
|---|---|
| **Recall@k** | Fraction of *essential* (grade 2) spans with at least one resolved chunk in `K`. Primary metric — top-k is literally what the generator sees. |
| **Hit@k** | 1 if any relevant chunk appears in `K`. Coarser, useful for the smoke-test slices. |
| **MRR** | Reciprocal rank of the first relevant chunk, averaged over queries. |
| **nDCG@10** | Standard formulation over grades 2 / 1 / 0. |
| **False-answer rate** | On the unanswerable slice: fraction where the system answered instead of refusing. |

Reported at k=5 and k=10, overall and sliced by `query_type` and `source_type`.

**Statistical caveat, and it belongs in the README too:** with ~120 queries, differences below roughly 8 percentage points are not distinguishable from noise. Report bootstrap confidence intervals alongside point estimates. Declaring victory on a two-point difference is the fastest way to undermine an otherwise good ablation table.

### Target ablations

| Configuration | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|
| Dense only (BGE-M3) | | | |
| Lexical only (FTS5 BM25) | | | |
| Hybrid + RRF | | | |
| Hybrid + RRF + reranker | | | |
| chunk 500 / 1000 / 1500 | | | |
| OCR quality sweep (150 / 100 / 72 DPI) | | | |

---

## Dataset B — Groundedness

`backend/eval/datasets/groundedness.jsonl`. Roughly 50 answerable queries drawn from dataset A.

```json
{
  "id": "gen-0012",
  "retrieval_id": "ret-0042",
  "query": "Why is the dot product scaled before the softmax in attention?",
  "reference_answer": "...",
  "required_claims": [
    "Large d_k causes dot products to grow in magnitude",
    "Large magnitudes push softmax into low-gradient regions",
    "Scaling by 1/sqrt(d_k) counteracts this"
  ],
  "must_cite": ["attention-is-all-you-need:14832-15104"]
}
```

Measured:

- **Faithfulness** — every claim in the generated answer traceable to retrieved context. Ragas, with a judge model *different* from the generator.
- **Claim coverage** — fraction of `required_claims` present.
- **Citation accuracy** — cited spans actually support the claims attached to them.
- **Refusal correctness** — on unanswerable queries, refusing rather than answering.

**Use a different model as judge than as generator.** A model scoring its own output exhibits self-preference bias and inflates every number. With Ollama generating locally, use a free-tier hosted model as judge, or at minimum a clearly different local model.

---

## Dataset C — Judge Calibration

`backend/eval/datasets/judge.jsonl`. Roughly 50 answers, hand-graded.

Daedalus grades a student's spoken or written interview answers with an LLM. That grader is itself a model whose accuracy is unmeasured unless it is checked against a human — and an ungraded grader is the weakest link in the product.

```json
{
  "id": "judge-0007",
  "question": "Explain why transformers use positional encodings.",
  "rubric_dimension": "correctness",
  "candidate_answer": "Transformers use positional encodings because self-attention is permutation-invariant...",
  "answer_tier": "confidently_wrong",
  "human_score": 1,
  "human_rationale": "Fluent and confident, but claims sinusoidal encodings are learned. They are fixed."
}
```

Scores are 0–4 against a fixed rubric.

Candidate answers are deliberately written across five tiers, because a judge that only ever sees good and terrible answers is never actually tested:

| `answer_tier` | Why it is included |
|---|---|
| `excellent` | Baseline: does the judge recognise a correct answer? |
| `partial` | The realistic middle, where grading is hardest |
| `confidently_wrong` | **The critical tier.** Fluent, assured, factually wrong. LLM judges reward confidence and fluency, and this is where they fail worst. |
| `off_topic` | Coherent but answering a different question |
| `verbose_empty` | Long, on-topic, and says nothing |

**Agreement metrics:** Spearman ρ, quadratic-weighted Cohen's κ, and mean absolute error against human scores. Report per tier — aggregate agreement hides the `confidently_wrong` failure.

Improving agreement — rubric wording, few-shot examples, structured output — and reporting before-and-after numbers is the single most differentiating result this project can produce.

---

## Construction Workflow

1. **Freeze the corpus.** Parse everything, commit `parsed/` and `manifest.json`.
2. **Sample spans** across documents, ingestion paths, and topics.
3. **Draft candidates.** A model proposes a query and a candidate span per sample.
4. **Verify every one by hand.** Fix the span, rewrite awkward phrasing, or reject. Roughly 6–10 hours for 120 queries, spread over several sessions.
5. **Write the unanswerable set manually.** These cannot be generated from spans, because they are defined by the absence of a source.
6. **Assign splits,** stratified.
7. **Validate,** then commit.

### The vocabulary-bias trap

A model writing a question from a chunk reuses that chunk's exact wording. The resulting queries are lexically near-identical to their targets, which inflates BM25 and makes hybrid retrieval look better than it is — the ablation table would then be measuring an artifact of dataset construction rather than a property of the system.

Mitigation: for the `conceptual` slice, instruct the generator to paraphrase away from source terminology, then check by eye that the question is one a student would naturally ask. Record `paraphrased: true` so the effect can be measured rather than assumed.

---

## Validation Rules

A validator runs in CI and rejects the dataset if any hold:

- `id` is duplicated or does not match `^(ret|gen|judge)-\d{4}$`
- `doc_id` is absent from `manifest.json`
- `char_start >= char_end`, or `char_end` exceeds the length of the parsed file
- `quote` does not appear at the given offsets in the parsed text (guards against drift after re-parsing)
- A span resolves to zero chunks under the default chunking configuration
- `answerable: true` with no `relevant_spans`, or `answerable: false` with any
- `grade` is not 1 or 2
- `split` is not `dev` or `test`
- Stratification drifts more than 5 percentage points from the target shares
- `verified_by` is not `human`

The quote check is the important one: it is what catches silent corruption when parsed text changes underneath the labels.

---

## Layout

```
corpus/                          # raw source material, committed
├── arXiv/
├── course notes/
├── notebooks/
├── Images/
└── manifest.json                # doc ids, types, checksums, sources

backend/eval/
├── parsed/                      # FROZEN extracted text — offsets anchor here
│   ├── attention-is-all-you-need.md
│   └── ...
├── datasets/
│   ├── retrieval.jsonl
│   ├── groundedness.jsonl
│   └── judge.jsonl
├── validate.py
├── retrieval.py                 # python -m daedalus.eval.retrieval
├── groundedness.py
└── judge.py
```

`backend/eval/` is committed in full. It must not live under `backend/data/`, which is gitignored — the datasets are source, not runtime state.

---

## Versioning

`manifest.json` carries a `dataset_version`. Bump it when the corpus changes, the parser changes, or labels are re-anchored. Results are only comparable within a version; the version appears in every report the harness prints.
