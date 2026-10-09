# Models and tracing

Which model does each job, how each provider's free quota is paced, and what leaves the machine
when model calls are traced.

**Contents:** [Model routing](#model-routing) · [Tracing model calls](#tracing-model-calls)

## Model routing

`backend/app/llm/models.py` decides which model does what. Pydantic AI's `FallbackModel` moves to the next model if one fails:

- **Question generation:** Groq → Gemini → local `qwen3.5:9b`
- **Grading:** Groq `qwen/qwen3.8-27b` → local `qwen3.5:9b` → Gemini. gpt-oss, which writes the questions, never grades the answers to them.
- **Tagging chunks and checking questions:** local `qwen3.5:4b` only, with no fallback. Both run over the whole library, so they stay off the cloud quotas, and both run with thinking off, temperature 0 and a fixed seed, which makes them repeatable.
- With `ENVIRONMENT=production` (free cloud hosting), Ollama is skipped and only cloud models are used.
- With `FAKE_MODELS=true`, every task gets a deterministic stand-in instead, embeddings and the counting of chunk sizes included, and nothing is sent to a model. The writer quotes whole sentences of the passage, so its questions pass the checks; the grader labels key points and claims by the words an answer shares with them.

In a batch and when grading, each cloud model also keeps its own pace: a sliding window of requests and tokens (and, for Qwen on Groq, of the tokens it writes), a daily allowance that comes back over 24 hours, and a wait when the provider says `retry-after`. A provider that is briefly full is waited out rather than abandoned, so a 429 doesn't spend the next provider's quota; one that says the day's allowance is spent is skipped until a request's worth has come back, and the next model takes over meanwhile.

## Tracing model calls

With a [Langfuse](https://langfuse.com) project's keys in `.env` (a project on Langfuse Cloud's
free Hobby plan will do), every model call is traced: each agent run and each request to a
model, with its provider, model, tokens in and out and how long it took, and each embedding
request with its model and token count. The calls one piece of work makes form one trace, which
carries the prompt version and the ids of what it was about:

- a question written: the writer and any repair of its quotes, the checker, and the duplicate
  check's embeddings;
- an answer graded, in practice, by `make calibrate` or by `make eval-grader LIVE=1`;
- a passage tagged, the topic map's tags embedded, a document's passages embedded, a search.

What leaves the machine:

- **By default, no text.** Prompts, passages, answers and what the models reply stay here. An
  error keeps its type but not its message or stack trace, since from a provider it can quote
  what the model wrote. `LANGFUSE_CONTENT=true` sends everything, for looking into a prompt.
- **Nothing without both keys,** and nothing at all with `FAKE_MODELS`. The API, the worker,
  `make ingest`, `make topics`, `make generate`, `make calibrate` and `make eval-grader LIVE=1`
  trace their calls; `make eval-retrieval` and replays don't.
- **Plain OpenTelemetry,** sent to Langfuse's OTLP endpoint (`backend/app/llm/tracing.py`):
  Pydantic AI's own spans, one span per piece of work and one per embedding request. Nothing
  else in the app is instrumented.
- **Deployed** (`ENVIRONMENT=production`), each piece of work's spans are sent as it ends, before
  the answer goes out: a serverless function is frozen as soon as it has answered, before the
  batch would go. Its traces carry the environment `production`, and `/health/deps` says what
  the process answering it has sent.

`make check` says whether tracing is on and what a trace holds, and `make check LIVE=1` says
whether Langfuse took the traces of its test prompts.
