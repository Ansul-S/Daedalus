# API reference

The backend is a FastAPI app. Run it with `make api`; while it runs locally, its interactive
documentation is at http://localhost:8000/docs. This page lists every endpoint and the rules they
follow.

**Contents:** [Endpoints](#endpoints) · [Rules](#rules) · [Examples](#examples)

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /health/deps` | The API is up; status of the database, the models, the API keys, tracing and the daily limits, and, while tracing is on, what the answering process has sent |
| `POST /documents/upload` | Upload a `.pdf` or `.ipynb` as the multipart field `file`. Options as query parameters: `ocr`, `formulas`, `force` |
| `POST /documents/arxiv` | Add a paper: `{"arxiv_id": "1706.03762"}`, optionally with `ocr`, `formulas` and `force` |
| `GET /documents`, `GET /documents/{id}` | Documents with their status, details, chunk count, how many of those chunks the topic map has tagged, and latest job |
| `GET /jobs`, `GET /jobs/{id}` | The newest jobs first (`kind` is `ingest`, `topics` or `generate`; `limit` at most 100), or one job: its status, progress and error |
| `GET /worker` | Whether a worker is running to take the queued jobs |
| `GET /search?q=…&limit=10&mode=hybrid` | Search all chunks. `mode` is `hybrid`, `vector` or `keyword`; `limit` is at most 50 |
| `POST /questions/generate` | Plan and queue a batch: `{"count": 20}`, optionally `document_id`. Returns **202** with the job the worker will run |
| `GET /questions` | The library, newest first. Filters: `status`, `topic_id`, `style`, `difficulty`, `document_id`, `source_updated`, `rating` (`good`, `poor` or `unrated`, by the latest rating); `limit` and `offset`, with the total of the whole match |
| `GET /questions/{id}` | One question with its sources, key points and quotes, misconceptions, validation report and token usage |
| `PATCH /questions/{id}` | Correct or retire a question: any of `text`, `reference_answer`, `key_points` (two to four, replaced as a whole) and `status` (`accepted` or `retired`), with an optional `reason`. Returns the question as `GET` does |
| `GET /topics` | The topic map with the passages and questions behind each topic |
| `POST /topics/build` | Queue a build of the topic map: the chunks without tags are tagged, then every tag is grouped into topics. Returns **202** with the job the worker will run |
| `POST /questions/{id}/attempts` | Answer an accepted question: `{"answer": "…"}`, at most 8,000 characters, optionally with `seconds` taken and an interview `time_limit`. Returns **201** with the attempt, its grade and, once graded, when the question comes back and what the answer earned |
| `POST /attempts/{id}/grades` | Grade an attempt again, e.g. after a failed grade. Earlier grades are kept |
| `GET /attempts/{id}` | An attempt with every grade it was given, oldest first |
| `GET /questions/{id}/attempts` | The answers given to a question, newest first, with their grades; `limit` (at most 100) and `offset` |
| `GET /practice/next` | The question to practise next, without its reference answer or key points, and why it was picked |
| `GET /practice/progress` | XP, level and streak, and the ten coins: when each was minted, or how far along it is |
| `GET /practice/map` | The dashboard's labyrinth: a room for each topic with questions, with its mastery and reviews due, the passages between the rooms, today's thread and the Minotaur's room |
| `GET /practice/stats` | The latest 12 scores, graded answers on each of the last 14 days, and questions due on each of the next 7 |
| `GET /practice/allowance` | What the daily limits leave: your grades today and everyone's over the last 24 hours, each with what it allows, and the limit that refuses a grade now, if any, with when it allows one again |
| `DELETE /practice` | Delete your practice: every answer with its grades, the review history and schedule, and your ratings. Answers with what was deleted. The day's grades still count against the limits |
| `POST /ratings` | Rate a question good or poor, or a grade fair or unfair: `{"question_id": 31, "value": -1, "note": "…"}` (or `grade_id`), `value` 1 or -1, with an optional note of at most 500 characters. Returns **201**. The latest rating comes with the question or grade as `rating` |

## Rules

- **Who is asking.** Locally a request is the built-in user's unless it carries a sign-in token. In production a token is needed on every per-person route (practice, attempts and grades, ratings), which answer 401 without one. The token is Better Auth's, sent as `Authorization: Bearer` and checked against the keys the frontend publishes at `/auth/jwks`. Another user's attempt or grade answers 404. The library is read by anyone, and in production nobody can change it: corrections answer 403 too.
- **Past a daily limit,** answering and grading again return **429** with the limit that refuses and `Retry-After`, and nothing is written down. A grade counts once a model has replied, even if it failed.
- **Adding material.** Both `POST` endpoints return **202** while the document's job is queued or running, and **200** when there is nothing to wait for because the document is already ingested.
  - A file over `MAX_UPLOAD_MB` gets 413; any other file type gets 415.
  - With `ENVIRONMENT=production`, both return 403, since ingestion runs locally.
- **Search results.** Each result has the chunk text, its section label, its page or cell range, and its rank in each retriever. It also has a citation, such as `RNN Intuition, pp. 7–8` or `Attention Is All You Need, § 3.2.1 Scaled Dot-Product Attention · 3.2.2 Multi-Head Attention`. For arXiv papers, a link points to the section or PDF page.
- **Without Ollama,** hybrid search falls back to keyword search and says so in `warning`, and `mode=vector` returns 503.
- **Starting a batch** needs the worker to be running, and returns the batch already in flight rather than planning a second one: two plans made at the same time would pick the same passages and pay for them twice. It returns **200** with no job when nothing is left to ask about, and 403 with `ENVIRONMENT=production`, since checking a question needs the local models.
- **Building the topic map** also needs the worker, and also returns the build already queued or running rather than a second one, which would find nothing new: a build tags whatever is untagged when it runs. The response says how many chunks are `untagged`. It returns **200** with no job when there is nothing to build from, and 403 with `ENVIRONMENT=production`. A build that fails keeps the tags it wrote, so the next one carries on.
- **The worker** holds a Postgres advisory lock for as long as it runs, and `GET /worker` reports whether that lock is held (`make ingest` and `make generate` hold it too). A job still marked `running` while no worker is running was stopped part way; the next worker to start queues it again.
- **A question is served with everything behind it:** the passages it was written from, cited as search results are, what an answer has to cover with the quote that proves each point, and the report from every check it went through, whether it passed or failed.
- **Correcting a question** holds it to the rule generation works to: each key point's quote has to be in the passage it names, one of the question's sources. Otherwise the edit is refused with 422, each problem pointing at its field, and nothing changes.
  - A new question text is embedded again for the duplicate check. Without Ollama the question is left without an embedding, and the check passes over it.
  - Every change is written into the question's validation report, with what it replaced and why.
  - Retiring a question takes it out of practice and out of the duplicate check, and keeps it and its answers; `{"status": "accepted"}` puts it back where its schedule left off.
  - A rejected question can't be edited (409).
- **A grade is served with what it refers to:** each key point's text and weight next to its label, and each claim with the passage behind its verdict, cited and linked. A question that was rejected or retired can't be answered (404). Grading runs on the cloud models first, so unlike writing questions it also works with `ENVIRONMENT=production`.
- **Practice follows a review schedule (FSRS).** An answer's first successful grade earns a rating: Again below 0.4, Hard below 0.7, Good below 0.9, Easy from 0.9, and Again whenever a claim contradicts the sources. The rating decides the practice day the question comes back, between 1 and 30 days later. `GET /practice/next` serves the question most overdue for review, then a new one from the weakest topic, then practice ahead on the question likeliest to have been forgotten. Grading an answer again doesn't reschedule it.
- **Practice earns XP, a level and coins,** all worked out from the review history, none of it stored.
  - An answer earns ten times its score and five for answering, half as much again when the question was due (late or not), five more inside an interview time limit, and, on the day's first answer, the streak's length that day, up to ten. An attempt carries what its first successful grade `earned`: the XP part by part, the level it reached and any coins.
  - Seven levels start at 50·(n−1)·n XP, from Apprentice at 0 to Daedalus at 2,100. The streak counts practice days in a row.
  - Each of the ten coins is minted by the first answer that meets its condition. Conditions about the library count only the questions it held at the time, so new questions never take a coin back.
  - The map's rooms fill a grid six wide, in topic order. A seeded maze joins them, so the same topics always give the same map. The Minotaur waits in the weakest room: the lowest mastery, then the one practised least.
- **Ratings are evaluation data:** which questions the generator got wrong, and where the grader goes wrong in real use. Every rating is kept and the latest one stands, so a change of mind leaves a trace. Any question can be rated, whatever its status: a rejected question rated good is a check that turned down too much. A failed grade has no verdict to judge and can't be rated (409).

## Examples

```sh
curl -X POST localhost:8000/documents/arxiv -H 'Content-Type: application/json' -d '{"arxiv_id": "1706.03762"}'
curl -F file=@notes.pdf 'localhost:8000/documents/upload?formulas=false'
curl 'localhost:8000/search?q=why+scale+dot-product+attention&limit=5'
curl -X POST localhost:8000/topics/build
curl 'localhost:8000/jobs?kind=topics&limit=1'
curl -X POST localhost:8000/questions/generate -H 'Content-Type: application/json' -d '{"count": 20}'
curl 'localhost:8000/questions?status=accepted&difficulty=3&limit=5'
curl -X PATCH localhost:8000/questions/31 -H 'Content-Type: application/json' -d '{"status": "retired", "reason": "asks for a reported number"}'
curl 'localhost:8000/questions/31/attempts?limit=5'
curl -X POST localhost:8000/ratings -H 'Content-Type: application/json' -d '{"question_id": 31, "value": -1, "note": "the second key point repeats the first"}'
```
