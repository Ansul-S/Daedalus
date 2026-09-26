# Daedalus frontend

The practice app in the browser: Next.js 16 (App Router), React 19, TypeScript and Tailwind CSS 4, with TanStack Query and a typed client for the Daedalus API. Setting up the whole project is described in the [main README](../README.md).

## Running it

The API has to be running (`make api` from the repository root). From this folder:

```sh
pnpm install --frozen-lockfile
pnpm build && pnpm start     # the production build, on http://localhost:3000
pnpm lint                    # ESLint
```

For development, `make web` from the repository root runs the development server, which reloads as files change.

- **Fonts** are downloaded when the app is built and served with it, so the build needs a network connection and the running app sends no requests to Google.
- **The browser calls the API directly,** not through a Next.js proxy, whose timeout would cut off a grade that waits on a model. The API's `CORS_ORIGINS` has to include this app's address.

| Setting | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | The API's address as the browser sees it. It is written into the build, so a change needs `pnpm build` again |
| `API_URL` | `NEXT_PUBLIC_API_URL` | The API's address as this app's server sees it, for when the two differ (e.g. in a container). The Setup page checks the API from the server |
| `SITE_URL` | `http://localhost:3000` | This app's own address, from which a shared link's preview picture is fetched. It is written into the build |

Set them in the environment or in `frontend/.env.local`.

## Pages

| Page | What it shows |
|---|---|
| `/` | The landing page: what Daedalus does and how, the dashboard's Minotaur, the grader's measured agreement with hand grades, and questions asked before starting (see below) |
| `/practice` | One question at a time: the question due next and why it was picked, your answer, the grader's verdict, then the next question (see below) |
| `/questions` | The question bank: every question, accepted, retired or rejected, filtered by topic, style, difficulty, source and your rating (see below) |
| `/questions/[id]` | One question with everything behind it: its passages, key points and quotes, the checks it went through and every correction since. It can be corrected, retired or put back, and rated |
| `/library` | The material questions are written from: add PDFs, notebooks and arXiv papers, build the topic map and write questions, following each job as the worker runs it (see below) |
| `/dashboard` | What practice has built: the labyrinth of topics, the level and its wing, the days practised, the latest scores and the coins (see below) |
| `/setup` | The status of the database, the local models and the API keys, checked each time the page loads |
| `/pattern-book` | The design system, live: pigments, type, controls, marks (with the coins, the wing and the states of a job), hatching, glyph pictures and Markdown with LaTeX |

The header shows the streak and the XP, and its name leads back to the landing page.

### The landing page

- **The front.** The headline and what Daedalus does, Enter the labyrinth (to `/practice`; `Enter` does the same while nothing on the page has focus) and How it works, the quick start (the Setup page's commands), and Charles Holroyd's *Daedalus* in Greek letters, settling out of noise as the page opens. On a narrow screen the picture is drawn from its coarse grid, 104 letters across, so the letters still read as letters.
- **How it works.** Four steps, from reading the sources to grading an answer, each with a small glyph picture: a page, the meander and the labyrinth mark, drawn in code and sampled cell by cell (`src/lib/glyph/drawings.ts`), and Daedalus at his bench from the etching. Being lines rather than pictures of paper, the drawings don't flip in the dark theme.
- **The Minotaur** stands for the dashboard: Antonio Tempesta's *Theseus and the Minotaur* in Greek letters, what the rooms and the lair are, and the coin for facing it. Face the Minotaur opens the dashboard. The picture is mounted on the panel as a print, ink on bone in both themes: flipped onto the dark panel, the etching's light line-work would sink into its hatched arena.
- **The grader, measured.** Spearman ρ, Cohen's κ and the number of hand-graded answers are read from the milestone table in `docs/design.md` when the page is built (`src/lib/measured.ts`), so the page quotes the design notes rather than restating them. If that sentence changes shape the build fails, naming the file to change. How it was measured links to the notes' section on GitHub.
- **Nothing on the page needs the API** (the header's streak and XP apart), so it stands on a fresh clone and with the API stopped.
- **The preview picture** of a shared link, `src/app/(landing)/opengraph-image.jpg` (1200 × 630, its alt text beside it), is a still of the front: the section as the built page draws it at 1360 px with reduced motion and without the paper grain, fitted onto the field. Take it again when the front changes.

### Practising

- **Answering.** Write the answer in Markdown, with `$maths$` for LaTeX, and check it under Preview. `⌘↵` (`Ctrl+↵` elsewhere) submits it; the API takes up to 8,000 characters.
- **Time.** A stopwatch counts while the question is on screen and the page is in view, and the time goes with the answer.
- **Interview mode.** The Interview switch beside the answer gives three minutes an answer, as in an interview: the stopwatch becomes a dimension line counting down, then into overtime, and it keeps counting in another tab. The limit goes with the answer, and an answer inside it earns 5 XP more. The browser remembers the switch.
- **Drafts.** An answer being written is kept in the browser's local storage, with its time, so a reload or a closed tab loses nothing. It is cleared once the answer is graded, and never leaves the browser before it is submitted.
- **The verdict.** The score and the rating it earns, and when the question comes back; how the score was reached; each key point covered, partly covered or missing, with the words of your answer that earned it, underlined in the answer too; each claim supported, contradicted or unverified, with the passage it was checked against; clarity, strengths, gaps, errors, a follow-up question and a model answer.
- **What it earned.** Under the score, the XP the answer earned and what made it up, a new level when one is reached (with a burst of Greek letters, unless the system asks for reduced motion), and the coins it minted, each also announced in a corner. The title block below the sheet then shows the new streak, level and XP, and the counts of questions due and new.
- **Your ratings.** At the foot of the verdict, rate the question (Good question, Poor question) and the grade (Fair grade, Unfair grade). A poor question or an unfair grade then asks for a short note on why. Ratings are kept as evaluation data: the questions worth fixing, and where the grader goes wrong.
- **The question's page.** The question number (Q.050) opens it in the question bank. A half-written answer and its time are kept for when you come back.
- **When grading fails,** the answer is already saved: the verdict says why, and Grade again tries the models again.
- **Next.** The Next question button, or `N` when the cursor isn't in a text field.

### The dashboard

- **The labyrinth.** A room for each topic with questions, hatched from empty to solid as its mastery grows, with a knot for the reviews due in it. The Minotaur waits in the weakest room, and today's thread runs from the entrance through the rooms answered in. The map comes from the API (`GET /practice/map`), so the same topics always give the same maze. Under 720 px it keeps its size and scrolls sideways in its card.
- **A room** opens a list of its questions, each with its latest score and when it comes back, and each opening its page in the question bank. The map is one stop in the tab order: the arrow keys move between rooms, `Enter` opens one and `Escape` closes it.
- **The wing** grows a feather for every fourteenth of the way to the next level; the one being grown is hatched.
- **The thread** shows the last 14 practice days, and **Scores** the last 12 answers. Point at a day or an answer, or focus the chart and use the arrow keys, to read it.
- **The treasury** holds the ten coins: struck in ochre when minted, with the day, and otherwise what is still needed.

### The question bank

- **The list.** Tabs split it into accepted questions (the library practice draws from), retired ones and the ones the checks rejected, each with its count; the filters narrow it by topic, style, difficulty, source and your latest rating. The filters live in the address, so a filtered list can be linked to and the back button undoes a change. 25 questions a page.
- **A question's page** shows its passages (linked to the source and readable in place), what an answer has to cover with the quote that proves each key point, the misconceptions it expects, the four checks it went through when it was written (quotes, answerable, not trivia, not a duplicate), every correction since with what it replaced and why, and the model that wrote it.
- **Correcting** a question replaces its text, reference answer or key points (two to four, as a whole). Each quote has to be in the passage it names, word for word: the API checks every one before anything changes, and a quote it can't find is marked beside its key point. `⌘↵` saves.
- **Retiring** takes a question out of practice and out of the duplicate check, with an optional reason; Put back returns it where its schedule left off. A rejected question stays as the record of why the checks turned it down: it can be rated, but not corrected.
- **Rating** a question works as on the verdict. Every rating is kept and the latest one stands.

### The library

The page runs from material to practice in four steps. Adding material, building the topic map and writing questions each queue a job for the worker (`make worker`), and the page follows each job until it ends.

- **Adding material.** Drop PDFs and notebooks on the page, or choose them; any other file is left out, by name. They are uploaded one after another, and each then shows how its reading goes: queued, running with the worker's progress, then ready with its passages, or failed with the reason. An arXiv paper is added by its ID (`1706.03762`, or `1706.03762v7` for one version) or its arxiv.org address. Formulas as LaTeX (on unless unticked, and slower) and OCR for scanned pages apply to PDFs, and to an arXiv paper without an HTML version. A file or paper already in the library is not read again unless Read it again is ticked, or a different arXiv version is asked for.
- **Building the topic map.** The local model tags the passages that have no tags yet, about 12 s each, and every tag in the library is then grouped into topics. With every passage tagged, Build it again only groups the tags again. A build that fails keeps the tags it wrote, so the next one carries on from there.
- **Writing questions.** How many (1 to 100), from every source or one of them. The API plans the batch when it is asked, from the passages the topic map has tagged by then, so the page says when passages are still outside the map and waits while a build is under way. One batch runs at a time. A batch that uses up the day's allowance stops with the rest of it left in the queue; one in which every question failed says so, and the worker's output gives each reason.
- **The worker.** The page asks the API every five seconds whether a worker is running (`GET /worker`). Without one, a line says that what is asked for here waits; once something waits, a notice counts the jobs and shows how to start a worker. A job still marked running while no worker is running was stopped part way, and is marked so until the next worker to start queues it again. A reading stopped part way starts again from the beginning; a build keeps the tags it wrote, and a batch the questions.
- **Following a job.** While a job waits or runs, the page asks after it every two seconds, and each step in its progress fetches again what it changes: the documents and their counts while the map is built, then the topics; the questions while a batch is written, then what practice and the dashboard show. While nothing waits or runs, the page only asks after the worker.
- **The shelves** hold every document, newest first: the kind of source, its file or arXiv ID, how its reading went, its length, its passages and how many are in the topic map, and its accepted questions, which open the question bank filtered to that source. A reading that failed says why; a document read before keeps its passages when a later reading fails.

## The end-to-end test

`make e2e`, from the repository root, walks through the app in a browser on the stand-in models of `FAKE_MODELS` (`backend/app/llm/fakes.py`). Enter the labyrinth on the landing page leads to an empty practice page and on to the library, where the test uploads a notebook written for it (`e2e/fixtures/training-notes.ipynb`, three short sections on training deep networks). The worker reads it into three passages, the topic map finds three topics, and a batch of three questions is written, all accepted. The test answers the question practice brings up and checks the verdict: graded by `fake-grader`, 0.75 and Good, two claims supported and cited by notebook and cell, and the XP earned. On the dashboard the labyrinth has three rooms, the one answered in practised with a mastery of 0.75, and the wing shows the same XP.

- **Before the first run,** download the browser it drives, Playwright's own Chromium (about 560 MB once unpacked): `pnpm exec playwright install chromium` in this folder. Postgres has to be running (`make db-up`), and ports 8000 and 3000 free: stop `make api`, and `make web` or `pnpm start`. Ollama and the API keys aren't needed.
- **A database of its own.** `backend/scripts/e2e.py` makes `daedalus_e2e` afresh on the Postgres server of `make db-up`, and a temporary folder for uploads. Both are removed when the run ends, whether it passed or not, and the development database is never touched. `playwright.config.ts` starts the API, a worker and the frontend on them, and stops them at the end; passages may be small there, so that each of the notebook's sections is one. Playwright started any other way refuses to run, since the database isn't there.
- **Nothing real is within reach.** The servers get dummy API keys and an Ollama address nothing answers at, so a model call that missed the stand-ins fails at once instead of spending quota.
- **The stand-ins** answer at once and the same way every time. A question is made from its passage's own sentences, which are also the quotes behind its key points, so it passes every check. The grader counts a key point covered by how many of its words the answer uses, and supports a claim that shares enough words with a passage. Embeddings count a text's words, hashed into 1,024 dimensions. The test proves the pieces work together, not what the models would write: `make calibrate` measures the real grader.
- **The frontend is built** as `pnpm build` builds it, with the API at `http://localhost:8000`, so `pnpm start` serves the same app afterwards. A run takes about half a minute, the build included; the walk itself takes about 10 s.
- **Watching it.** `make e2e ARGS=--headed` runs it in a browser window, and `make e2e ARGS="--trace on"` records every step. A failed run keeps its trace and a screenshot in `test-results/`, and `pnpm exec playwright show-trace test-results/*/trace.zip` replays it, with the page at each step.

## Structure

```
src/app/             pages, the root layout (fonts, theme, header, footer), error and 404 pages
src/app/(landing)/   the landing page at /, its parts and its preview picture
src/app/globals.css  tokens, themes, type roles, hatching and Markdown styles
src/components/      the design system's pieces: labyrinth mark, Ariadne's thread, hatching,
                     dimension-line timer, drafting sheet and title block, glyph mosaic, Markdown,
                     coins and the wing, the rating control, form fields, job status marks, the
                     file drop
src/components/ui/   shadcn/ui components (Radix, "lyra" style), restyled to the tokens
src/client/          typed API client (generated)
src/lib/             API address and error type, theme, reading a grade, drafts, the stopwatch,
                     interview mode, what practice earned in words, the level-up burst, naming a
                     question and reading its validation report, the library's work in words,
                     the grader's measurement from the design notes, the glyph pictures' engine,
                     grids and drawings
openapi.json         the API schema the client is generated from (generated)
e2e/                 the end-to-end test and the notebook written for it
playwright.config.ts the servers the end-to-end test runs on
```

## Design system

- **Four pigments:** bone (the paper), ink, ochre (only for what is earned) and sinopia, the accent (Ariadne's thread, focus). Components use role tokens such as `ground`, `surface`, `fg`, `line` and `thread`, which the dark theme redefines. Tailwind's default colours are switched off, so only these exist.
- **Three typefaces,** self-hosted through `next/font`: Big Shoulders for titles and numbers, Source Serif 4 for reading text and questions, JetBrains Mono for labels, measurements and the glyph pictures.
- **Themes.** Light and dark follow the system; the switch in the header picks Light, Dark or Auto, and the browser remembers it.
- **Status never depends on colour alone.** Hatching marks key points (covered, partial, missing), ratings (Again, Hard, Good, Easy), a topic's mastery on the dashboard and where a job in the library stands (waiting, under way, stopped part way, done, failed).
- **Motion.** A glyph picture settles out of noise when it comes into view (on the landing page, as the page opens), a room with reviews due and a job under way send out a ring, and a new level throws up Greek letters (`canvas-confetti`, loaded only then). With reduced motion the picture is drawn settled and nothing else moves.

`/pattern-book` is the reference: every piece, live, in both themes. A component added with the shadcn/ui CLI (`components.json`) arrives in the CLI's own style: restyle it to the tokens, and check `package.json` for packages the CLI added.

## Generated code

Both are committed, and rebuilt by a command rather than edited by hand.

- **The API client,** `src/client/`. `make client` writes the backend's OpenAPI schema to `openapi.json` and generates the client from it with `@hey-api/openapi-ts` (`openapi-ts.config.ts`): types, one function per API operation named after its backend handler (`practiceNext()`), and TanStack Query options. Run it after changing the API. A failed call throws an `ApiError` (`src/lib/api-errors.ts`) carrying the HTTP status, or `null` when the API couldn't be reached.
- **The glyph pictures,** `src/lib/glyph/grids.ts`. `make glyphs DAEDALUS=… MINOTAUR=…` redraws two etchings in the public domain in Greek letters, in 13 steps from paper to ink, and writes the grids run-length encoded: Charles Holroyd's *Daedalus* (1895, British Museum 1918,0608.347), whose crop boxes were measured on a 736 × 942 copy, and Antonio Tempesta's *Theseus and the Minotaur* (after 1606, the Metropolitan Museum of Art, 35.6(75)), whose crop box was measured on the museum's 4000 × 3570 open-access scan (https://images.metmuseum.org/CRDImages/dp/original/DP-15360-001.jpg). Only the grids enter the repository, never the pictures. The script, `backend/scripts/glyphs.py`, runs in the backend's `glyphs` dependency group (NumPy, OpenCV, Pillow).
