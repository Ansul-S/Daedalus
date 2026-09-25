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

Set them in the environment or in `frontend/.env.local`.

## Pages

| Page | What it shows |
|---|---|
| `/` | Redirects to `/practice` |
| `/practice` | One question at a time: the question due next and why it was picked, your answer, the grader's verdict, then the next question (see below) |
| `/questions` | The question bank: every question, accepted, retired or rejected, filtered by topic, style, difficulty, source and your rating (see below) |
| `/questions/[id]` | One question with everything behind it: its passages, key points and quotes, the checks it went through and every correction since. It can be corrected, retired or put back, and rated |
| `/dashboard` | What practice has built: the labyrinth of topics, the level and its wing, the days practised, the latest scores and the coins (see below) |
| `/setup` | The status of the database, the local models and the API keys, checked each time the page loads |
| `/pattern-book` | The design system, live: pigments, type, controls, marks (with the coins and the wing), hatching, glyph pictures and Markdown with LaTeX |

Library is in the navigation, without a link, until it is built. The header shows the streak and the XP.

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

## Structure

```
src/app/             pages, the root layout (fonts, theme, header, footer), error and 404 pages
src/app/globals.css  tokens, themes, type roles, hatching and Markdown styles
src/components/      the design system's pieces: labyrinth mark, Ariadne's thread, hatching,
                     dimension-line timer, drafting sheet and title block, glyph mosaic, Markdown,
                     coins and the wing, the rating control, form fields
src/components/ui/   shadcn/ui components (Radix, "lyra" style), restyled to the tokens
src/client/          typed API client (generated)
src/lib/             API address and error type, theme, reading a grade, drafts, the stopwatch,
                     interview mode, what practice earned in words, the level-up burst, naming a
                     question and reading its validation report, the glyph pictures' engine and
                     grids
openapi.json         the API schema the client is generated from (generated)
```

## Design system

- **Four pigments:** bone (the paper), ink, ochre (only for what is earned) and sinopia, the accent (Ariadne's thread, focus). Components use role tokens such as `ground`, `surface`, `fg`, `line` and `thread`, which the dark theme redefines. Tailwind's default colours are switched off, so only these exist.
- **Three typefaces,** self-hosted through `next/font`: Big Shoulders for titles and numbers, Source Serif 4 for reading text and questions, JetBrains Mono for labels, measurements and the glyph pictures.
- **Themes.** Light and dark follow the system; the switch in the header picks Light, Dark or Auto, and the browser remembers it.
- **Status never depends on colour alone.** Hatching marks key points (covered, partial, missing), ratings (Again, Hard, Good, Easy) and a topic's mastery on the dashboard.
- **Motion.** A glyph picture settles out of noise when it comes into view, a room with reviews due sends out a ring, and a new level throws up Greek letters (`canvas-confetti`, loaded only then). With reduced motion the picture is drawn settled and nothing else moves.

`/pattern-book` is the reference: every piece, live, in both themes. A component added with the shadcn/ui CLI (`components.json`) arrives in the CLI's own style: restyle it to the tokens, and check `package.json` for packages the CLI added.

## Generated code

Both are committed, and rebuilt by a command rather than edited by hand.

- **The API client,** `src/client/`. `make client` writes the backend's OpenAPI schema to `openapi.json` and generates the client from it with `@hey-api/openapi-ts` (`openapi-ts.config.ts`): types, one function per API operation named after its backend handler (`practiceNext()`), and TanStack Query options. Run it after changing the API. A failed call throws an `ApiError` (`src/lib/api-errors.ts`) carrying the HTTP status, or `null` when the API couldn't be reached.
- **The glyph pictures,** `src/lib/glyph/grids.ts`. `make glyphs IMAGE=path/to/etching.jpeg` redraws Charles Holroyd's etching *Daedalus* (1895, British Museum 1918,0608.347, public domain) in Greek letters, in 13 steps from paper to ink, and writes the grids run-length encoded. The crop boxes were measured on a 736 × 942 copy of the etching, and only the grids enter the repository, never the picture. The script, `backend/scripts/glyphs.py`, runs in the backend's `glyphs` dependency group (NumPy, OpenCV, Pillow).
