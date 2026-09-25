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
| `/setup` | The status of the database, the local models and the API keys, checked each time the page loads |
| `/pattern-book` | The design system, live: pigments, type, controls, marks, hatching, glyph pictures and Markdown with LaTeX |

Questions, Library and Dashboard are in the navigation, without links, until they are built.

### Practising

- **Answering.** Write the answer in Markdown, with `$maths$` for LaTeX, and check it under Preview. `⌘↵` (`Ctrl+↵` elsewhere) submits it; the API takes up to 8,000 characters.
- **Time.** A stopwatch counts while the question is on screen and the page is in view, and the time goes with the answer.
- **Drafts.** An answer being written is kept in the browser's local storage, with its time, so a reload or a closed tab loses nothing. It is cleared once the answer is graded, and never leaves the browser before it is submitted.
- **The verdict.** The score and the rating it earns, and when the question comes back; how the score was reached; each key point covered, partly covered or missing, with the words of your answer that earned it, underlined in the answer too; each claim supported, contradicted or unverified, with the passage it was checked against; clarity, strengths, gaps, errors, a follow-up question and a model answer.
- **When grading fails,** the answer is already saved: the verdict says why, and Grade again tries the models again.
- **Next.** The Next question button, or `N` when the cursor isn't in a text field.

## Structure

```
src/app/             pages, the root layout (fonts, theme, header, footer), error and 404 pages
src/app/globals.css  tokens, themes, type roles, hatching and Markdown styles
src/components/      the design system's pieces: labyrinth mark, Ariadne's thread, hatching,
                     dimension-line timer, drafting sheet and title block, glyph mosaic, Markdown
src/components/ui/   shadcn/ui components (Radix, "lyra" style), restyled to the tokens
src/client/          typed API client (generated)
src/lib/             API address and error type, theme, reading a grade, drafts, the stopwatch,
                     the glyph pictures' engine and grids
openapi.json         the API schema the client is generated from (generated)
```

## Design system

- **Four pigments:** bone (the paper), ink, ochre (only for what is earned) and sinopia, the accent (Ariadne's thread, focus). Components use role tokens such as `ground`, `surface`, `fg`, `line` and `thread`, which the dark theme redefines. Tailwind's default colours are switched off, so only these exist.
- **Three typefaces,** self-hosted through `next/font`: Big Shoulders for titles and numbers, Source Serif 4 for reading text and questions, JetBrains Mono for labels, measurements and the glyph pictures.
- **Themes.** Light and dark follow the system; the switch in the header picks Light, Dark or Auto, and the browser remembers it.
- **Status never depends on colour alone.** Hatching marks key points (covered, partial, missing) and ratings (Again, Hard, Good, Easy).
- **Motion.** A glyph picture settles out of noise when it comes into view. With reduced motion it is drawn settled.

`/pattern-book` is the reference: every piece, live, in both themes. A component added with the shadcn/ui CLI (`components.json`) arrives in the CLI's own style: restyle it to the tokens, and check `package.json` for packages the CLI added.

## Generated code

Both are committed, and rebuilt by a command rather than edited by hand.

- **The API client,** `src/client/`. `make client` writes the backend's OpenAPI schema to `openapi.json` and generates the client from it with `@hey-api/openapi-ts` (`openapi-ts.config.ts`): types, one function per API operation named after its backend handler (`practiceNext()`), and TanStack Query options. Run it after changing the API. A failed call throws an `ApiError` (`src/lib/api-errors.ts`) carrying the HTTP status, or `null` when the API couldn't be reached.
- **The glyph pictures,** `src/lib/glyph/grids.ts`. `make glyphs IMAGE=path/to/etching.jpeg` redraws Charles Holroyd's etching *Daedalus* (1895, British Museum 1918,0608.347, public domain) in Greek letters, in 13 steps from paper to ink, and writes the grids run-length encoded. The crop boxes were measured on a 736 × 942 copy of the etching, and only the grids enter the repository, never the picture. The script, `backend/scripts/glyphs.py`, runs in the backend's `glyphs` dependency group (NumPy, OpenCV, Pillow).
