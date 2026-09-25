"use client";

import { keepPreviousData, useQueries, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";

import {
  listDocumentsOptions,
  listQuestionsOptions,
  listTopicsOptions,
} from "@/client/@tanstack/react-query.gen";
import type { ListQuestionsData, QuestionOut } from "@/client/types.gen";
import { ApiProblem } from "@/components/api-problem";
import { Commands, FILL_THE_LABYRINTH } from "@/components/commands";
import { Field, Select } from "@/components/field";
import { Pips } from "@/components/hatching";
import { Markdown } from "@/components/markdown";
import { ratingWord } from "@/components/rating";
import { SheetSection, TitleBlock } from "@/components/sheet";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  questionNumber,
  questionPath,
  STATUS_LABEL,
  STATUSES,
  type Status,
  STYLE_NAMES,
  statusLabel,
  styleLabel,
} from "@/lib/questions";
import { cn } from "@/lib/utils";

// The question bank: every question written from the library, the accepted ones practice draws
// from, the retired ones taken out by hand and the ones the checks turned down. The filters
// live in the address, so a filtered list can be linked to and the back button undoes a change.

const PAGE_SIZE = 25;
const MONO = "font-mono text-[11px] leading-[1.4] tracking-[0.04em]";

type Query = NonNullable<ListQuestionsData["query"]>;
type Tab = Status | "all";
type RatingFilter = NonNullable<Query["rating"]>;

const TABS: Tab[] = [...STATUSES, "all"];
const RATINGS: [RatingFilter, string][] = [
  ["good", "Good question"],
  ["poor", "Poor question"],
  ["unrated", "Not rated"],
];

type Filters = {
  status: Tab;
  topic: number | null;
  style: string | null;
  difficulty: number | null;
  source: number | null;
  rating: RatingFilter | null;
  page: number;
};

function whole(value: string | null, most = Number.MAX_SAFE_INTEGER): number | null {
  if (value === null || !/^\d+$/.test(value)) return null;
  const found = Number(value);
  return found >= 1 && found <= most ? found : null;
}

/** /questions?status=retired&topic=4&page=2; anything unknown is left out. */
function readFilters(params: URLSearchParams): Filters {
  const status = params.get("status");
  const style = params.get("style");
  const rating = params.get("rating");
  return {
    status: status === "all" || STATUSES.includes(status as Status) ? (status as Tab) : "accepted",
    topic: whole(params.get("topic")),
    style: style !== null && STYLE_NAMES.includes(style) ? style : null,
    difficulty: whole(params.get("difficulty"), 5),
    source: whole(params.get("source")),
    rating: RATINGS.some(([value]) => value === rating) ? (rating as RatingFilter) : null,
    page: whole(params.get("page")) ?? 1,
  };
}

function filtering(filters: Filters): boolean {
  const { topic, style, difficulty, source, rating } = filters;
  return [topic, style, difficulty, source, rating].some((value) => value !== null);
}

/** The API's query for these filters, under one status tab. */
function query(filters: Filters, tab: Tab, page: Pick<Query, "limit" | "offset">): Query {
  return {
    status: tab === "all" ? undefined : tab,
    topic_id: filters.topic ?? undefined,
    style: (filters.style ?? undefined) as Query["style"],
    difficulty: filters.difficulty ?? undefined,
    document_id: filters.source ?? undefined,
    rating: filters.rating ?? undefined,
    ...page,
  };
}

/** This page's address with some of its parameters changed; null removes one. */
function address(params: URLSearchParams, changes: Record<string, string | null>): string {
  const next = new URLSearchParams(params);
  for (const [name, value] of Object.entries(changes)) {
    if (value === null) next.delete(name);
    else next.set(name, value);
  }
  const search = next.toString();
  return search ? `/questions?${search}` : "/questions";
}

function StatusTabs({
  filters,
  params,
  counts,
}: {
  filters: Filters;
  params: URLSearchParams;
  counts: (number | undefined)[];
}) {
  return (
    <nav aria-label="Status">
      <ul className="grid w-full grid-cols-2 border-t border-l border-line-2 min-[560px]:w-fit min-[560px]:grid-cols-4">
        {TABS.map((tab, i) => {
          const current = filters.status === tab;
          return (
            <li key={tab} className="border-r border-b border-line-2">
              <Link
                href={address(params, { status: tab === "accepted" ? null : tab, page: null })}
                scroll={false}
                aria-current={current ? "true" : undefined}
                className={cn(
                  "type-label flex items-baseline justify-between gap-4 px-3.5 py-2.5",
                  current ? "bg-fg text-ground" : "hover:bg-surface",
                )}
              >
                {tab === "all" ? "All" : STATUS_LABEL[tab]}
                <span
                  className={cn(
                    "font-display text-[1.05rem] leading-none font-bold tracking-[0.02em]",
                    !current && "text-fg-2",
                  )}
                >
                  {counts[i] ?? "–"}
                </span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

function FilterBar({
  filters,
  params,
  onChange,
}: {
  filters: Filters;
  params: URLSearchParams;
  onChange: (name: string, value: string) => void;
}) {
  const topics = useQuery(listTopicsOptions({ query: { limit: 500 } }));
  const documents = useQuery(listDocumentsOptions());
  const topicChoices = (topics.data ?? [])
    .filter((topic) => topic.question_count > 0)
    .sort((a, b) => a.name.localeCompare(b.name));
  const sourceChoices = [...(documents.data ?? [])].sort((a, b) => a.title.localeCompare(b.title));
  // A filter from the address still shows while the choices load, or when it names nothing
  const lostTopic = filters.topic !== null && !topicChoices.some(({ id }) => id === filters.topic);
  const lostSource =
    filters.source !== null && !sourceChoices.some(({ id }) => id === filters.source);

  return (
    <form
      aria-label="Filters"
      onSubmit={(event) => event.preventDefault()}
      className="grid grid-cols-1 items-end gap-x-4 gap-y-3 min-[360px]:grid-cols-2 md:grid-cols-3 xl:grid-cols-[repeat(5,minmax(0,1fr))_auto]"
    >
      <Field label="Topic">
        {(control) => (
          <Select
            {...control}
            value={filters.topic ?? ""}
            onChange={(event) => onChange("topic", event.target.value)}
          >
            <option value="">All topics</option>
            {lostTopic && <option value={filters.topic ?? ""}>topic {filters.topic}</option>}
            {topicChoices.map((topic) => (
              <option key={topic.id} value={topic.id}>
                {topic.name} ({topic.question_count})
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Style">
        {(control) => (
          <Select
            {...control}
            value={filters.style ?? ""}
            onChange={(event) => onChange("style", event.target.value)}
          >
            <option value="">All styles</option>
            {STYLE_NAMES.map((style) => (
              <option key={style} value={style}>
                {styleLabel(style)}
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Difficulty">
        {(control) => (
          <Select
            {...control}
            value={filters.difficulty ?? ""}
            onChange={(event) => onChange("difficulty", event.target.value)}
          >
            <option value="">Any difficulty</option>
            {[1, 2, 3, 4, 5].map((level) => (
              <option key={level} value={level}>
                {level} of 5
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Source">
        {(control) => (
          <Select
            {...control}
            value={filters.source ?? ""}
            onChange={(event) => onChange("source", event.target.value)}
          >
            <option value="">All sources</option>
            {lostSource && <option value={filters.source ?? ""}>document {filters.source}</option>}
            {sourceChoices.map((document) => (
              <option key={document.id} value={document.id}>
                {document.title}
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Your rating">
        {(control) => (
          <Select
            {...control}
            value={filters.rating ?? ""}
            onChange={(event) => onChange("rating", event.target.value)}
          >
            <option value="">Any rating</option>
            {RATINGS.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </Select>
        )}
      </Field>
      {filtering(filters) && (
        <Link
          href={address(params, {
            topic: null,
            style: null,
            difficulty: null,
            source: null,
            rating: null,
            page: null,
          })}
          scroll={false}
          className="type-label thread-link justify-self-start pb-2.5"
        >
          Clear filters
        </Link>
      )}
    </form>
  );
}

/** One question in the list; `showStatus` when the list mixes accepted, retired and rejected. */
function QuestionRow({ question, showStatus }: { question: QuestionOut; showStatus: boolean }) {
  const marks = [
    showStatus &&
      question.status !== "accepted" && (
      <Badge key="status" variant="dashed">
        {statusLabel(question.status)}
      </Badge>
    ),
    question.rating && (
      <Badge key="rating" variant={question.rating.value === 1 ? "default" : "thread"}>
        {ratingWord("question", question.rating.value)}
      </Badge>
    ),
    question.source_updated && (
      <Badge key="updated" variant="thread">
        Source updated
      </Badge>
    ),
  ].filter(Boolean);

  return (
    <li className="relative grid grid-cols-1 gap-x-6 gap-y-1.5 border-b border-dotted border-line-2 py-3.5 hover:bg-surface/50 sm:grid-cols-[minmax(0,1fr)_auto]">
      <p className={cn(MONO, "text-fg-2 uppercase")}>
        {questionNumber(question.id)} · {question.topic ?? "no topic"} ·{" "}
        {styleLabel(question.style)} · difficulty{" "}
        <Pips value={question.difficulty} label="difficulty" />
      </p>
      {marks.length > 0 && (
        <p className="flex flex-wrap gap-1.5 sm:col-start-2 sm:row-start-1 sm:justify-end">
          {marks}
        </p>
      )}
      {/* The whole row opens the question */}
      <Link
        href={questionPath(question.id)}
        className="max-w-[48rem] text-[1.0625rem] leading-[1.45] decoration-thread decoration-[1.5px] underline-offset-[3px] after:absolute after:inset-0 hover:underline sm:col-span-2"
      >
        <Markdown inline links={false}>
          {question.text}
        </Markdown>
      </Link>
      <p className={cn(MONO, "text-fg-2 sm:col-span-2")}>{question.citations.join(" · ")}</p>
    </li>
  );
}

function Pager({
  page,
  pages,
  params,
}: {
  page: number;
  pages: number;
  params: URLSearchParams;
}) {
  const go = (to: number, label: string, disabled: boolean) =>
    disabled ? (
      <Button variant="outline" size="sm" disabled>
        {label}
      </Button>
    ) : (
      <Button asChild variant="outline" size="sm">
        <Link href={address(params, { page: to === 1 ? null : String(to) })}>{label}</Link>
      </Button>
    );
  return (
    <nav aria-label="Pages" className="mt-6 flex flex-wrap items-center gap-3">
      {go(page - 1, "Previous", page <= 1)}
      <span className="type-label text-fg-2">
        Page {page} of {pages}
      </span>
      {go(page + 1, "Next", page >= pages)}
    </nav>
  );
}

export function QuestionBank({ children }: { children: React.ReactNode }) {
  const params = useSearchParams();
  const router = useRouter();
  const filters = readFilters(params);
  const offset = (filters.page - 1) * PAGE_SIZE;

  const list = useQuery({
    ...listQuestionsOptions({ query: query(filters, filters.status, { limit: PAGE_SIZE, offset }) }),
    placeholderData: keepPreviousData,
  });
  // How many there are under each tab, with the other filters as they are
  const counts = useQueries({
    queries: TABS.map((tab) => ({
      ...listQuestionsOptions({ query: query(filters, tab, { limit: 1 }) }),
      placeholderData: keepPreviousData,
    })),
  });
  const totals = counts.map((count) => count.data?.total);
  const found = list.data;

  function change(name: string, value: string) {
    router.replace(address(params, { [name]: value || null, page: null }), { scroll: false });
  }

  const pages = found ? Math.max(1, Math.ceil(found.total / PAGE_SIZE)) : 1;
  const shown = found?.results.length ?? 0;
  const libraryEmpty = !filtering(filters) && totals[TABS.indexOf("all")] === 0;

  return (
    <>
      <SheetSection aria-busy={list.isFetching}>
        {children}
        <div className="grid gap-5">
          <StatusTabs filters={filters} params={params} counts={totals} />
          <FilterBar filters={filters} params={params} onChange={change} />
        </div>
        <div className="mt-8">
          {list.isError && !found ? (
            <ApiProblem error={list.error} retry={() => void list.refetch()} />
          ) : !found ? (
            <p className="text-fg-2">Opening the question bank…</p>
          ) : libraryEmpty ? (
            <>
              <p className="mb-5 max-w-[62ch]">
                No questions yet. Add study material, build the topic map and write questions
                from it:
              </p>
              <Commands title="Filling the labyrinth" lines={FILL_THE_LABYRINTH} />
            </>
          ) : (
            <>
              <p role="status" className="type-label mb-2 text-fg-2">
                {shown === 0
                  ? found.total === 0
                    ? "No questions match these filters"
                    : "Nothing on this page"
                  : `${offset + 1}–${offset + shown} of ${found.total}`}
              </p>
              {shown > 0 && (
                <ul className="border-t border-line-2">
                  {found.results.map((question) => (
                    <QuestionRow
                      key={question.id}
                      question={question}
                      showStatus={filters.status === "all"}
                    />
                  ))}
                </ul>
              )}
              {shown === 0 && found.total > 0 && (
                <Link href={address(params, { page: null })} className="thread-link">
                  Back to the first page
                </Link>
              )}
              {pages > 1 && <Pager page={Math.min(filters.page, pages)} pages={pages} params={params} />}
            </>
          )}
        </div>
      </SheetSection>
      <TitleBlock
        cells={[
          { label: "Sheet", value: "Q-01" },
          { label: "Drawing", value: "Questions" },
          ...STATUSES.map((status) => ({
            label: STATUS_LABEL[status],
            value: totals[TABS.indexOf(status)] ?? "–",
          })),
          {
            label: "Showing",
            value: found && shown > 0 ? `${offset + 1}–${offset + shown}` : "–",
          },
        ]}
      />
    </>
  );
}
