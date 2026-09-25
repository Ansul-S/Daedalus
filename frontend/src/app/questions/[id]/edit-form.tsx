"use client";

import { useEffect, useRef, useState } from "react";

import type {
  KeyPointIn,
  KeyPointOut,
  QuestionDetailOut,
  QuestionEditIn,
  SourceOut,
} from "@/client/types.gen";
import { Disclosure } from "@/components/disclosure";
import { Field, Select } from "@/components/field";
import { Markdown } from "@/components/markdown";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, fieldErrors } from "@/lib/api-errors";
import { isSubmitKey, useSubmitKeys } from "@/lib/keyboard";
import {
  MAX_KEY_POINTS,
  MAX_POINT_CHARS,
  MAX_QUOTE_CHARS,
  MAX_REASON_CHARS,
  MAX_REFERENCE_CHARS,
  MAX_TEXT_CHARS,
  MIN_KEY_POINTS,
  questionNumber,
} from "@/lib/questions";

import { useEditQuestion } from "./save";

// Correcting a question under the rule generation works to: every key point's quote has to be
// in the passage it names. The API checks each quote before anything changes; a quote it
// can't find is shown beside its key point, and nothing is saved until all of them are there.

type Weight = 1 | 2 | 3;
type PointDraft = { key: number; text: string; weight: Weight; quote: string; passage: number | null };
type Draft = { text: string; reference: string; points: PointDraft[]; reason: string };

export type EditOutcome = "saved" | "unchanged" | "cancelled";

function weightOf(weight: number): Weight {
  return weight <= 1 ? 1 : weight >= 3 ? 3 : 2;
}

function draftOf(question: QuestionDetailOut): Draft {
  return {
    text: question.text,
    reference: question.reference_answer,
    points: question.key_points.map((point, i) => ({
      key: i,
      text: point.text,
      weight: weightOf(point.weight),
      quote: point.evidence_quote,
      passage: point.chunk_id ?? null,
    })),
    reason: "",
  };
}

function samePoints(points: KeyPointIn[], stored: KeyPointOut[]): boolean {
  return (
    points.length === stored.length &&
    points.every(
      (point, i) =>
        point.text === stored[i].text.trim() &&
        point.weight === stored[i].weight &&
        point.evidence_quote === stored[i].evidence_quote.trim() &&
        point.chunk_id === stored[i].chunk_id,
    )
  );
}

/** The correction as the API takes it: only what changed, or null when nothing did. Every
 * key point has its passage by now: the form can't be sent without one. */
function correction(draft: Draft, question: QuestionDetailOut): QuestionEditIn | null {
  const body: QuestionEditIn = {};
  const text = draft.text.trim();
  const reference = draft.reference.trim();
  const points = draft.points.map((point) => ({
    text: point.text.trim(),
    weight: point.weight,
    evidence_quote: point.quote.trim(),
    chunk_id: point.passage ?? 0,
  }));
  if (text !== question.text.trim()) body.text = text;
  if (reference !== question.reference_answer.trim()) body.reference_answer = reference;
  if (!samePoints(points, question.key_points)) body.key_points = points;
  if (Object.keys(body).length === 0) return null;
  const reason = draft.reason.trim();
  return reason ? { ...body, reason } : body;
}

function PointEditor({
  point,
  index,
  sources,
  errors,
  removable,
  onChange,
  onRemove,
}: {
  point: PointDraft;
  index: number;
  sources: SourceOut[];
  errors: Map<string, string>;
  removable: boolean;
  onChange: (change: Partial<PointDraft>) => void;
  onRemove: () => void;
}) {
  const at = `key_points.${index}`;
  const passage = sources.findIndex((source) => source.chunk_id === point.passage);
  return (
    <fieldset data-point={point.key} className="grid min-w-0 gap-3.5 border border-line-2 px-4 pt-1.5 pb-4">
      <legend className="type-label px-1.5">
        Key point {index + 1} · k{index + 1}
      </legend>
      <Field label="What an answer has to say" error={errors.get(`${at}.text`)}>
        {(control) => (
          <Textarea
            {...control}
            required
            maxLength={MAX_POINT_CHARS}
            value={point.text}
            onChange={(event) => onChange({ text: event.target.value })}
            className="min-h-14 py-2.5"
          />
        )}
      </Field>
      <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-[10rem_minmax(0,1fr)]">
        <Field label="Weight · 1 to 3" error={errors.get(`${at}.weight`)}>
          {(control) => (
            <Select
              {...control}
              value={point.weight}
              onChange={(event) => onChange({ weight: weightOf(Number(event.target.value)) })}
            >
              <option value={1}>1 · a detail</option>
              <option value={2}>2</option>
              <option value={3}>3 · the heart of it</option>
            </Select>
          )}
        </Field>
        <Field label="Its passage" error={errors.get(`${at}.chunk_id`)}>
          {(control) => (
            <Select
              {...control}
              required
              value={point.passage ?? ""}
              onChange={(event) =>
                onChange({ passage: event.target.value ? Number(event.target.value) : null })
              }
            >
              <option value="">Choose the passage</option>
              {sources.map((source, j) => (
                <option key={source.chunk_id} value={source.chunk_id}>
                  [{j + 1}] {source.citation}
                </option>
              ))}
            </Select>
          )}
        </Field>
      </div>
      <Field
        label="Its quote, word for word"
        hint="Six or more words copied from the passage, showing the point is there."
        error={errors.get(`${at}.evidence_quote`)}
      >
        {(control) => (
          <Textarea
            {...control}
            required
            maxLength={MAX_QUOTE_CHARS}
            value={point.quote}
            onChange={(event) => onChange({ quote: event.target.value })}
            className="min-h-14 py-2.5 text-small"
          />
        )}
      </Field>
      {passage !== -1 && (
        <Disclosure summary={`Passage ${passage + 1}, to copy from`}>
          <Markdown className="text-small">{sources[passage].text}</Markdown>
        </Disclosure>
      )}
      {removable && (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="justify-self-start"
          onClick={onRemove}
        >
          Remove key point {index + 1}
        </Button>
      )}
    </fieldset>
  );
}

function refusal(error: Error, errors: Map<string, string>): string {
  if (error instanceof ApiError && error.status === null) {
    return "The API can't be reached, so nothing was saved. Start it with make api, then save again.";
  }
  if (errors.size === 0) return `Nothing was saved: ${error.message.replace(/\.$/, "")}.`;
  const general = errors.get("");
  if (general) return `Nothing was saved: ${general}.`;
  return errors.size === 1
    ? "Nothing was saved: one field needs another look (marked below)."
    : `Nothing was saved: ${errors.size} fields need another look (marked below).`;
}

export function EditForm({
  question,
  onDone,
}: {
  question: QuestionDetailOut;
  onDone: (outcome: EditOutcome) => void;
}) {
  const [draft, setDraft] = useState(() => draftOf(question));
  const nextKey = useRef(question.key_points.length);
  const newPoint = useRef<number | null>(null);
  const form = useRef<HTMLFormElement>(null);
  const why = useRef<HTMLParagraphElement>(null);
  const addButton = useRef<HTMLButtonElement>(null);
  const save = useEditQuestion();
  const keys = useSubmitKeys();
  const errors = fieldErrors(save.error);
  const name = questionNumber(question.id);

  useEffect(() => {
    form.current?.querySelector("textarea")?.focus();
  }, []);

  // When the API refuses the change, go to the first field it named, or else to why
  useEffect(() => {
    if (!save.error) return;
    const invalid = form.current?.querySelector<HTMLElement>('[aria-invalid="true"]');
    (invalid ?? why.current)?.focus();
  }, [save.error]);

  // A key point just added: straight into its first field
  useEffect(() => {
    if (newPoint.current === null) return;
    form.current
      ?.querySelector<HTMLElement>(`[data-point="${newPoint.current}"] textarea`)
      ?.focus();
    newPoint.current = null;
  }, [draft.points]);

  function updatePoint(key: number, change: Partial<PointDraft>) {
    setDraft((current) => ({
      ...current,
      points: current.points.map((point) => (point.key === key ? { ...point, ...change } : point)),
    }));
  }

  function addPoint() {
    // The API's problems point at key points by position, which changes now
    save.reset();
    const key = nextKey.current++;
    newPoint.current = key;
    setDraft((current) => ({
      ...current,
      points: [
        ...current.points,
        { key, text: "", weight: 2, quote: "", passage: question.sources[0]?.chunk_id ?? null },
      ],
    }));
  }

  function removePoint(key: number) {
    save.reset();
    setDraft((current) => ({
      ...current,
      points: current.points.filter((point) => point.key !== key),
    }));
    addButton.current?.focus();
  }

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (save.isPending) return;
    const body = correction(draft, question);
    if (body === null) {
      onDone("unchanged");
      return;
    }
    save.mutate({ id: question.id, body }, { onSuccess: () => onDone("saved") });
  }

  return (
    <form
      ref={form}
      aria-label={`Correct ${name}`}
      onSubmit={submit}
      onKeyDown={(event) => {
        if (!isSubmitKey(event)) return;
        event.preventDefault();
        event.currentTarget.requestSubmit();
      }}
      className="grid max-w-[60rem] gap-7"
    >
      <div className="grid gap-1.5">
        <p className="type-label">Correcting {name}</p>
        <p className="max-w-[62ch] text-small text-fg-2">
          Every key point&apos;s quote has to be in the passage it names, word for word. The API
          checks each one before anything changes, and keeps what the correction replaced.
        </p>
      </div>
      <Field label="The question" error={errors.get("text")}>
        {(control) => (
          <Textarea
            {...control}
            required
            maxLength={MAX_TEXT_CHARS}
            value={draft.text}
            onChange={(event) => setDraft((current) => ({ ...current, text: event.target.value }))}
            className="min-h-20 text-[1.1875rem] leading-snug"
          />
        )}
      </Field>
      <Field label="Reference answer" error={errors.get("reference_answer")}>
        {(control) => (
          <Textarea
            {...control}
            required
            maxLength={MAX_REFERENCE_CHARS}
            value={draft.reference}
            onChange={(event) =>
              setDraft((current) => ({ ...current, reference: event.target.value }))
            }
          />
        )}
      </Field>
      <fieldset className="grid min-w-0 gap-4">
        <legend className="type-label mb-3 text-fg-2">
          Key points · {MIN_KEY_POINTS} to {MAX_KEY_POINTS}, replacing the old ones as a whole
        </legend>
        {errors.get("key_points") && (
          <p className="text-small text-thread">{errors.get("key_points")}</p>
        )}
        {draft.points.map((point, i) => (
          <PointEditor
            key={point.key}
            point={point}
            index={i}
            sources={question.sources}
            errors={errors}
            removable={draft.points.length > MIN_KEY_POINTS}
            onChange={(change) => updatePoint(point.key, change)}
            onRemove={() => removePoint(point.key)}
          />
        ))}
        {draft.points.length < MAX_KEY_POINTS && (
          <Button
            ref={addButton}
            type="button"
            variant="outline"
            size="sm"
            className="justify-self-start"
            onClick={addPoint}
          >
            Add a key point
          </Button>
        )}
      </fieldset>
      <Field
        label={
          <>
            Why <span className="text-fg-3">· optional, kept with the change</span>
          </>
        }
        error={errors.get("reason")}
      >
        {(control) => (
          <Textarea
            {...control}
            maxLength={MAX_REASON_CHARS}
            value={draft.reason}
            onChange={(event) => setDraft((current) => ({ ...current, reason: event.target.value }))}
            className="min-h-12 py-2.5 text-small"
          />
        )}
      </Field>
      {save.error && (
        <p ref={why} tabIndex={-1} role="alert" className="max-w-[62ch] text-thread outline-none">
          {refusal(save.error, errors)}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" aria-keyshortcuts={keys.aria} disabled={save.isPending}>
          {save.isPending ? (
            "Saving…"
          ) : (
            <>
              Save changes <kbd aria-hidden>{keys.label}</kbd>
            </>
          )}
        </Button>
        <Button type="button" variant="ghost" onClick={() => onDone("cancelled")}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
