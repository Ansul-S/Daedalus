import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { Sheet } from "@/components/sheet";
import { questionNumber } from "@/lib/questions";

import { QuestionSheet } from "./question-sheet";

// The largest id the database's integer column holds
const LARGEST_ID = 2_147_483_647;

function questionId(segment: string): number | null {
  if (!/^\d+$/.test(segment)) return null;
  const id = Number(segment);
  return id >= 1 && id <= LARGEST_ID ? id : null;
}

export async function generateMetadata({ params }: PageProps<"/questions/[id]">): Promise<Metadata> {
  const id = questionId((await params).id);
  return { title: id === null ? "Not found" : questionNumber(id) };
}

export default async function QuestionPage({ params }: PageProps<"/questions/[id]">) {
  const id = questionId((await params).id);
  if (id === null) notFound();
  return (
    <Sheet>
      <QuestionSheet id={id} />
    </Sheet>
  );
}
