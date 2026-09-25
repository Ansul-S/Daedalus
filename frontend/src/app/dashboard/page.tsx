import type { Metadata } from "next";

import { Sheet, SheetHead } from "@/components/sheet";

import { Dashboard } from "./dashboard";

export const metadata: Metadata = { title: "Dashboard" };

export default function DashboardPage() {
  return (
    <Sheet>
      <Dashboard>
        <SheetHead number="Sheet D-01" title="Dashboard" sigil="μ">
          Each room of the labyrinth is one of your topics, and practice fills it with hatching.
          The Minotaur waits in the weakest room. The wing grows a feather with every step
          towards the next level.
        </SheetHead>
      </Dashboard>
    </Sheet>
  );
}
