"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { practiceProgressOptions } from "@/client/@tanstack/react-query.gen";
import { LabyrinthMark } from "@/components/labyrinth-mark";
import { ThemeSwitch } from "@/components/theme-switch";
import { number } from "@/lib/progress";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/practice", label: "Practice" },
  { href: "/questions", label: "Questions" },
  { href: "/library", label: "Library" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/setup", label: "Setup" },
] as const;

/** The streak and the XP, as practice left them; nothing while the API can't be reached. */
function Standing() {
  const { data } = useQuery(practiceProgressOptions());
  if (!data) return null;
  const { days } = data.streak;
  return (
    <p className="hidden font-mono text-[10.5px] leading-none tracking-[0.1em] whitespace-nowrap text-fg-2 uppercase sm:block">
      {days > 0 && `${days}-day thread · `}
      {number(data.xp)} XP
    </p>
  );
}

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-ground/93 backdrop-blur-[6px]">
      <div className="mx-auto flex max-w-[calc(1240px+2*var(--gutter))] flex-wrap items-center gap-x-6 gap-y-2 px-(--gutter) py-2.5">
        <Link
          href="/"
          className="inline-flex items-center gap-2 font-display text-[1.05rem] leading-none font-extrabold tracking-[0.08em] uppercase"
        >
          <LabyrinthMark className="h-6 w-[22px]" />
          Daedalus
        </Link>
        <nav
          aria-label="Main"
          className="order-last -mx-(--gutter) w-[calc(100%+2*var(--gutter))] overflow-x-auto px-(--gutter) md:order-none md:mx-0 md:w-auto md:px-0"
        >
          <ul className="flex gap-[18px] py-1.5 font-mono text-[10.5px] leading-none tracking-[0.1em] whitespace-nowrap uppercase">
            {NAV.map((item) => {
              const current = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={current ? "page" : undefined}
                    className={cn(
                      "inline-block pb-[5px] hover:text-fg-2",
                      current && "bg-[linear-gradient(var(--thread),var(--thread))] bg-[length:100%_2px] bg-bottom bg-no-repeat",
                    )}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
        <div className="ml-auto flex items-center gap-x-6">
          <Standing />
          <ThemeSwitch />
        </div>
      </div>
    </header>
  );
}
