import { cn } from "@/lib/utils";

// Every page is a drawing sheet: a hairline frame with registration marks at its corners, a
// sheet number, a title with a Greek sigil, and a title block along the bottom.

const CORNERS = ["-top-[9px] -left-[9px]", "-top-[9px] -right-[9px]", "-bottom-[9px] -left-[9px]", "-right-[9px] -bottom-[9px]"];

export function Sheet({ className, children, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      className={cn("relative mx-auto w-full max-w-[1240px] border border-line-2 bg-ground", className)}
      {...props}
    >
      {CORNERS.map((corner) => (
        <span
          key={corner}
          aria-hidden
          className={cn(
            "pointer-events-none absolute size-[17px] before:absolute before:top-0 before:left-2 before:h-[17px] before:w-px before:bg-line-2 after:absolute after:top-2 after:left-0 after:h-px after:w-[17px] after:bg-line-2",
            corner,
          )}
        />
      ))}
      {children}
    </div>
  );
}

export function SheetSection({ className, ...props }: React.ComponentProps<"section">) {
  return (
    <section
      className={cn(
        "border-line-2 px-[clamp(18px,4.5vw,64px)] py-[clamp(44px,6.5vw,96px)] not-first-of-type:border-t",
        className,
      )}
      {...props}
    />
  );
}

/** The sheet number, the title with its sigil, and a lede under the title. */
export function SheetHead({
  number,
  title,
  sigil,
  as: Heading = "h1",
  id,
  children,
  className,
}: {
  number: string;
  title: React.ReactNode;
  sigil?: string;
  as?: "h1" | "h2";
  id?: string;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <header
      className={cn(
        "mb-[clamp(28px,4vw,52px)] grid grid-cols-1 items-baseline gap-x-8 gap-y-2 md:grid-cols-[9.5rem_minmax(0,1fr)]",
        className,
      )}
    >
      <span className="type-label text-fg-2">{number}</span>
      <Heading id={id} className="type-title">
        {title}
        {sigil && (
          <sup
            lang="el"
            className="relative top-[0.1em] ml-[0.14em] align-top font-serif text-[0.38em] font-normal tracking-normal text-thread normal-case italic"
          >
            {sigil}
          </sup>
        )}
      </Heading>
      {children && <div className="mt-3 max-w-[62ch] text-fg-2 md:col-start-2">{children}</div>}
    </header>
  );
}

/** The title block along the foot of a sheet: label over value, cell by cell. */
export function TitleBlock({
  cells,
  className,
}: {
  cells: { label: string; value: React.ReactNode }[];
  className?: string;
}) {
  return (
    <dl
      className={cn(
        "grid grid-cols-2 border-t border-line-2 sm:grid-cols-[repeat(auto-fit,minmax(9rem,1fr))]",
        className,
      )}
    >
      {cells.map(({ label, value }) => (
        <div
          key={label}
          className="grid gap-1 border-line-2 px-3.5 pt-2.5 pb-3 not-first:border-l max-sm:border-t max-sm:odd:border-l-0 max-sm:[&:nth-child(-n+2)]:border-t-0"
        >
          <dt className="font-mono text-[9px] leading-none font-medium tracking-[0.12em] text-fg-2 uppercase">
            {label}
          </dt>
          <dd className="font-display text-[1.05rem] leading-[1.1] font-bold tracking-[0.04em] uppercase">
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}
