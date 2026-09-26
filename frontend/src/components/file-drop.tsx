"use client";

import { useEffect, useId, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** Files dropped on a dashed box, or chosen with the button inside it. While the box is on
 * the page, a file dropped beside it is caught rather than opened in place of the page. */
export function FileDrop({
  accept,
  onFiles,
  disabled = false,
  children,
  className,
}: {
  accept: string[];
  onFiles: (files: File[]) => void;
  disabled?: boolean;
  children: React.ReactNode;
  className?: string;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const words = useId();

  useEffect(() => {
    const stray = (event: DragEvent) => {
      if (event.dataTransfer?.types.includes("Files")) event.preventDefault();
    };
    window.addEventListener("dragover", stray);
    window.addEventListener("drop", stray);
    return () => {
      window.removeEventListener("dragover", stray);
      window.removeEventListener("drop", stray);
    };
  }, []);

  function take(list: FileList | null) {
    const files = Array.from(list ?? []);
    if (files.length > 0) onFiles(files);
  }

  function carrying(event: React.DragEvent) {
    if (disabled || !event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setOver(true);
  }

  return (
    <div
      onDragEnter={carrying}
      onDragOver={carrying}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOver(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        if (!disabled) take(event.dataTransfer.files);
      }}
      className={cn(
        "grid justify-items-start gap-3 border-[1.5px] border-dashed border-line-2 px-4 py-4 transition-colors",
        over && "border-solid border-fg bg-surface",
        className,
      )}
    >
      <p id={words} className="text-small text-fg-2">
        {children}
      </p>
      <input
        ref={input}
        type="file"
        multiple
        accept={accept.join(",")}
        tabIndex={-1}
        aria-hidden
        className="sr-only"
        onChange={(event) => {
          take(event.target.files);
          // The same file can be chosen again
          event.target.value = "";
        }}
      />
      <Button
        type="button"
        variant="outline"
        size="sm"
        disabled={disabled}
        aria-describedby={words}
        onClick={() => input.current?.click()}
      >
        Choose files
      </Button>
    </div>
  );
}
