"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ApiError } from "@/lib/api-errors";

function retry(failures: number, error: Error): boolean {
  // A request the API refused (4xx) fails the same way again; try the rest once more.
  const refused = error instanceof ApiError && error.status !== null && error.status < 500;
  return !refused && failures < 1;
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          // A question must not change under the reader because the window regained focus.
          queries: { retry, refetchOnWindowFocus: false, staleTime: 60_000 },
        },
      }),
  );

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        {children}
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}
