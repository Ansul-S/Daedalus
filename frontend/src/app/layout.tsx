import type { Metadata, Viewport } from "next";
import { Big_Shoulders, JetBrains_Mono, Source_Serif_4 } from "next/font/google";

import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { THEME_SCRIPT } from "@/lib/theme-script";

import { Providers } from "./providers";
import "./globals.css";

// Self-hosted at build time: no request goes to Google when the app runs.
const display = Big_Shoulders({
  subsets: ["latin"],
  axes: ["opsz"],
  variable: "--font-big-shoulders",
  // next/font has no metrics for it to size a fallback by; name narrow faces instead
  adjustFontFallback: false,
  fallback: ["Arial Narrow", "Helvetica Neue", "sans-serif"],
});
// Greek for the inscription and the sigils; the glyph pictures are drawn in the mono.
const serif = Source_Serif_4({
  subsets: ["latin", "greek"],
  style: ["normal", "italic"],
  axes: ["opsz"],
  variable: "--font-source-serif",
});
const mono = JetBrains_Mono({
  subsets: ["latin", "greek"],
  variable: "--font-jetbrains-mono",
});

export const metadata: Metadata = {
  title: { default: "Daedalus", template: "%s · Daedalus" },
  description:
    "Practice that remembers what you forget: AI/ML interview questions written from your own study material, graded claim by claim against the same sources.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f5f0e4" },
    { media: "(prefers-color-scheme: dark)", color: "#1c1411" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${display.variable} ${serif.variable} ${mono.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="flex min-h-dvh flex-col">
        <Providers>
          <SiteHeader />
          <main className="flex-1 px-(--gutter) pt-9">{children}</main>
          <SiteFooter />
        </Providers>
      </body>
    </html>
  );
}
