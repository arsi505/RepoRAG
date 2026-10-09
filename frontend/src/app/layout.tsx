import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RepoRAG — Repository-aware AI code assistant",
  description:
    "Ask questions about a software repository and get grounded answers backed by exact source evidence.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
