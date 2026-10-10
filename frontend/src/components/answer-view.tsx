import { useState } from "react";
import ReactMarkdown from "react-markdown";

import type { QaResponse } from "../lib/api";

interface AnswerViewProps {
  result: QaResponse;
  onCitation: (sourceId: string) => void;
}

function titleCase(value: string): string {
  if (value === "bm25") return "BM25";
  if (value === "deepseek") return "DeepSeek";
  if (value === "openai") return "OpenAI";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function citationLinks(markdown: string): string {
  return markdown
    .split(/(```[\s\S]*?```|`[^`\n]+`)/g)
    .map((part) => (part.startsWith("`") ? part : part.replace(/\[(S\d+)\]/g, "[$1](#source-$1)")))
    .join("");
}

function ResultMetadata({ result }: { result: QaResponse }) {
  return (
    <div className="metadata-chips" aria-label="Answer metadata">
      <span className="metadata-chip">{titleCase(result.provider)} · {result.model}</span>
      <span className="metadata-chip">{titleCase(result.retrieval_method)}</span>
      <span className="metadata-chip">{titleCase(result.evidence_scope)} evidence</span>
      <span className="metadata-chip">{result.sources.length} {result.sources.length === 1 ? "source" : "sources"}</span>
      <span className={`metadata-chip ${result.citations_valid ? "metadata-success" : "metadata-error"}`}>
        {result.citations_valid && <span className="validation-check" aria-hidden="true">✓</span>}
        {result.citations_valid ? "Citations verified" : "Citation validation failed"}
      </span>
      {result.generation_latency_seconds != null && (
        <span className="metadata-chip">{result.generation_latency_seconds.toFixed(2)}s</span>
      )}
      {result.usage?.total_tokens != null && (
        <span className="metadata-chip">{result.usage.total_tokens.toLocaleString()} tokens</span>
      )}
    </div>
  );
}

export function AnswerView({ result, onCitation }: AnswerViewProps) {
  const [copied, setCopied] = useState(false);

  const copyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(result.answer);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const exportAnswer = () => {
    const sources = result.sources.map((source) => {
      const symbol = sourceSymbol(source);
      return `- [${source.id}] \`${source.file_path}\`${symbol ? ` — \`${symbol}\`` : ""} (lines ${source.start_line}–${source.end_line})`;
    }).join("\n");
    const markdown = `# RepoRAG answer\n\n## Question\n\n${result.question}\n\n## Answer\n\n${result.answer}\n\n## Retrieved evidence\n\n${sources}\n`;
    const url = URL.createObjectURL(new Blob([markdown], { type: "text/markdown;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "reporag-answer.md";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section aria-labelledby="answer-heading" className="answer-section">
      <div className="answer-heading-row">
        <div>
          <p className="eyebrow">Grounded response</p>
          <h2 id="answer-heading" className="mt-1.5 text-[21px] font-semibold tracking-[-0.015em] text-[var(--text-primary)]">Answer</h2>
        </div>
        <div className="answer-actions">
          <span className="copy-status" aria-live="polite">{copied ? "Copied" : ""}</span>
          <button type="button" className={`utility-button ${copied ? "utility-success" : ""}`} onClick={() => void copyAnswer()}>
            {copied ? "Copied" : "Copy answer"}
          </button>
          <button type="button" className="utility-button" onClick={exportAnswer}>Export .md</button>
        </div>
      </div>
      <div className="answer-surface">
        <div className="answer-surface-header" aria-hidden="true">
          <span className="answer-signal"><span /> Evidence-backed response</span>
          <span className="answer-rule" />
        </div>
        <div className="answer-markdown">
          <ReactMarkdown
            components={{
              a: ({ href, children }) => {
                const sourceId = /^#source-(S\d+)$/.exec(href ?? "")?.[1];
                if (sourceId) {
                  return (
                    <button
                      type="button"
                      onClick={() => onCitation(sourceId)}
                      aria-label={`Show source ${sourceId}`}
                      className="citation-marker"
                    >
                      {children}
                    </button>
                  );
                }
                return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
              },
              pre: ({ children }) => <pre>{children}</pre>,
              code: ({ className, children }) => {
                const block = Boolean(className) || String(children).includes("\n");
                return block ? <code className={className}>{children}</code> : <code>{children}</code>;
              },
            }}
          >
            {citationLinks(result.answer)}
          </ReactMarkdown>
        </div>
        <ResultMetadata result={result} />
      </div>
    </section>
  );
}

function sourceSymbol(source: QaResponse["sources"][number]): string | null {
  return source.parent_symbol
    ? [source.parent_symbol, source.symbol].filter(Boolean).join(".")
    : source.symbol;
}
