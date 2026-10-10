import { useState, type CSSProperties } from "react";

import type { SourceEvidence } from "../lib/api";

interface EvidenceExplorerProps {
  sources: SourceEvidence[];
  selectedSourceId: string | null;
  citationTargetId: string | null;
  citedSourceIds: string[];
  loading: boolean;
  loadingPhase: "searching" | "generating";
  onSelectSource: (sourceId: string) => void;
}

function titleCase(value: string): string {
  if (value === "bm25") return "BM25";
  if (value === "typescript") return "TypeScript";
  if (value === "javascript") return "JavaScript";
  if (value === "cpp") return "C++";
  if (value === "tsx") return "TSX";
  if (value === "jsx") return "JSX";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function sourceSymbol(source: SourceEvidence): string | null {
  return source.parent_symbol
    ? [source.parent_symbol, source.symbol].filter(Boolean).join(".")
    : source.symbol;
}

function SourceViewer({ source, emphasized, cited }: { source: SourceEvidence; emphasized: boolean; cited: boolean }) {
  const symbol = sourceSymbol(source);
  const lines = source.content.split("\n");
  const [copied, setCopied] = useState(false);

  const copySource = async () => {
    try {
      await navigator.clipboard.writeText(source.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <section
      id="source-viewer"
      className={`source-viewer source-viewer-enter ${emphasized ? "source-viewer-emphasized" : ""}`}
      aria-labelledby="source-viewer-heading"
    >
      <div className="source-viewer-toolbar">
        <div className="flex min-w-0 items-start gap-3">
        <span className="source-badge">{source.id}</span>
        <div className="min-w-0">
          <h3 id="source-viewer-heading" className="code-font break-words text-[13.5px] font-semibold leading-5 text-[var(--text-primary)]">
            {source.file_path}
          </h3>
          {symbol && <p className="code-font mt-1 text-[12.5px] text-[#c7d9ee]">{symbol}</p>}
        </div>
        </div>
        <div className="source-viewer-actions">
          {cited && <span className="cited-badge">Cited</span>}
          <button type="button" className={`utility-button utility-compact ${copied ? "utility-success" : ""}`} onClick={() => void copySource()}>
            {copied ? "Copied" : "Copy source"}
          </button>
        </div>
      </div>
      <p className="mb-3 text-[12px] text-[var(--text-muted)]">
        Lines {source.start_line}–{source.end_line} · {titleCase(source.language)} · {source.chunk_type} · Rank {source.retrieval_rank}
      </p>
      <div className="code-viewer" role="region" aria-label={`${source.id} source code`} tabIndex={0}>
        <table>
          <tbody>
            {lines.map((line, index) => (
              <tr
                key={`${source.start_line + index}-${index}`}
                style={{ "--line-index": Math.min(index, 14) } as CSSProperties}
              >
                <td className="line-number" aria-hidden="true">{source.start_line + index}</td>
                <td className="code-line"><code>{line || " "}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function EvidenceSkeletons({ phase }: { phase: "searching" | "generating" }) {
  return (
    <div className="evidence-loading" aria-label={phase === "searching" ? "Retrieving evidence" : "Preparing evidence context"}>
      <div className="evidence-loading-label">
        <span className="spinner" aria-hidden="true" />
        {phase === "searching" ? "Ranking source evidence" : "Preparing grounded context"}
      </div>
      <div className="skeleton-source-list" aria-hidden="true">
        {Array.from({ length: 5 }, (_, index) => (
          <div className="skeleton-source" key={index} style={{ "--skeleton-index": index } as CSSProperties}>
            <span className="skeleton-badge" />
            <span className="skeleton-copy"><span /><span /><span /></span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function EvidenceExplorer({
  sources,
  selectedSourceId,
  citationTargetId,
  citedSourceIds,
  loading,
  loadingPhase,
  onSelectSource,
}: EvidenceExplorerProps) {
  const selected = sources.find((source) => source.id === selectedSourceId) ?? sources[0] ?? null;

  return (
    <aside className="glass-panel evidence-panel" aria-label="Evidence explorer">
      <div className="panel-heading-row">
        <div>
          <p className="eyebrow">Ranked retrieval</p>
          <h2 className="panel-title">Evidence</h2>
        </div>
        <span className="panel-index">{sources.length.toString().padStart(2, "0")}</span>
      </div>

      {loading ? (
        <EvidenceSkeletons phase={loadingPhase} />
      ) : sources.length === 0 ? (
        <div className="evidence-empty">
          <div className="evidence-stack" aria-hidden="true">
            <span /><span /><span />
          </div>
          <p className="text-[15px] font-semibold text-[#c9d8e9]">Evidence will appear here</p>
          <p className="mx-auto mt-1.5 max-w-[230px] text-[13.5px] leading-5 text-[var(--text-secondary)]">Ask a question to inspect ranked source chunks and exact code.</p>
        </div>
      ) : (
        <>
          <div className="source-list" role="listbox" aria-label="Ranked sources">
            {sources.map((source) => {
              const symbol = sourceSymbol(source);
              const selectedRow = source.id === selected?.id;
              const cited = citedSourceIds.includes(source.id);
              return (
                <button
                  type="button"
                  role="option"
                  key={source.id}
                  className={`source-row ${selectedRow ? "source-selected" : ""} ${citationTargetId === source.id ? "source-citation-target" : ""}`}
                  onClick={() => onSelectSource(source.id)}
                  aria-selected={selectedRow}
                  title={source.file_path}
                >
                  <span className="source-id-stack"><span className="source-badge">{source.id}</span>{cited && <span className="cited-badge">Cited</span>}</span>
                  <span className="min-w-0 flex-1 text-left">
                    <span className="code-font block truncate text-[13.5px] font-semibold text-[var(--text-primary)]">{source.file_path}</span>
                    {symbol && <span className="code-font mt-0.5 block truncate text-[12.5px] text-[#c7d9ee]">{symbol}</span>}
                    <span className="mt-1 block text-[12px] text-[var(--text-muted)]">
                      {titleCase(source.language)} · {source.chunk_type} · Rank {source.retrieval_rank}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
          {selected && (
            <SourceViewer
              key={selected.id}
              source={selected}
              emphasized={citationTargetId === selected.id}
              cited={citedSourceIds.includes(selected.id)}
            />
          )}
        </>
      )}
    </aside>
  );
}
