"use client";

import {
  FormEvent,
  KeyboardEvent,
  ReactNode,
  useCallback,
  useEffect,
  useState,
} from "react";

import {
  askRepository,
  checkHealth,
  EvidenceScope,
  Provider,
  QaRequest,
  QaResponse,
  RepoRagApiError,
  RetrievalMethod,
  SourceEvidence,
} from "../lib/api";

type HealthState = "checking" | "connected" | "offline";
type LoadingPhase = "searching" | "generating";

const fieldClass =
  "mt-2 w-full rounded-md border border-white/15 bg-white/[0.06] px-3 py-2.5 text-sm text-stone-100 shadow-sm transition-colors placeholder:text-stone-500 hover:border-white/25 focus:border-amber-500";

const recoveryText: Record<string, string> = {
  api_offline: "Start the RepoRAG API on 127.0.0.1:8000, then retry.",
  repository_not_found: "Check that the local repository path exists on this machine.",
  repository_not_indexed: "Index this repository with RepoRAG before asking questions.",
  incompatible_artifacts: "Re-index the repository to rebuild compatible runtime artifacts.",
  stale_artifacts: "Re-index the repository at its current commit before retrying.",
  git_repository_required: "Choose the Git repository used to build these artifacts.",
  provider_key_missing:
    "Configure the selected provider API key in RepoRAG's local .env file.",
  generation_failed: "Check the local provider configuration and try again.",
  generation_output_limit: "Increase Output tokens and retry the question.",
  no_evidence: "Try a broader evidence scope or rephrase the question.",
};

function titleCase(value: string): string {
  if (value === "bm25") return "BM25";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function StatusBadge({ health, onRetry }: { health: HealthState; onRetry: () => void }) {
  const connected = health === "connected";
  return (
    <div className="flex items-center gap-3 text-xs font-medium">
      <span
        className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 ${
          connected
            ? "border-emerald-800 bg-emerald-950/60 text-emerald-300"
            : "border-stone-700 bg-stone-900 text-stone-300"
        }`}
        aria-live="polite"
      >
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            connected
              ? "bg-emerald-400"
              : health === "checking"
                ? "bg-amber-400"
                : "bg-red-400"
          }`}
          aria-hidden="true"
        />
        API {connected ? "Connected" : health === "checking" ? "Checking" : "Offline"}
      </span>
      {health === "offline" && (
        <button
          type="button"
          onClick={onRetry}
          className="text-stone-300 underline decoration-stone-600 underline-offset-4 hover:text-white"
        >
          Retry
        </button>
      )}
    </div>
  );
}

function AnswerText({ text, onCitation }: { text: string; onCitation: (id: string) => void }) {
  const pieces = text.split(/(\[S\d+\])/g);
  return (
    <div className="whitespace-pre-wrap text-[15px] leading-7 text-stone-800">
      {pieces.map((piece, index) => {
        const sourceId = /^\[(S\d+)\]$/.exec(piece)?.[1];
        return sourceId ? (
          <button
            type="button"
            key={`${sourceId}-${index}`}
            onClick={() => onCitation(sourceId)}
            aria-label={`Show source ${sourceId}`}
            className="mx-0.5 inline rounded bg-amber-100 px-1 py-0.5 font-mono text-xs font-semibold text-amber-900 hover:bg-amber-200"
          >
            {piece}
          </button>
        ) : (
          <span key={index}>{piece}</span>
        );
      })}
    </div>
  );
}

function SourceCard({ source, highlighted }: { source: SourceEvidence; highlighted: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const symbol = source.parent_symbol
    ? [source.parent_symbol, source.symbol].filter(Boolean).join(".")
    : source.symbol;

  return (
    <article
      id={`source-${source.id}`}
      className={`scroll-mt-24 rounded-lg border bg-white transition-colors ${
        highlighted ? "border-amber-500 ring-2 ring-amber-200" : "border-stone-200"
      }`}
    >
      <div className="p-4 sm:p-5">
        <div className="flex items-start gap-3">
          <span className="code-font mt-0.5 rounded bg-stone-900 px-2 py-1 text-[11px] font-semibold text-stone-50">
            {source.id}
          </span>
          <div className="min-w-0 flex-1">
            <p className="code-font break-all text-sm font-semibold text-stone-900">
              {source.file_path}
            </p>
            {symbol && <p className="code-font mt-1 text-xs text-stone-600">{symbol}</p>}
          </div>
        </div>
        <dl className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-stone-500">
          <div>
            <dt className="sr-only">Lines</dt>
            <dd>Lines {source.start_line}–{source.end_line}</dd>
          </div>
          <div>
            <dt className="sr-only">Language and chunk type</dt>
            <dd>{titleCase(source.language)} · {source.chunk_type}</dd>
          </div>
          <div>
            <dt className="sr-only">Retrieval rank</dt>
            <dd>Rank {source.retrieval_rank}</dd>
          </div>
        </dl>
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          aria-expanded={expanded}
          aria-controls={`evidence-${source.id}`}
          className="mt-4 text-xs font-semibold text-amber-800 underline decoration-amber-300 underline-offset-4 hover:text-amber-950"
        >
          {expanded ? "Hide evidence" : "View evidence"}
        </button>
      </div>
      {expanded && (
        <div id={`evidence-${source.id}`} className="border-t border-stone-200 bg-[#202220] p-4 sm:p-5">
          <pre className="code-font overflow-x-auto text-xs leading-6 text-stone-200">
            <code>{source.content}</code>
          </pre>
        </div>
      )}
    </article>
  );
}

function ResultMetadata({ result }: { result: QaResponse }) {
  const items: ReactNode[] = [
    <span key="provider">{titleCase(result.provider)} · {result.model}</span>,
    <span key="retrieval">{titleCase(result.retrieval_method)} · {titleCase(result.evidence_scope)}</span>,
    <span key="sources">{result.sources.length} {result.sources.length === 1 ? "source" : "sources"}</span>,
    <span
      key="citations"
      className={result.citations_valid ? "text-emerald-700" : "text-red-700"}
    >
      {result.citations_valid ? "Citations verified" : "Citation validation failed"}
    </span>,
  ];
  if (result.generation_latency_seconds != null) {
    items.push(<span key="latency">{result.generation_latency_seconds.toFixed(2)}s</span>);
  }
  if (result.usage?.total_tokens != null) {
    items.push(<span key="tokens">{result.usage.total_tokens.toLocaleString()} tokens</span>);
  }
  return <div className="mt-4 flex flex-wrap gap-x-4 gap-y-2 text-xs text-stone-500">{items}</div>;
}

export function RepoRagApp() {
  const [health, setHealth] = useState<HealthState>("checking");
  const [repository, setRepository] = useState("");
  const [question, setQuestion] = useState("");
  const [retrievalMethod, setRetrievalMethod] = useState<RetrievalMethod>("hybrid");
  const [evidenceScope, setEvidenceScope] = useState<EvidenceScope>("all");
  const [provider, setProvider] = useState<Provider>("deepseek");
  const [contextK, setContextK] = useState(5);
  const [maxOutputTokens, setMaxOutputTokens] = useState(2000);
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState<LoadingPhase>("searching");
  const [result, setResult] = useState<QaResponse | null>(null);
  const [error, setError] = useState<{ message: string; recovery?: string } | null>(null);
  const [highlightedSource, setHighlightedSource] = useState<string | null>(null);

  const refreshHealth = useCallback(() => {
    setHealth("checking");
    void checkHealth().then(
      () => setHealth("connected"),
      () => setHealth("offline"),
    );
  }, []);

  useEffect(() => {
    let active = true;
    void checkHealth().then(
      () => {
        if (active) setHealth("connected");
      },
      () => {
        if (active) setHealth("offline");
      },
    );
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!loading) return;
    const timer = window.setTimeout(() => setLoadingPhase("generating"), 650);
    return () => window.clearTimeout(timer);
  }, [loading]);

  const showSource = (sourceId: string) => {
    setHighlightedSource(sourceId);
    document.getElementById(`source-${sourceId}`)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
    window.setTimeout(() => setHighlightedSource(null), 1800);
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!question.trim() || !repository.trim() || loading) return;

    const request: QaRequest = {
      repository,
      question,
      retrieval_method: retrievalMethod,
      evidence_scope: evidenceScope,
      provider,
      context_k: contextK,
      max_output_tokens: maxOutputTokens,
    };
    setLoading(true);
    setLoadingPhase("searching");
    setError(null);
    try {
      const response = await askRepository(request);
      setResult(response);
      setHealth("connected");
    } catch (caught) {
      const apiError = caught instanceof RepoRagApiError ? caught : null;
      if (apiError?.code === "api_offline") setHealth("offline");
      setError({
        message: apiError?.message || "RepoRAG could not complete the request.",
        recovery: apiError ? recoveryText[apiError.code] : undefined,
      });
    } finally {
      setLoading(false);
    }
  };

  const keyboardSubmit = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-[#f5f4ef]">
      <header className="border-b border-white/10 bg-[#171918] px-5 py-4 text-stone-100 sm:px-8">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-5">
          <div>
            <div className="flex items-center gap-3">
              <span className="code-font flex h-8 w-8 items-center justify-center rounded bg-amber-700 text-sm font-bold text-white" aria-hidden="true">R</span>
              <h1 className="text-lg font-semibold tracking-tight">RepoRAG</h1>
            </div>
            <p className="mt-1 pl-11 text-xs text-stone-400">Repository-aware AI code assistant</p>
          </div>
          <StatusBadge health={health} onRetry={refreshHealth} />
        </div>
      </header>

      <form onSubmit={submit} className="mx-auto grid min-h-[calc(100vh-81px)] max-w-[1500px] lg:grid-cols-[310px_minmax(0,1fr)]">
        <aside className="border-b border-stone-800 bg-[#202220] p-5 text-stone-100 sm:p-7 lg:border-r lg:border-b-0">
          <div className="mb-7">
            <p className="code-font text-[10px] font-semibold uppercase tracking-[0.2em] text-amber-500">Configuration</p>
            <p className="mt-2 text-xs leading-5 text-stone-400">Choose the indexed repository and evidence strategy for this question.</p>
          </div>

          <div className="space-y-5">
            <label className="block text-xs font-medium text-stone-300">
              Repository path
              <input
                type="text"
                aria-label="Repository path"
                value={repository}
                onChange={(event) => setRepository(event.target.value)}
                placeholder="D:\Masters-Projects\RealTimeCollab"
                className={`${fieldClass} code-font`}
                required
              />
              <span className="mt-2 block text-[11px] font-normal leading-4 text-stone-500">The repository must already be indexed by RepoRAG.</span>
            </label>

            <label className="block text-xs font-medium text-stone-300">
              Retrieval method
              <select aria-label="Retrieval method" value={retrievalMethod} onChange={(event) => setRetrievalMethod(event.target.value as RetrievalMethod)} className={fieldClass}>
                <option value="hybrid">Hybrid</option>
                <option value="vector">Vector</option>
                <option value="bm25">BM25</option>
                <option value="reranked">Reranked</option>
              </select>
            </label>

            <label className="block text-xs font-medium text-stone-300">
              Evidence scope
              <select aria-label="Evidence scope" value={evidenceScope} onChange={(event) => setEvidenceScope(event.target.value as EvidenceScope)} className={fieldClass}>
                <option value="all">All</option>
                <option value="code">Code</option>
                <option value="docs">Docs</option>
              </select>
            </label>

            <label className="block text-xs font-medium text-stone-300">
              Provider
              <select aria-label="Provider" value={provider} onChange={(event) => setProvider(event.target.value as Provider)} className={fieldClass}>
                <option value="deepseek">DeepSeek</option>
                <option value="gemini">Gemini</option>
                <option value="openai">OpenAI</option>
              </select>
              <span className="mt-2 block text-[11px] font-normal text-stone-500">Credentials stay in the backend environment.</span>
            </label>

            <div className="grid grid-cols-2 gap-3 lg:grid-cols-1 xl:grid-cols-2">
              <label className="block text-xs font-medium text-stone-300">
                Context chunks
                <input aria-label="Context chunks" type="number" min="1" value={contextK} onChange={(event) => setContextK(Number(event.target.value))} className={fieldClass} />
              </label>
              <label className="block text-xs font-medium text-stone-300">
                Output tokens
                <input aria-label="Output tokens" type="number" min="1" value={maxOutputTokens} onChange={(event) => setMaxOutputTokens(Number(event.target.value))} className={fieldClass} />
              </label>
            </div>
          </div>
        </aside>

        <main className="min-w-0 px-5 py-7 sm:px-8 sm:py-9 xl:px-12">
          <div className="mx-auto max-w-5xl">
            <section aria-labelledby="ask-heading">
              <div className="mb-4">
                <p className="code-font text-[10px] font-semibold uppercase tracking-[0.2em] text-amber-700">Question</p>
                <h2 id="ask-heading" className="mt-2 text-2xl font-semibold tracking-tight text-stone-900">Ask RepoRAG</h2>
                <p className="mt-2 text-sm text-stone-600">Ask questions about a software repository and get grounded answers backed by exact source evidence.</p>
              </div>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={keyboardSubmit}
                rows={4}
                placeholder="Where is JWT authentication enforced?"
                aria-label="Repository question"
                className="w-full resize-y rounded-lg border border-stone-300 bg-white px-4 py-3 text-[15px] leading-6 text-stone-900 shadow-sm placeholder:text-stone-400 focus:border-amber-600"
              />
              <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs text-stone-500"><span className="code-font">Ctrl/⌘ + Enter</span> to submit</p>
                <button
                  type="submit"
                  disabled={loading || !question.trim() || !repository.trim()}
                  className="rounded-md bg-stone-900 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-stone-700 disabled:cursor-not-allowed disabled:bg-stone-300 disabled:text-stone-500"
                >
                  {loading ? "Working…" : "Ask RepoRAG"}
                </button>
              </div>
              <div className="min-h-7 pt-3 text-sm" aria-live="polite">
                {loading && (
                  <p className="flex items-center gap-2 text-amber-800">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-amber-600" aria-hidden="true" />
                    {loadingPhase === "searching" ? "Searching repository…" : "Generating grounded answer…"}
                  </p>
                )}
              </div>
            </section>

            <div className="my-7 border-t border-stone-300" />

            {error && (
              <section role="alert" className="rounded-lg border border-red-200 bg-red-50 p-5">
                <p className="text-sm font-semibold text-red-900">Request could not be completed</p>
                <p className="mt-2 text-sm leading-6 text-red-800">{error.message}</p>
                {error.recovery && <p className="mt-2 text-xs leading-5 text-red-700">{error.recovery}</p>}
              </section>
            )}

            {!result && !error && !loading && (
              <section className="rounded-lg border border-stone-200 bg-white p-6 sm:p-8">
                <p className="code-font text-[10px] font-semibold uppercase tracking-[0.2em] text-stone-500">Ready for a question</p>
                <h2 className="mt-3 text-lg font-semibold text-stone-900">Explore the repository with evidence</h2>
                <p className="mt-2 text-sm text-stone-600">Ask about authentication and authorization, realtime connections, retry and recovery behavior, command parsing, or background jobs.</p>
              </section>
            )}

            {result && !loading && (
              <div className="space-y-8">
                <section aria-labelledby="answer-heading">
                  <p className="code-font text-[10px] font-semibold uppercase tracking-[0.2em] text-amber-700">Grounded response</p>
                  <h2 id="answer-heading" className="mt-2 text-xl font-semibold text-stone-900">Answer</h2>
                  <div className="mt-4 rounded-lg border border-stone-200 bg-white p-5 shadow-sm sm:p-7">
                    <AnswerText text={result.answer} onCitation={showSource} />
                    <ResultMetadata result={result} />
                  </div>
                </section>

                <section aria-labelledby="sources-heading">
                  <div className="mb-4 flex items-end justify-between gap-4">
                    <div>
                      <p className="code-font text-[10px] font-semibold uppercase tracking-[0.2em] text-amber-700">Retrieved evidence</p>
                      <h2 id="sources-heading" className="mt-2 text-xl font-semibold text-stone-900">Sources</h2>
                    </div>
                    <p className="text-xs text-stone-500">Ranked retrieval order</p>
                  </div>
                  <div className="space-y-3">
                    {result.sources.map((source) => (
                      <SourceCard key={source.id} source={source} highlighted={highlightedSource === source.id} />
                    ))}
                  </div>
                </section>
              </div>
            )}
          </div>
        </main>
      </form>
    </div>
  );
}
