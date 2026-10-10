"use client";

import { FormEvent, KeyboardEvent, useCallback, useEffect, useState } from "react";

import { AnswerView } from "./answer-view";
import { AppHeader, type HealthState } from "./app-header";
import { ConfigurationPanel } from "./configuration-panel";
import { CommandPalette } from "./command-palette";
import { EvidenceExplorer } from "./evidence-explorer";
import { QuestionComposer } from "./question-composer";
import { RepositoryGraph } from "./repository-graph";
import {
  askRepository,
  checkHealth,
  type EvidenceScope,
  type Provider,
  type QaRequest,
  type QaResponse,
  RepoRagApiError,
  type RetrievalMethod,
} from "../lib/api";

type LoadingPhase = "searching" | "generating";

function RetrievalJourney({ phase }: { phase: LoadingPhase }) {
  const generating = phase === "generating";

  return (
    <div className={`retrieval-journey ${generating ? "journey-generating" : "journey-searching"}`} role="status">
      <div className="journey-visual" aria-hidden="true">
        <span className="journey-node journey-query">Q</span>
        <span className="journey-path"><span /></span>
        <span className="journey-node journey-evidence">5</span>
        <span className="journey-path"><span /></span>
        <span className="journey-node journey-answer">A</span>
      </div>
      <div className="min-w-0">
        <p className="loading-message">
          {generating ? "Generating grounded response…" : "Retrieving repository evidence…"}
        </p>
        <p className="journey-detail">
          {generating ? "Composing from the selected source context" : "Ranking candidate chunks across the indexed corpus"}
        </p>
      </div>
    </div>
  );
}

const recoveryText: Record<string, string> = {
  api_offline: "Start the RepoRAG API on 127.0.0.1:8000, then retry.",
  repository_not_found: "Check that the local repository path exists on this machine.",
  repository_not_indexed: "Index this repository with RepoRAG before asking questions.",
  incompatible_artifacts: "Re-index the repository to rebuild compatible runtime artifacts.",
  stale_artifacts: "Re-index the repository at its current commit before retrying.",
  git_repository_required: "Choose the Git repository used to build these artifacts.",
  provider_key_missing: "Configure the selected provider API key in RepoRAG's backend .env file.",
  generation_failed: "Check the local provider configuration and try again.",
  generation_output_limit: "Increase Output tokens and retry the question.",
  no_evidence: "Try a broader evidence scope or rephrase the question.",
};

const examplePrompts = [
  "Where is authentication enforced?",
  "How does retry behavior work?",
  "Where are realtime connections handled?",
  "How is failure recovery implemented?",
];

function EmptyState({ onPrompt }: { onPrompt: (prompt: string) => void }) {
  return (
    <section className="empty-state" aria-labelledby="empty-state-heading">
      <RepositoryGraph />
      <p className="eyebrow">Grounded repository analysis</p>
      <h2 id="empty-state-heading" className="mt-2 text-[19px] font-semibold text-[var(--text-primary)]">
        Ask about this repository
      </h2>
      <p className="mx-auto mt-2 max-w-2xl text-[14px] leading-6 text-[var(--text-secondary)]">
        Explore implementation, architecture, reliability, and behavior using grounded repository evidence.
      </p>
      <div className="capability-row" aria-label="RepoRAG capabilities">
        <span>Trace implementation</span>
        <span>Validate citations</span>
        <span>Inspect exact source</span>
      </div>
      <div className="prompt-grid">
        {examplePrompts.map((prompt) => (
          <button key={prompt} type="button" className="prompt-card" onClick={() => onPrompt(prompt)}>
            <span>{prompt}</span><span aria-hidden="true">↗</span>
          </button>
        ))}
      </div>
    </section>
  );
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
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [configOpen, setConfigOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const [focusMode, setFocusMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState<LoadingPhase>("searching");
  const [result, setResult] = useState<QaResponse | null>(null);
  const [error, setError] = useState<{ message: string; recovery?: string } | null>(null);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [citationTargetId, setCitationTargetId] = useState<string | null>(null);

  const refreshHealth = useCallback(() => {
    setHealth("checking");
    void checkHealth().then(() => setHealth("connected"), () => setHealth("offline"));
  }, []);

  useEffect(() => {
    let active = true;
    void checkHealth().then(
      () => { if (active) setHealth("connected"); },
      () => { if (active) setHealth("offline"); },
    );
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!loading) return;
    const timer = window.setTimeout(() => setLoadingPhase("generating"), 650);
    return () => window.clearTimeout(timer);
  }, [loading]);

  useEffect(() => {
    if (!citationTargetId) return;
    const timer = window.setTimeout(() => setCitationTargetId(null), 1100);
    return () => window.clearTimeout(timer);
  }, [citationTargetId]);

  useEffect(() => {
    const openCommands = (event: globalThis.KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCommandOpen((value) => !value);
      }
    };
    window.addEventListener("keydown", openCommands);
    return () => window.removeEventListener("keydown", openCommands);
  }, []);

  const selectSource = useCallback((sourceId: string, reveal = false) => {
    setSelectedSourceId(sourceId);
    setCitationTargetId(reveal ? sourceId : null);
    if (reveal) {
      window.requestAnimationFrame(() => {
        document.getElementById("source-viewer")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    }
  }, []);

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
      setSelectedSourceId(response.sources[0]?.id ?? null);
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
    <div className={`app-shell ${focusMode ? "focus-mode" : ""}`}>
      <AppHeader
        health={health}
        repository={repository}
        configOpen={configOpen}
        focusMode={focusMode}
        onRetry={refreshHealth}
        onToggleConfig={() => setConfigOpen((value) => !value)}
        onToggleFocusMode={() => setFocusMode((value) => !value)}
        onOpenCommands={() => setCommandOpen(true)}
      />
      <form onSubmit={submit} className="workspace-grid">
        <ConfigurationPanel
          repository={repository}
          retrievalMethod={retrievalMethod}
          evidenceScope={evidenceScope}
          provider={provider}
          contextK={contextK}
          maxOutputTokens={maxOutputTokens}
          advancedOpen={advancedOpen}
          configOpen={configOpen}
          onRepositoryChange={setRepository}
          onRetrievalMethodChange={setRetrievalMethod}
          onEvidenceScopeChange={setEvidenceScope}
          onProviderChange={setProvider}
          onContextKChange={setContextK}
          onMaxOutputTokensChange={setMaxOutputTokens}
          onToggleAdvanced={() => setAdvancedOpen((value) => !value)}
        />

        <main className="glass-panel center-panel">
          <QuestionComposer
            question={question}
            loading={loading}
            canSubmit={Boolean(question.trim() && repository.trim())}
            retrievalMethod={retrievalMethod}
            evidenceScope={evidenceScope}
            provider={provider}
            onQuestionChange={setQuestion}
            onKeyboardSubmit={keyboardSubmit}
          />
          <div className="status-region" aria-live="polite">
            {loading && <RetrievalJourney phase={loadingPhase} />}
          </div>

          {health === "offline" && !error && (
            <section className="offline-notice" aria-label="API offline notice">
              <div>
                <p className="font-semibold text-[var(--text-primary)]">RepoRAG API is offline.</p>
                <p className="mt-1 text-[13px] text-[var(--text-secondary)]">Start the local API on 127.0.0.1:8000.</p>
              </div>
              <button type="button" onClick={refreshHealth} className="secondary-button">Retry</button>
            </section>
          )}

          {error && (
            <section role="alert" className="error-block">
              <p className="font-semibold text-[#fecdd3]">Request could not be completed</p>
              <p className="mt-2 text-[14px] leading-6 text-[#fda4af]">{error.message}</p>
              {error.recovery && <p className="mt-2 text-[13px] leading-5 text-[#fb7185]">{error.recovery}</p>}
            </section>
          )}

          {!result && !error && !loading && <EmptyState onPrompt={setQuestion} />}
          {result && !loading && <AnswerView result={result} onCitation={(sourceId) => selectSource(sourceId, true)} />}
        </main>

        <EvidenceExplorer
          sources={result?.sources ?? []}
          citedSourceIds={result?.cited_source_ids ?? []}
          selectedSourceId={selectedSourceId}
          citationTargetId={citationTargetId}
          loading={loading}
          loadingPhase={loadingPhase}
          onSelectSource={(sourceId) => selectSource(sourceId)}
        />
      </form>
      <CommandPalette
        open={commandOpen}
        focusMode={focusMode}
        onClose={() => setCommandOpen(false)}
        onFocusRepository={() => document.querySelector<HTMLInputElement>('[aria-label="Repository path"]')?.focus()}
        onFocusQuestion={() => document.querySelector<HTMLTextAreaElement>('[aria-label="Repository question"]')?.focus()}
        onToggleFocusMode={() => setFocusMode((value) => !value)}
        onSetRetrievalMethod={setRetrievalMethod}
        onSetEvidenceScope={setEvidenceScope}
        onOpenAdvanced={() => { setAdvancedOpen(true); setConfigOpen(true); }}
      />
    </div>
  );
}
