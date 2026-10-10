import type { EvidenceScope, Provider, RetrievalMethod } from "../lib/api";

interface QuestionComposerProps {
  question: string;
  loading: boolean;
  canSubmit: boolean;
  retrievalMethod: RetrievalMethod;
  evidenceScope: EvidenceScope;
  provider: Provider;
  onQuestionChange: (value: string) => void;
  onKeyboardSubmit: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
}

function display(value: string): string {
  if (value === "bm25") return "BM25";
  if (value === "deepseek") return "DeepSeek";
  if (value === "openai") return "OpenAI";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function QuestionComposer({
  question,
  loading,
  canSubmit,
  retrievalMethod,
  evidenceScope,
  provider,
  onQuestionChange,
  onKeyboardSubmit,
}: QuestionComposerProps) {
  return (
    <section aria-labelledby="ask-heading">
      <div className="mb-4">
        <p className="eyebrow">Repository question</p>
        <h2 id="ask-heading" className="mt-1.5 text-[20px] font-semibold tracking-[-0.015em] text-[var(--text-primary)]">
          Ask RepoRAG
        </h2>
        <p className="section-intro">Trace behavior across the repository and answer from retrieved source evidence.</p>
      </div>

      <div className="composer">
        <textarea
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          onKeyDown={onKeyboardSubmit}
          rows={4}
          placeholder="Ask anything about this repository…"
          aria-label="Repository question"
          aria-describedby="question-shortcut"
        />
        <div className="composer-toolbar">
          <div className="composer-context" aria-label="Active question configuration">
            <span className="context-pill"><span aria-hidden="true" />{display(retrievalMethod)}</span>
            <span className="context-pill"><span aria-hidden="true" />{display(evidenceScope)}</span>
            <span className="context-pill"><span aria-hidden="true" />{display(provider)}</span>
          </div>
          <button type="submit" className="ask-button" disabled={!canSubmit || loading}>
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Working
              </>
            ) : (
              <>
                Ask RepoRAG
                <span className="code-font text-[13.5px] font-normal opacity-70">↵</span>
              </>
            )}
          </button>
        </div>
      </div>
      <p id="question-shortcut" className="mt-2.5 text-[13px] text-[var(--text-muted)]">
        <span className="code-font">Ctrl/⌘ + Enter</span> to submit
      </p>
    </section>
  );
}
