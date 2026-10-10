import type { EvidenceScope, Provider, RetrievalMethod } from "../lib/api";

interface ConfigurationPanelProps {
  repository: string;
  retrievalMethod: RetrievalMethod;
  evidenceScope: EvidenceScope;
  provider: Provider;
  contextK: number;
  maxOutputTokens: number;
  advancedOpen: boolean;
  configOpen: boolean;
  onRepositoryChange: (value: string) => void;
  onRetrievalMethodChange: (value: RetrievalMethod) => void;
  onEvidenceScopeChange: (value: EvidenceScope) => void;
  onProviderChange: (value: Provider) => void;
  onContextKChange: (value: number) => void;
  onMaxOutputTokensChange: (value: number) => void;
  onToggleAdvanced: () => void;
}

const methodAccent: Record<RetrievalMethod, string> = {
  hybrid: "accent-blue",
  vector: "accent-cyan",
  bm25: "accent-purple",
  reranked: "accent-indigo",
};

const scopeAccent: Record<EvidenceScope, string> = {
  all: "accent-blue",
  code: "accent-cyan",
  docs: "accent-purple",
};

const providerAccent: Record<Provider, string> = {
  deepseek: "accent-cyan",
  gemini: "accent-purple",
  openai: "accent-green",
};

function SelectField({
  label,
  ariaLabel,
  value,
  accent,
  onChange,
  children,
}: {
  label: string;
  ariaLabel: string;
  value: string;
  accent: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
}) {
  return (
    <label className="control-label">
      {label}
      <span className={`select-shell ${accent}`}>
        <span className="select-dot" aria-hidden="true" />
        <select aria-label={ariaLabel} value={value} onChange={(event) => onChange(event.target.value)}>
          {children}
        </select>
        <svg viewBox="0 0 20 20" aria-hidden="true" className="select-chevron">
          <path d="m6 8 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
    </label>
  );
}

export function ConfigurationPanel(props: ConfigurationPanelProps) {
  return (
    <aside
      id="configuration-panel"
      className={`glass-panel configuration-panel ${props.configOpen ? "" : "config-collapsed"}`}
      aria-label="Repository configuration"
    >
      <div className="panel-heading-row">
        <div>
          <p className="eyebrow">Workspace</p>
          <h2 className="panel-title">Repository</h2>
        </div>
        <span className="panel-index">01</span>
      </div>

      <label className="control-label mt-5">
        Repository path
        <input
          type="text"
          aria-label="Repository path"
          value={props.repository}
          onChange={(event) => props.onRepositoryChange(event.target.value)}
          placeholder="D:\Masters-Projects\RealTimeCollab"
          className="field code-font repository-path-input"
          title={props.repository || "Repository path"}
          required
        />
      </label>
      <p className="helper-text">Repository must already be indexed by RepoRAG.</p>

      <div className="section-divider" />
      <p className="eyebrow mb-4">Configuration</p>
      <div className="space-y-4">
        <SelectField
          label="Retrieval"
          ariaLabel="Retrieval method"
          value={props.retrievalMethod}
          accent={methodAccent[props.retrievalMethod]}
          onChange={(value) => props.onRetrievalMethodChange(value as RetrievalMethod)}
        >
          <option value="hybrid">Hybrid</option>
          <option value="vector">Vector</option>
          <option value="bm25">BM25</option>
          <option value="reranked">Reranked</option>
        </SelectField>

        <SelectField
          label="Evidence"
          ariaLabel="Evidence scope"
          value={props.evidenceScope}
          accent={scopeAccent[props.evidenceScope]}
          onChange={(value) => props.onEvidenceScopeChange(value as EvidenceScope)}
        >
          <option value="all">All</option>
          <option value="code">Code</option>
          <option value="docs">Docs</option>
        </SelectField>

        <SelectField
          label="Provider"
          ariaLabel="Provider"
          value={props.provider}
          accent={providerAccent[props.provider]}
          onChange={(value) => props.onProviderChange(value as Provider)}
        >
          <option value="deepseek">DeepSeek</option>
          <option value="gemini">Gemini</option>
          <option value="openai">OpenAI</option>
        </SelectField>
      </div>
      <p className="helper-text">Credentials remain in the local backend environment.</p>

      <div className="section-divider" />
      <button
        type="button"
        className="advanced-toggle"
        onClick={props.onToggleAdvanced}
        aria-expanded={props.advancedOpen}
        aria-controls="advanced-settings"
      >
        <span>
          <span className="block text-[14px] font-semibold text-[var(--text-primary)]">Advanced</span>
          <span className="mt-0.5 block text-[12px] text-[var(--text-muted)]">Context and generation limits</span>
        </span>
        <svg viewBox="0 0 20 20" aria-hidden="true" className={props.advancedOpen ? "rotate-180" : ""}>
          <path d="m6 8 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      <div id="advanced-settings" className={`advanced-settings ${props.advancedOpen ? "advanced-open" : ""}`}>
        <div className="grid grid-cols-2 gap-3 pt-4 xl:grid-cols-1 2xl:grid-cols-2">
          <label className="control-label">
            Context chunks
            <input
              aria-label="Context chunks"
              type="number"
              min="1"
              value={props.contextK}
              onChange={(event) => props.onContextKChange(Number(event.target.value))}
              className="field"
            />
          </label>
          <label className="control-label">
            Output tokens
            <input
              aria-label="Output tokens"
              type="number"
              min="1"
              value={props.maxOutputTokens}
              onChange={(event) => props.onMaxOutputTokensChange(Number(event.target.value))}
              className="field"
            />
          </label>
        </div>
      </div>
    </aside>
  );
}
