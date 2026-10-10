import { RepoRagLogo } from "./reporag-logo";

type HealthState = "checking" | "connected" | "offline";

interface AppHeaderProps {
  health: HealthState;
  repository: string;
  configOpen: boolean;
  focusMode: boolean;
  onRetry: () => void;
  onToggleConfig: () => void;
  onToggleFocusMode: () => void;
  onOpenCommands: () => void;
}

function repositoryName(repository: string): string {
  const clean = repository.trim().replace(/[\\/]+$/, "");
  return clean.split(/[\\/]/).filter(Boolean).at(-1) ?? "";
}

export function AppHeader({
  health,
  repository,
  configOpen,
  focusMode,
  onRetry,
  onToggleConfig,
  onToggleFocusMode,
  onOpenCommands,
}: AppHeaderProps) {
  const activeRepository = repositoryName(repository);
  const connected = health === "connected";

  return (
    <header className="app-header">
      <div className="flex min-w-0 items-center gap-3">
        <div className="brand-mark">
          <RepoRagLogo />
        </div>
        <div className="brand-copy min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="brand-name">
            RepoRAG
            </h1>
            <span className="brand-version">v1</span>
          </div>
          <p className="brand-tagline">Repository intelligence, grounded in source</p>
        </div>
      </div>

      <div className="hidden min-w-0 flex-1 justify-center px-6 sm:flex">
        {activeRepository && (
          <div className="repository-context" title={repository}>
            <span className="status-dot bg-[var(--blue-bright)]" aria-hidden="true" />
            <span className="repository-label">Active repository</span>
            <span className="repository-separator" aria-hidden="true" />
            <span className="code-font truncate text-[var(--text-primary)]">{activeRepository}</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        <button type="button" className="header-tool command-trigger" onClick={onOpenCommands} aria-label="Open command palette">
          <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.1" /><path d="m12.4 12.4 3.7 3.7" /></svg>
          <span>Commands</span><kbd>⌘K</kbd>
        </button>
        <button
          type="button"
          className={`header-tool focus-trigger ${focusMode ? "tool-active" : ""}`}
          onClick={onToggleFocusMode}
          aria-label={focusMode ? "Exit focus mode" : "Enter focus mode"}
          aria-pressed={focusMode}
          title={focusMode ? "Exit focus mode" : "Enter focus mode"}
        >
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 3H3v4M13 3h4v4M7 17H3v-4M13 17h4v-4" /></svg>
        </button>
        <button
          type="button"
          className="secondary-button config-toggle"
          onClick={onToggleConfig}
          aria-expanded={configOpen}
          aria-controls="configuration-panel"
        >
          Configuration
        </button>
        <div className="flex items-center gap-2">
          <span
            className={`status-pill ${connected ? "status-connected" : health === "offline" ? "status-offline" : "status-checking"}`}
            aria-live="polite"
          >
            <span className="status-dot" aria-hidden="true" />
            API {connected ? "Connected" : health === "checking" ? "Checking" : "Offline"}
          </span>
          {health === "offline" && (
            <button type="button" onClick={onRetry} className="secondary-button px-2.5 py-1.5">
              Retry
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

export type { HealthState };
