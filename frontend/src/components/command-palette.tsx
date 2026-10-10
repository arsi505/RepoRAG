import { useEffect, useMemo, useRef, useState } from "react";

import type { EvidenceScope, RetrievalMethod } from "../lib/api";

interface CommandPaletteProps {
  open: boolean;
  focusMode: boolean;
  onClose: () => void;
  onFocusRepository: () => void;
  onFocusQuestion: () => void;
  onToggleFocusMode: () => void;
  onSetRetrievalMethod: (method: RetrievalMethod) => void;
  onSetEvidenceScope: (scope: EvidenceScope) => void;
  onOpenAdvanced: () => void;
}

interface PaletteCommand {
  id: string;
  label: string;
  description: string;
  shortcut?: string;
  action: () => void;
}

export function CommandPalette(props: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const commands = useMemo<PaletteCommand[]>(() => [
    { id: "question", label: "Focus repository question", description: "Move directly to the question composer", shortcut: "Q", action: props.onFocusQuestion },
    { id: "repository", label: "Focus repository path", description: "Choose the indexed repository workspace", shortcut: "R", action: props.onFocusRepository },
    { id: "focus", label: props.focusMode ? "Exit focus mode" : "Enter focus mode", description: "Prioritize the answer and evidence workspace", shortcut: "F", action: props.onToggleFocusMode },
    { id: "hybrid", label: "Use Hybrid retrieval", description: "Select the Hybrid retrieval method", action: () => props.onSetRetrievalMethod("hybrid") },
    { id: "code", label: "Use code-only evidence", description: "Restrict generation context to implementation evidence", action: () => props.onSetEvidenceScope("code") },
    { id: "all", label: "Use all evidence", description: "Allow code and documentation evidence", action: () => props.onSetEvidenceScope("all") },
    { id: "advanced", label: "Open advanced settings", description: "Review context and output-token limits", shortcut: "G", action: props.onOpenAdvanced },
  ], [props]);

  const filtered = commands.filter((command) => `${command.label} ${command.description}`.toLowerCase().includes(query.toLowerCase()));

  useEffect(() => {
    if (!props.open) return;
    window.requestAnimationFrame(() => inputRef.current?.focus());
  }, [props.open]);

  if (!props.open) return null;

  const safeActiveIndex = Math.min(activeIndex, Math.max(0, filtered.length - 1));

  const close = () => {
    setQuery("");
    setActiveIndex(0);
    props.onClose();
  };

  const run = (command: PaletteCommand) => {
    command.action();
    close();
  };

  return (
    <div className="command-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
      <section className="command-palette" role="dialog" aria-modal="true" aria-labelledby="command-heading">
        <div className="command-search">
          <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.25" /><path d="m12.5 12.5 4 4" /></svg>
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => { setQuery(event.target.value); setActiveIndex(0); }}
            onKeyDown={(event) => {
              if (event.key === "Escape") close();
              if (event.key === "ArrowDown") { event.preventDefault(); setActiveIndex((index) => filtered.length ? (index + 1) % filtered.length : 0); }
              if (event.key === "ArrowUp") { event.preventDefault(); setActiveIndex((index) => filtered.length ? (index - 1 + filtered.length) % filtered.length : 0); }
              if (event.key === "Enter" && filtered[safeActiveIndex]) { event.preventDefault(); run(filtered[safeActiveIndex]); }
            }}
            placeholder="Search actions…"
            aria-label="Search commands"
          />
          <kbd>Esc</kbd>
        </div>
        <div className="command-header">
          <h2 id="command-heading">Workspace commands</h2>
          <span>{filtered.length} actions</span>
        </div>
        <div className="command-list" role="listbox" aria-label="Available commands">
          {filtered.length ? filtered.map((command, index) => (
            <button
              key={command.id}
              type="button"
              role="option"
              aria-selected={index === safeActiveIndex}
              className={`command-item ${index === safeActiveIndex ? "command-active" : ""}`}
              onMouseEnter={() => setActiveIndex(index)}
              onClick={() => run(command)}
            >
              <span className="command-item-mark" aria-hidden="true">›</span>
              <span className="command-item-copy"><strong>{command.label}</strong><small>{command.description}</small></span>
              {command.shortcut && <kbd>{command.shortcut}</kbd>}
            </button>
          )) : <p className="command-empty">No matching commands</p>}
        </div>
        <footer className="command-footer"><span>↑↓ Navigate</span><span>↵ Run</span><span>Esc Close</span></footer>
      </section>
    </div>
  );
}
