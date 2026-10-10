import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { QaResponse } from "../lib/api";
import { RepoRagApp } from "./reporag-app";

const healthResponse = {
  ok: true,
  status: 200,
  json: async () => ({ status: "ok", service: "RepoRAG" }),
} as Response;

const answerResponse: QaResponse = {
  question: "Where is JWT authentication enforced?",
  repository: "RealTimeCollab",
  retrieval_method: "hybrid",
  evidence_scope: "all",
  provider: "deepseek",
  model: "deepseek-flash",
  answer: [
    "Authentication is enforced by the **JWT guard** [S2].",
    "",
    "- It extracts the `Bearer` token.",
    "- It verifies the signed payload.",
    "",
    "```ts",
    "return this.jwtService.verifyAsync(token);",
    "```",
  ].join("\n"),
  citations_valid: true,
  cited_source_ids: ["S2"],
  generation_latency_seconds: 4.63,
  usage: { input_tokens: 1853, output_tokens: 676, reasoning_tokens: 198, total_tokens: 2529 },
  sources: [
    {
      id: "S1",
      file_path: "docs/authentication.md",
      symbol: null,
      parent_symbol: null,
      start_line: 71,
      end_line: 97,
      language: "markdown",
      chunk_type: "file",
      retrieval_rank: 1,
      content: "# Authentication\nProtected routes require a bearer token.",
    },
    {
      id: "S2",
      file_path: "backend/src/auth/guards/jwt-auth.guard.ts",
      symbol: "canActivate",
      parent_symbol: "JwtAuthGuard",
      start_line: 19,
      end_line: 42,
      language: "typescript",
      chunk_type: "method",
      retrieval_rank: 2,
      content: "canActivate(context: ExecutionContext) {\n  return verifyJwt(context);\n}",
    },
  ],
};

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return { ok, status, json: async () => body } as Response;
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    callback(0);
    return 1;
  });
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    configurable: true,
    value: vi.fn(),
    writable: true,
  });
});

function renderWithHealth() {
  fetchMock.mockResolvedValueOnce(healthResponse);
  return render(<RepoRagApp />);
}

function fillRequiredFields() {
  fireEvent.change(screen.getByLabelText("Repository path"), {
    target: { value: "D:\\Masters-Projects\\RealTimeCollab" },
  });
  fireEvent.change(screen.getByLabelText("Repository question"), {
    target: { value: "Where is JWT authentication enforced?" },
  });
}

async function submitSuccessful(response: QaResponse = answerResponse) {
  fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(response));
  render(<RepoRagApp />);
  fillRequiredFields();
  fireEvent.click(screen.getByRole("button", { name: /Ask RepoRAG/ }));
  expect(await screen.findByRole("heading", { name: "Answer" })).toBeInTheDocument();
}

describe("RepoRAG v2 workbench", () => {
  it("renders the RepoRAG repository-graph identity", () => {
    renderWithHealth();
    expect(screen.getByRole("heading", { name: "RepoRAG" })).toBeInTheDocument();
    expect(document.querySelector(".brand-mark svg")).toBeInTheDocument();
    expect(screen.getByText("Repository intelligence, grounded in source")).toBeInTheDocument();
  });

  it("presents the grounded-analysis capabilities in the empty state", () => {
    renderWithHealth();
    const capabilities = screen.getByLabelText("RepoRAG capabilities");
    expect(capabilities).toHaveTextContent("Trace implementation");
    expect(capabilities).toHaveTextContent("Validate citations");
    expect(capabilities).toHaveTextContent("Inspect exact source");
  });

  it("renders a pointer-responsive repository graph with a stable reset", () => {
    renderWithHealth();
    const graph = screen.getByTestId("repository-graph");
    vi.spyOn(graph, "getBoundingClientRect").mockReturnValue({
      width: 200,
      height: 100,
      top: 0,
      left: 0,
      right: 200,
      bottom: 100,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
    fireEvent.pointerMove(graph, { clientX: 180, clientY: 20 });
    expect(graph.style.getPropertyValue("--graph-rotate-x")).not.toBe("0deg");
    expect(graph.style.getPropertyValue("--graph-rotate-y")).not.toBe("0deg");
    fireEvent.pointerLeave(graph);
    expect(graph.style.getPropertyValue("--graph-rotate-x")).toBe("0deg");
    expect(graph.style.getPropertyValue("--graph-rotate-y")).toBe("0deg");
  });

  it("shows a successful initial API status", async () => {
    renderWithHealth();
    expect(await screen.findByText("API Connected")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("shows the offline state and retries health safely", async () => {
    const user = userEvent.setup();
    fetchMock.mockRejectedValueOnce(new TypeError("offline")).mockResolvedValueOnce(healthResponse);
    render(<RepoRagApp />);
    expect(await screen.findByText("API Offline")).toBeInTheDocument();
    expect(screen.getByText("RepoRAG API is offline.")).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "Retry" })[0]);
    expect(await screen.findByText("API Connected")).toBeInTheDocument();
  });

  it("renders an accessible repository input and indexed-repository guidance", () => {
    renderWithHealth();
    expect(screen.getByLabelText("Repository path")).toBeRequired();
    expect(screen.getByText("Repository must already be indexed by RepoRAG.")).toBeInTheDocument();
  });

  it("uses unchanged retrieval, evidence, and provider defaults", () => {
    renderWithHealth();
    expect(screen.getByLabelText("Retrieval method")).toHaveValue("hybrid");
    expect(screen.getByLabelText("Evidence scope")).toHaveValue("all");
    expect(screen.getByLabelText("Provider")).toHaveValue("deepseek");
  });

  it("keeps Advanced collapsed initially", () => {
    renderWithHealth();
    expect(screen.getByRole("button", { name: /Advanced/ })).toHaveAttribute("aria-expanded", "false");
    expect(document.querySelector("#advanced-settings")).not.toHaveClass("advanced-open");
  });

  it("opens Advanced and preserves numeric defaults", async () => {
    const user = userEvent.setup();
    renderWithHealth();
    await user.click(screen.getByRole("button", { name: /Advanced/ }));
    expect(screen.getByRole("button", { name: /Advanced/ })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByLabelText("Context chunks")).toHaveValue(5);
    expect(screen.getByLabelText("Output tokens")).toHaveValue(2000);
  });

  it("does not submit an empty question", () => {
    renderWithHealth();
    fireEvent.change(screen.getByLabelText("Repository path"), { target: { value: "D:\\Repo" } });
    const submit = screen.getByRole("button", { name: /Ask RepoRAG/ });
    expect(submit).toBeDisabled();
    fireEvent.click(submit);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("populates the composer from an example prompt without submitting", () => {
    renderWithHealth();
    fireEvent.click(screen.getByRole("button", { name: "How does retry behavior work?" }));
    expect(screen.getByLabelText("Repository question")).toHaveValue("How does retry behavior work?");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("posts exactly the supported request fields", async () => {
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    fillRequiredFields();
    fireEvent.change(screen.getByLabelText("Evidence scope"), { target: { value: "code" } });
    fireEvent.click(screen.getByRole("button", { name: /Ask RepoRAG/ }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const [, options] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(JSON.parse(options.body as string)).toEqual({
      repository: "D:\\Masters-Projects\\RealTimeCollab",
      question: "Where is JWT authentication enforced?",
      retrieval_method: "hybrid",
      evidence_scope: "code",
      provider: "deepseek",
      context_k: 5,
      max_output_tokens: 2000,
    });
  });

  it("propagates provider, scope, retrieval, and Advanced overrides", async () => {
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    fillRequiredFields();
    fireEvent.change(screen.getByLabelText("Retrieval method"), { target: { value: "vector" } });
    fireEvent.change(screen.getByLabelText("Evidence scope"), { target: { value: "docs" } });
    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "gemini" } });
    fireEvent.change(screen.getByLabelText("Context chunks"), { target: { value: "3" } });
    fireEvent.change(screen.getByLabelText("Output tokens"), { target: { value: "1500" } });
    fireEvent.click(screen.getByRole("button", { name: /Ask RepoRAG/ }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const payload = JSON.parse((fetchMock.mock.calls[1][1] as RequestInit).body as string);
    expect(payload).toMatchObject({ retrieval_method: "vector", evidence_scope: "docs", provider: "gemini", context_k: 3, max_output_tokens: 1500 });
  });

  it("submits with Ctrl+Enter", async () => {
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    fillRequiredFields();
    fireEvent.keyDown(screen.getByLabelText("Repository question"), { key: "Enter", ctrlKey: true });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("submits with Cmd+Enter", async () => {
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    fillRequiredFields();
    fireEvent.keyDown(screen.getByLabelText("Repository question"), { key: "Enter", metaKey: true });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("shows loading state and disables duplicate submission", () => {
    fetchMock.mockResolvedValueOnce(healthResponse).mockImplementationOnce(() => new Promise(() => {}));
    render(<RepoRagApp />);
    fillRequiredFields();
    fireEvent.click(screen.getByRole("button", { name: /Ask RepoRAG/ }));
    expect(screen.getByText("Retrieving repository evidence…")).toBeInTheDocument();
    expect(screen.getByLabelText("Retrieving evidence")).toBeInTheDocument();
    expect(document.querySelectorAll(".skeleton-source")).toHaveLength(5);
    expect(screen.getByRole("button", { name: /Working/ })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /Working/ }));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("renders Markdown strong, inline code, list, and fenced code safely", async () => {
    await submitSuccessful();
    expect(screen.getByText("JWT guard").tagName).toBe("STRONG");
    expect(screen.getByText("Bearer").tagName).toBe("CODE");
    expect(screen.getByText("It extracts the", { exact: false }).closest("li")).toBeInTheDocument();
    const block = screen.getByText(/return this\.jwtService/);
    expect(block.closest("pre")).toBeInTheDocument();
  });

  it("does not execute raw HTML from model output", async () => {
    const unsafe = { ...answerResponse, answer: '<img src="x" onerror="window.hacked=true">Safe answer [S1].' };
    await submitSuccessful(unsafe);
    expect(document.querySelector(".answer-markdown img")).not.toBeInTheDocument();
    expect((window as Window & { hacked?: boolean }).hacked).toBeUndefined();
  });

  it("renders citation markers as keyboard-accessible controls", async () => {
    await submitSuccessful();
    expect(screen.getByRole("button", { name: "Show source S2" })).toHaveTextContent("S2");
  });

  it("selects the cited source and reveals its exact viewer", async () => {
    const user = userEvent.setup();
    await submitSuccessful();
    await user.click(screen.getByRole("button", { name: "Show source S2" }));
    const option = screen.getByRole("option", { name: /jwt-auth\.guard\.ts/ });
    expect(option).toHaveAttribute("aria-selected", "true");
    expect(option).toHaveClass("source-citation-target");
    expect(screen.getByRole("region", { name: "S2 source code" })).toHaveTextContent("verifyJwt");
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });

  it("selects the first ranked source by default", async () => {
    await submitSuccessful();
    expect(screen.getByRole("option", { name: /docs\/authentication\.md/ })).toHaveAttribute("aria-selected", "true");
  });

  it("allows direct evidence-row source selection", async () => {
    const user = userEvent.setup();
    await submitSuccessful();
    await user.click(screen.getByRole("option", { name: /jwt-auth\.guard\.ts/ }));
    expect(screen.getByRole("region", { name: "S2 source code" })).toBeInTheDocument();
  });

  it("renders full source metadata and line-numbered exact content", async () => {
    const user = userEvent.setup();
    await submitSuccessful();
    await user.click(screen.getByRole("option", { name: /jwt-auth\.guard\.ts/ }));
    const viewer = screen.getByRole("region", { name: "S2 source code" });
    expect(screen.getAllByText("backend/src/auth/guards/jwt-auth.guard.ts").length).toBeGreaterThan(0);
    expect(screen.getAllByText("JwtAuthGuard.canActivate").length).toBeGreaterThan(0);
    expect(screen.getByText(/Lines 19–42 · TypeScript · method · Rank 2/)).toBeInTheDocument();
    expect(within(viewer).getByText("19")).toBeInTheDocument();
    expect(viewer).toHaveTextContent("return verifyJwt(context)");
  });

  it("renders separate metadata chips including verification, latency, and tokens", async () => {
    await submitSuccessful();
    const metadata = screen.getByLabelText("Answer metadata");
    expect(metadata).toHaveTextContent("DeepSeek · deepseek-flash");
    expect(metadata).toHaveTextContent("Hybrid");
    expect(metadata).toHaveTextContent("All evidence");
    expect(metadata).toHaveTextContent("2 sources");
    expect(metadata).toHaveTextContent("Citations verified");
    expect(metadata.querySelector(".validation-check")).toBeInTheDocument();
    expect(metadata).toHaveTextContent("4.63s");
    expect(metadata).toHaveTextContent("2,529 tokens");
  });

  it("does not hide failed citation validation", async () => {
    await submitSuccessful({ ...answerResponse, citations_valid: false });
    expect(screen.getByText("Citation validation failed")).toBeInTheDocument();
  });

  it("renders safe API errors with recovery guidance", async () => {
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse({
      detail: "Compatible RepoRAG runtime artifacts are missing; index the repository first.",
      code: "repository_not_indexed",
    }, false, 409));
    render(<RepoRagApp />);
    fillRequiredFields();
    fireEvent.click(screen.getByRole("button", { name: /Ask RepoRAG/ }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("index the repository first");
    expect(alert).toHaveTextContent("Index this repository with RepoRAG");
  });

  it("contains no API key or secret input", () => {
    renderWithHealth();
    expect(screen.queryByLabelText(/API key/i)).not.toBeInTheDocument();
    expect(document.querySelector('input[type="password"]')).not.toBeInTheDocument();
  });

  it("shows the active repository name in the header", () => {
    renderWithHealth();
    fireEvent.change(screen.getByLabelText("Repository path"), { target: { value: "D:\\Masters-Projects\\RealTimeCollab" } });
    expect(screen.getByText("RealTimeCollab")).toBeInTheDocument();
  });

  it("toggles the medium-layout configuration control accessibly", () => {
    renderWithHealth();
    const button = screen.getByRole("button", { name: "Configuration" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
  });

  it("opens and closes the workspace command palette with the keyboard", async () => {
    renderWithHealth();
    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    const palette = screen.getByRole("dialog", { name: "Workspace commands" });
    expect(palette).toBeInTheDocument();
    expect(screen.getByLabelText("Search commands")).toHaveFocus();
    fireEvent.keyDown(screen.getByLabelText("Search commands"), { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Workspace commands" })).not.toBeInTheDocument();
  });

  it("enters focus mode from the command palette and exits from the header", async () => {
    const user = userEvent.setup();
    renderWithHealth();
    await user.click(screen.getByRole("button", { name: "Open command palette" }));
    await user.click(screen.getByRole("option", { name: /Enter focus mode/ }));
    expect(document.querySelector(".app-shell")).toHaveClass("focus-mode");
    const exit = screen.getByRole("button", { name: "Exit focus mode" });
    expect(exit).toHaveAttribute("aria-pressed", "true");
    await user.click(exit);
    expect(document.querySelector(".app-shell")).not.toHaveClass("focus-mode");
  });

  it("applies evidence scope commands without submitting", async () => {
    const user = userEvent.setup();
    renderWithHealth();
    await user.click(screen.getByRole("button", { name: "Open command palette" }));
    await user.click(screen.getByRole("option", { name: /Use code-only evidence/ }));
    expect(screen.getByLabelText("Evidence scope")).toHaveValue("code");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("copies the exact generated answer", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    await submitSuccessful();
    fireEvent.click(screen.getByRole("button", { name: "Copy answer" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(answerResponse.answer));
    expect(screen.getByRole("button", { name: "Copied" })).toBeInTheDocument();
  });

  it("labels only genuinely cited evidence and copies selected source content", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    await submitSuccessful();
    const citedOption = screen.getByRole("option", { name: /jwt-auth\.guard\.ts/ });
    const uncitedOption = screen.getByRole("option", { name: /docs\/authentication\.md/ });
    expect(citedOption).toHaveTextContent("Cited");
    expect(uncitedOption).not.toHaveTextContent("Cited");
    fireEvent.click(citedOption);
    fireEvent.click(screen.getByRole("button", { name: "Copy source" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(answerResponse.sources[1].content));
  });
});
