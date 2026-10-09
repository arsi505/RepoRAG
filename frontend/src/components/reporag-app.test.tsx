import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { RepoRagApp } from "./reporag-app";
import type { QaResponse } from "../lib/api";

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
  answer: "Authentication is enforced by the JWT guard [S1].",
  citations_valid: true,
  cited_source_ids: ["S1"],
  generation_latency_seconds: 4.63,
  usage: {
    input_tokens: 1853,
    output_tokens: 676,
    reasoning_tokens: 198,
    total_tokens: 2529,
  },
  sources: [
    {
      id: "S1",
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
});

function renderWithHealth() {
  fetchMock.mockResolvedValueOnce(healthResponse);
  return render(<RepoRagApp />);
}

async function fillRequiredFields(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Repository path"), "D:\\Masters-Projects\\RealTimeCollab");
  await user.type(screen.getByLabelText("Repository question"), "Where is JWT authentication enforced?");
}

describe("RepoRAG interface", () => {
  it("shows a successful initial health status", async () => {
    renderWithHealth();
    expect(await screen.findByText("API Connected")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("shows offline health status and a retry action on failure", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("offline"));
    render(<RepoRagApp />);
    expect(await screen.findByText("API Offline")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("uses the backend Q&A defaults", async () => {
    renderWithHealth();
    expect(screen.getByLabelText("Retrieval method")).toHaveValue("hybrid");
    expect(screen.getByLabelText("Evidence scope")).toHaveValue("all");
    expect(screen.getByLabelText("Provider")).toHaveValue("deepseek");
    expect(screen.getByLabelText("Context chunks")).toHaveValue(5);
    expect(screen.getByLabelText("Output tokens")).toHaveValue(2000);
  });

  it("does not submit an empty question", async () => {
    const user = userEvent.setup();
    renderWithHealth();
    await user.type(screen.getByLabelText("Repository path"), "D:\\Repo");
    const submit = screen.getByRole("button", { name: "Ask RepoRAG" });
    expect(submit).toBeDisabled();
    await user.click(submit);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("posts exactly the supported request fields", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    await fillRequiredFields(user);
    await user.selectOptions(screen.getByLabelText("Evidence scope"), "code");
    await user.click(screen.getByRole("button", { name: "Ask RepoRAG" }));

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

  it("shows a loading state while the completed response is pending", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(healthResponse).mockImplementationOnce(() => new Promise(() => {}));
    render(<RepoRagApp />);
    await fillRequiredFields(user);
    await user.click(screen.getByRole("button", { name: "Ask RepoRAG" }));
    expect(screen.getByText("Searching repository…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Working…" })).toBeDisabled();
  });

  it("renders a successful grounded answer without changing its text", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    await fillRequiredFields(user);
    await user.click(screen.getByRole("button", { name: "Ask RepoRAG" }));
    expect(await screen.findByRole("heading", { name: "Answer" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show source S1" })).toHaveTextContent("[S1]");
    expect(screen.getByText(/Authentication is enforced by the JWT guard/)).toBeInTheDocument();
  });

  it("renders complete source metadata", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    await fillRequiredFields(user);
    await user.click(screen.getByRole("button", { name: "Ask RepoRAG" }));
    expect(await screen.findByText("backend/src/auth/guards/jwt-auth.guard.ts")).toBeInTheDocument();
    expect(screen.getByText("JwtAuthGuard.canActivate")).toBeInTheDocument();
    expect(screen.getByText("Lines 19–42")).toBeInTheDocument();
    expect(screen.getByText("Typescript · method")).toBeInTheDocument();
    expect(screen.getByText("Rank 2")).toBeInTheDocument();
  });

  it("expands exact source evidence on demand", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    await fillRequiredFields(user);
    await user.click(screen.getByRole("button", { name: "Ask RepoRAG" }));
    const toggle = await screen.findByRole("button", { name: "View evidence" });
    expect(screen.queryByText(/verifyJwt/)).not.toBeInTheDocument();
    await user.click(toggle);
    expect(screen.getByText(/verifyJwt/)).toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-expanded", "true");
  });

  it("does not hide failed citation validation", async () => {
    const user = userEvent.setup();
    const invalid = { ...answerResponse, citations_valid: false };
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(invalid));
    render(<RepoRagApp />);
    await fillRequiredFields(user);
    await user.click(screen.getByRole("button", { name: "Ask RepoRAG" }));
    expect(await screen.findByText("Citation validation failed")).toBeInTheDocument();
  });

  it("shows returned token and latency metadata", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    await fillRequiredFields(user);
    await user.click(screen.getByRole("button", { name: "Ask RepoRAG" }));
    expect(await screen.findByText("4.63s")).toBeInTheDocument();
    expect(screen.getByText("2,529 tokens")).toBeInTheDocument();
  });

  it("renders safe API errors with relevant recovery guidance", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(
      jsonResponse(
        {
          detail: "Compatible RepoRAG runtime artifacts are missing; index the repository first.",
          code: "repository_not_indexed",
        },
        false,
        409,
      ),
    );
    render(<RepoRagApp />);
    await fillRequiredFields(user);
    await user.click(screen.getByRole("button", { name: "Ask RepoRAG" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("index the repository first");
    expect(screen.getByRole("alert")).toHaveTextContent("Index this repository with RepoRAG");
  });

  it("contains no API key or secret input", () => {
    renderWithHealth();
    expect(screen.queryByLabelText(/API key/i)).not.toBeInTheDocument();
    expect(document.querySelector('input[type="password"]')).not.toBeInTheDocument();
  });

  it("submits with Ctrl+Enter from the question field", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(jsonResponse(answerResponse));
    render(<RepoRagApp />);
    await fillRequiredFields(user);
    fireEvent.keyDown(screen.getByLabelText("Repository question"), {
      key: "Enter",
      ctrlKey: true,
    });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });
});
