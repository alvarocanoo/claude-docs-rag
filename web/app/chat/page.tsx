"use client";

import { useRef, useState } from "react";

type Citation = {
  chunk_id: number;
  source_url: string;
  section_path: string;
};

type DonePayload = {
  citations: Citation[];
  call: {
    provider: string;
    model: string;
    input_tokens: number;
    output_tokens: number;
    cost_usd: number;
  } | null;
  timings_ms: Record<string, number>;
  total_ms: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  meta?: DonePayload;
  streaming?: boolean;
  error?: string;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const SAMPLES = [
  "How does prompt caching work?",
  "How do I stream messages from the Claude API in Python?",
  "What is the maximum context window of Claude Sonnet 4.6?",
  "How do I define a tool schema for Claude to use?",
];

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function send(question: string) {
    if (!question.trim() || busy) return;
    setBusy(true);
    setInput("");
    const userMsg: Message = { role: "user", content: question };
    const placeholder: Message = { role: "assistant", content: "", streaming: true };
    setMessages((m) => [...m, userMsg, placeholder]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k: 5, max_tokens: 800 }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text.slice(0, 300)}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          if (!frame.trim()) continue;
          let eventType = "message";
          let data = "";
          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) eventType = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (!data) continue;
          try {
            const payload = JSON.parse(data);
            if (eventType === "token") {
              const text = payload.text ?? "";
              setMessages((m) => {
                const next = [...m];
                const last = next[next.length - 1];
                if (last && last.role === "assistant") {
                  next[next.length - 1] = { ...last, content: last.content + text };
                }
                return next;
              });
            } else if (eventType === "done") {
              setMessages((m) => {
                const next = [...m];
                const last = next[next.length - 1];
                if (last && last.role === "assistant") {
                  next[next.length - 1] = {
                    ...last,
                    streaming: false,
                    citations: payload.citations,
                    meta: payload,
                  };
                }
                return next;
              });
            } else if (eventType === "error") {
              throw new Error(payload.error ?? "stream error");
            }
          } catch (e) {
            console.error("SSE parse error", e, frame);
          }
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessages((m) => {
        const next = [...m];
        const last = next[next.length - 1];
        if (last && last.role === "assistant") {
          next[next.length - 1] = { ...last, streaming: false, error: msg };
        }
        return next;
      });
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      <main className="mx-auto max-w-3xl px-6 py-10">
        <header className="mb-6 flex items-baseline justify-between border-b border-zinc-200 dark:border-zinc-800 pb-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">claude-docs-rag · chat</h1>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
              Streaming RAG answer with inline <span className="font-mono">[n]</span> citations.
              Hybrid retrieval (BM25 + dense + reranker) feeds the LLM.
            </p>
          </div>
          <a
            href="/"
            className="text-xs text-zinc-500 underline hover:text-zinc-900 dark:hover:text-zinc-100"
          >
            ← back to search
          </a>
        </header>

        {messages.length === 0 && (
          <div className="mb-6">
            <p className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400 mb-2">
              Try one of these
            </p>
            <div className="flex flex-wrap gap-2">
              {SAMPLES.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => send(q)}
                  className="rounded-full border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-3 py-1 text-xs text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <ul className="space-y-4">
          {messages.map((m, i) => (
            <li
              key={i}
              className={
                m.role === "user"
                  ? "rounded-lg bg-zinc-900 dark:bg-zinc-100 text-zinc-100 dark:text-zinc-900 px-4 py-3 ml-12"
                  : "rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-3 mr-12"
              }
            >
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1">
                {m.role}
                {m.streaming && " · streaming…"}
                {m.error && " · error"}
              </div>
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {m.content}
                {m.streaming && (
                  <span className="ml-1 inline-block h-3 w-2 animate-pulse bg-current opacity-50" />
                )}
              </div>
              {m.error && (
                <div className="mt-2 text-xs text-red-600 dark:text-red-400 font-mono">
                  {m.error}
                </div>
              )}
              {m.citations && m.citations.length > 0 && (
                <div className="mt-3 border-t border-zinc-200 dark:border-zinc-800 pt-2">
                  <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1">
                    Citations
                  </div>
                  <ul className="space-y-1">
                    {m.citations.map((c) => (
                      <li key={c.chunk_id} className="text-xs">
                        <span className="font-mono text-zinc-500">[{c.chunk_id}]</span>{" "}
                        <span className="text-zinc-700 dark:text-zinc-300">{c.section_path}</span>{" "}
                        <a
                          href={c.source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-700 dark:text-blue-400 hover:underline break-all"
                        >
                          {c.source_url}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {m.meta?.call && (
                <div className="mt-2 text-[10px] font-mono text-zinc-500 dark:text-zinc-500">
                  {m.meta.call.provider}/{m.meta.call.model} · in {m.meta.call.input_tokens}/out{" "}
                  {m.meta.call.output_tokens} · ${m.meta.call.cost_usd.toFixed(5)} ·{" "}
                  {(m.meta.total_ms / 1000).toFixed(1)}s
                </div>
              )}
            </li>
          ))}
        </ul>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="mt-6 flex gap-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything about the Claude API…"
            className="flex-1 rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-3 py-2 text-sm shadow-sm focus:border-zinc-900 dark:focus:border-zinc-100 focus:outline-none"
            disabled={busy}
          />
          {busy ? (
            <button
              type="button"
              onClick={stop}
              className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white hover:opacity-90"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="rounded-md bg-zinc-900 dark:bg-zinc-100 px-4 py-2 text-sm font-medium text-white dark:text-zinc-900 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          )}
        </form>

        <footer className="mt-8 text-xs text-zinc-500 dark:text-zinc-500">
          API base: <code>{API_BASE}</code> · Endpoint: <code>POST /ask/stream</code> (SSE)
        </footer>
      </main>
    </div>
  );
}
