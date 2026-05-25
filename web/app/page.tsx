"use client";

import { useState } from "react";

type Hit = {
  source_url: string;
  title: string;
  section_path: string;
  rerank_score: number;
  fusion_score: number;
  excerpt: string;
};

type SearchResponse = {
  query: string;
  hits: Hit[];
  latency_ms: number;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const SAMPLE_QUERIES = [
  "How do I stream messages from the Claude API?",
  "How does prompt caching work?",
  "What is the bash tool?",
  "How do I define a tool schema?",
  "What HTTP status codes does the API return on errors?",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setHits(null);
    setLatency(null);
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, k }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }
      const data: SearchResponse = await res.json();
      setHits(data.hits);
      setLatency(data.latency_ms);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      <main className="mx-auto max-w-4xl px-6 py-10">
        <header className="mb-8 border-b border-zinc-200 dark:border-zinc-800 pb-6">
          <h1 className="text-2xl font-semibold tracking-tight">
            claude-docs-rag
          </h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Hybrid retrieval over the Anthropic Claude API docs — BM25 + dense
            (bge-small) fused with RRF, reranked by a cross-encoder.
          </p>
        </header>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            runSearch(query);
          }}
          className="flex flex-col gap-3 sm:flex-row sm:items-end"
        >
          <div className="flex-1">
            <label
              htmlFor="q"
              className="mb-1 block text-xs font-medium text-zinc-500 dark:text-zinc-400"
            >
              Question
            </label>
            <input
              id="q"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="How do I stream messages from the Claude API?"
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-3 py-2 text-sm shadow-sm focus:border-zinc-900 dark:focus:border-zinc-100 focus:outline-none"
            />
          </div>
          <div>
            <label
              htmlFor="k"
              className="mb-1 block text-xs font-medium text-zinc-500 dark:text-zinc-400"
            >
              Top-K
            </label>
            <input
              id="k"
              type="number"
              min={1}
              max={20}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-20 rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-3 py-2 text-sm shadow-sm focus:outline-none"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded-md bg-zinc-900 dark:bg-zinc-100 px-4 py-2 text-sm font-medium text-white dark:text-zinc-900 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </form>

        <div className="mt-3 flex flex-wrap gap-2">
          {SAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => {
                setQuery(q);
                runSearch(q);
              }}
              className="rounded-full border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-3 py-1 text-xs text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800"
            >
              {q}
            </button>
          ))}
        </div>

        {error && (
          <div className="mt-6 rounded-md border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950 p-3 text-sm text-red-700 dark:text-red-300">
            <strong>Error:</strong> {error}
            <div className="mt-1 text-xs">
              Is the API server running at <code>{API_BASE}</code>? Try{" "}
              <code>uv run cdrag serve --port 8000</code>.
            </div>
          </div>
        )}

        {latency !== null && hits && (
          <div className="mt-6 text-xs text-zinc-500 dark:text-zinc-400">
            {hits.length} result{hits.length === 1 ? "" : "s"} in{" "}
            <span className="font-mono">{latency.toFixed(0)} ms</span>
          </div>
        )}

        <ul className="mt-3 space-y-3">
          {hits?.map((h, i) => (
            <li
              key={`${h.source_url}-${i}`}
              className="rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-4"
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                  rerank {h.rerank_score.toFixed(3)} · fusion{" "}
                  {h.fusion_score.toFixed(3)}
                </span>
                <span className="text-xs text-zinc-400">#{i + 1}</span>
              </div>
              <div className="mt-1 text-sm font-medium text-zinc-900 dark:text-zinc-100">
                {h.section_path || h.title}
              </div>
              <a
                href={h.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-1 block break-all text-xs text-blue-700 dark:text-blue-400 hover:underline"
              >
                {h.source_url}
              </a>
              <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-line">
                {h.excerpt}
              </p>
            </li>
          ))}
        </ul>

        <footer className="mt-12 text-xs text-zinc-500 dark:text-zinc-500">
          API base: <code>{API_BASE}</code> · Override with{" "}
          <code>NEXT_PUBLIC_API_BASE_URL</code>.
        </footer>
      </main>
    </div>
  );
}
