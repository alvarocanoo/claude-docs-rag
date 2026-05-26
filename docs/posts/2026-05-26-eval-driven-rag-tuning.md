---
title: "Eval-driven RAG tuning: how three ADRs tripled my citation accuracy on a free-tier LLM"
date: 2026-05-26
status: draft
canonical: https://github.com/alvarocanoo/claude-docs-rag/blob/main/docs/posts/2026-05-26-eval-driven-rag-tuning.md
tags: [rag, llm, ai-engineering, evaluation, hybrid-retrieval]
---

# Eval-driven RAG tuning: how three ADRs tripled my citation accuracy on a free-tier LLM

> **TL;DR.** Built a production-grade RAG over the Anthropic Claude API docs running entirely on free-tier Groq Llama 3.1 8B. Started with `citation_match = 0.188`. Three documented ADRs later it sits at `0.594` — a 3.16× lift — at flat cost and 14 % less latency. The number wasn't what mattered. *How I knew it improved* was.

This post is a write-up of three Architecture Decision Records ([ADR-010, ADR-011, ADR-012](https://github.com/alvarocanoo/claude-docs-rag/blob/main/docs/DECISIONS.md)) shipped in a single day on [`claude-docs-rag`](https://github.com/alvarocanoo/claude-docs-rag), a hybrid-retrieval RAG agent answering technical questions against `platform.claude.com/docs`. The system is deployed live (Hugging Face Spaces backend + Vercel frontend) and runs at $0 on free tiers end-to-end.

If you build LLM apps and your "eval" today is a Jupyter notebook with three cherry-picked queries — this is the post I wish I'd read.

---

## Why eval-driven, not vibes-driven

The default mode in LLM-app development is *vibes-driven*: paste in a prompt, eyeball the output, declare victory if it "looks good". This works exactly long enough for the third feature change to silently destroy the second one, with no signal until a user reports it.

Eval-driven is the opposite: write a tiny golden dataset of question/expected-answer pairs *first*, define numeric metrics, automate the loop, gate every change against the previous numbers. The cost is one weekend of plumbing. The payoff is that **every change after that comes with a number**.

In this project that meant:

- A golden dataset of 32 Q&A across 14 doc categories, each annotated with `expected_topics` (substrings the answer should contain) and `expected_url_patterns` (URL fragments the citations should hit).
- Three metrics: `avg_topic_coverage`, `avg_citation_match`, `citation_rate`.
- A `cdrag eval` CLI that runs the whole RAG end-to-end on each row and writes `evals/latest.json`.
- A `scripts/check_regression.py` gate that compares latest vs `evals/baseline.json` on the **same subset of question ids** (apples-to-apples) with a 2 % tolerance.
- The gate runs in GitHub Actions on every PR. Merge blocks on regression.

This is the substrate. With it, every change below is an experiment, not a hope.

---

## ADR-010 — Pluggable LLM provider (or: how to baseline at $0)

The original agent was wired against the official Anthropic SDK. Beautiful code, type-safe blocks, prompt caching — and a hard dependency on a paid LLM. The portfolio constraint was clear: the demo had to run at $0 for anyone cloning the repo.

The minimal-effort fix: a thin provider abstraction in `agent/client.py`. `LLM_PROVIDER` env var selects `anthropic` or `groq`. Two backends — `_create_message_anthropic` (one-shot + streaming), `_create_message_groq` (OpenAI-compatible). The `CallResult` dataclass gains a `provider: str` field so the eval report records *who answered*. The `/ask` endpoint checks `settings.is_llm_configured` instead of `anthropic_api_key`.

```python
class CallResult:
    text: str
    provider: str      # NEW — eval baseline records the backend
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    stop_reason: str | None
    cost_usd: float
```

Two pricing dicts in code, one per provider. The Groq paid-tier prices are applied to the eval numbers even when no money actually changes hands — so the cost line is *what it would cost*, not just "$0 because free tier".

**Outcome.** First committed `evals/baseline.json`: 32 Q&A, Groq Llama 3.1 8B Instant, free tier.

| Metric                  | Target  | Initial baseline | Verdict             |
|-------------------------|---------|------------------|---------------------|
| `avg_topic_coverage`    | ≥ 0.85  | 0.526            | Under target        |
| `avg_citation_match`    | ≥ 0.90  | **0.188**        | Way under target    |
| `citation_rate`         | ≥ 0.95  | 0.312            | Way under target    |
| `avg_latency_seconds`   | —       | 27.97 s          | Mostly retry-burn   |
| `avg_cost_usd / query`  | ≤ $0.005| $0.00014         | 35× under budget ✅ |

`citation_match = 0.188` says: less than one in five answers cited the right URL. That's the kind of number you need to *see* to know your retrieval is failing. Eyeballing five queries would have completely missed this.

---

## ADR-011 — HyDE on the dense leg

Manual inspection of `baseline.json` showed a clear failure mode: for *conceptual* questions ("how do I X?", "what is the maximum Y?"), the hybrid pipeline returned `/docs/en/api/terraform/...` and `/docs/en/api/{lang}/messages/batches/...` instead of `/docs/en/build-with-claude/...`. The dense embedding of "how do I define a tool schema?" landed closer to API-reference pages (which contain dense keyword overlap with *schema*) than to the conceptual "define tools" page.

This is exactly the case [HyDE](https://arxiv.org/abs/2212.10496) is for. The intuition: the user's question and a documentation passage live in different distributions in embedding space. Asking the LLM to *first answer* the question (even imperfectly) produces text that looks like a doc passage — and matches the corpus much better.

Implementation, 65 lines including the system prompt and the defensive plumbing:

```python
# retrieval/hyde.py
async def _hyde_expand(query: str) -> str:
    if not settings.hyde_enabled or not settings.is_llm_configured:
        return query
    try:
        result = await asyncio.wait_for(
            create_message(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
                max_tokens=200,
                temperature=0.0,
            ),
            timeout=settings.hyde_timeout_seconds,
        )
        passage = (result.text or "").strip()
        return f"{query}\n\n{passage}" if passage else query
    except Exception:
        return query  # never let HyDE break retrieval
```

Three defensive guards (env switch, LLM credential check, timeout-and-catch) so a HyDE failure can never break retrieval. BM25 still runs on the raw query — BM25 is keyword-driven and benefits from the verbatim user terms. Only the dense leg sees the expanded text.

**Ablation, same 32 Q&A:**

| Metric                  | Pre-HyDE | **+ HyDE**  | Delta            |
|-------------------------|----------|-------------|------------------|
| `avg_topic_coverage`    | 0.526    | **0.563**   | **+7 %**         |
| `avg_citation_match`    | 0.188    | **0.438**   | **+133 %**       |
| `citation_rate`         | 0.312    | **0.562**   | **+80 %**        |
| `avg_latency_seconds`   | 27.97    | 27.93       | flat             |
| `avg_cost_usd / query`  | $0.00014 | $0.00013    | flat             |

`citation_match` more than doubles. Latency flat (the HyDE call is ~150 output tokens and gets dwarfed by tenacity backoff against free-tier 429s anyway). Cost unchanged at the floor.

This is the kind of result that exists *only* because there was a number to measure it against. Pre-eval, a developer might have shipped HyDE and felt good about it. Eval-driven, you ship HyDE and you *prove* the +133 %.

---

## ADR-012 — The bug HyDE didn't fix

While writing ADR-011 I noticed a comment in `hybrid.py` that I'd written months earlier and forgotten:

```python
# Map back to the full chunk row (only dense_hits carry the content; for
# candidates returned only by BM25 we'd need a DB lookup — kept simple here
# by retaining only the dense-side payloads.)
```

That's a real bug, hiding in plain sight. The Reciprocal Rank Fusion stage was producing a list of `(source_url, chunk_index)` tuples that *included* BM25-only winners. But the next loop was dropping anything not in `dense_hits` because dense_hits was the only path carrying full chunk payloads. The reranker never saw the sparse-only candidates.

The fix is one batched SELECT — `fetch_chunks_by_keys` over an `UNNEST(text[], int[])`:

```python
sql = """
    SELECT source_url, title, section_path, chunk_index, content
    FROM documents d
    JOIN UNNEST(%s::text[], %s::int[]) AS k(url, idx)
      ON d.source_url = k.url AND d.chunk_index = k.idx
"""
```

And in `hybrid_search`:

```python
by_key = {(h.source_url, h.chunk_index): h for h in dense_hits}
missing_keys = [
    (item.source_url, item.chunk_index)
    for item in fused
    if (item.source_url, item.chunk_index) not in by_key
]
if missing_keys:
    recovered = await fetch_chunks_by_keys(missing_keys)
    by_key.update(recovered)
```

One extra round-trip per request when sparse-only winners exist, ~20-80 ms on Neon. Negligible against a 3-4 s rerank stage.

### The trap of small subsets

I ran `cdrag eval --limit 5` as a smoke test before committing the full re-baseline. The numbers came back **mixed**:

| 5-Q smoke               | Pre-fix  | Post-fix    | Delta              |
|-------------------------|----------|-------------|--------------------|
| `avg_topic_coverage`    | 0.600    | 0.487       | **−18.9 %** ⚠     |
| `avg_citation_match`    | 0.600    | **0.800**   | +33 % ✅           |
| `citation_rate`         | 0.800    | 0.800       | flat               |

The 19 % topic-coverage regression on a 5-question subset was tempting to roll back at that point. *Stop. Don't ship this.* But the metric is a substring-overlap proxy and 5 questions is statistical noise. So I ran the full 32:

| Metric                  | + HyDE (ADR-011) | **+ HyDE + sparse-recovery (ADR-012)** | Delta vs ADR-011  |
|-------------------------|------------------|------------------------------------------|-------------------|
| `avg_topic_coverage`    | 0.563            | **0.636**                                | **+13 %**         |
| `avg_citation_match`    | 0.438            | **0.594**                                | **+36 %**         |
| `citation_rate`         | 0.562            | **0.688**                                | **+22 %**         |
| `avg_latency_seconds`   | 27.93 s          | **23.94 s**                              | **−14 %** (faster) |
| `avg_cost_usd / query`  | $0.00013         | $0.00012                                 | flat               |

All three guarded metrics up. Latency *down* 14 % because the rerank stage now lands on chunks that are more reliably grounded — fewer hopeless retries against irrelevant context.

The lesson: **the 5-Q sanity test was wrong**. Without committing to the full-corpus ablation, the right fix would have been left in a branch with a "topic coverage regressed, do not merge" note. Eval discipline means trusting the bigger sample.

---

## The arc, end to end

| Metric                 | Pre-HyDE (ADR-010) | + HyDE (ADR-011) | **+ Sparse-recovery (ADR-012)** | Total |
|------------------------|--------------------|------------------|----------------------------------|-------|
| `avg_topic_coverage`   | 0.526              | 0.563            | **0.636**                        | **+21 %** |
| `avg_citation_match`   | 0.188              | 0.438            | **0.594**                        | **+216 %** |
| `citation_rate`        | 0.312              | 0.562            | **0.688**                        | **+121 %** |
| `avg_latency_seconds`  | 27.97 s            | 27.93 s          | **23.94 s**                      | **−14 %** |
| `avg_cost_usd / query` | $0.00014           | $0.00013         | **$0.00012**                     | flat |

`citation_match` more than tripled. Latency dropped. Cost stayed at the floor. All on free-tier Llama 3.1 8B.

None of these numbers is *good* against the README targets (0.85 / 0.90 / 0.95). What matters is that the next ADRs in the queue have *measurable next steps*:

- **ADR-013** — promote the agent to `llama-3.3-70b-versatile`. The 8B model is the cap on `citation_rate` (it just doesn't always emit `[n]` markers); a stronger free-tier model should close that gap.
- **ADR-014** — query rewriting or multi-query expansion on top of HyDE, if `topic_coverage` stalls.

If I'd shipped HyDE blindly, the celebration would have been "I added HyDE!". With eval, the celebration is "I added HyDE, measured +133 % on the right metric, found a fusion bug *while* writing the ADR, fixed it, measured another +36 %, and the next ADR's tolerance is already wired into CI."

---

## Things that surprised me

- **HyDE adds zero latency in practice.** The extra LLM call is ~150 output tokens. On free-tier Groq with tenacity backoff dominating the latency budget, it's invisible.
- **Sparse-only winners were ~30 % of fusion candidates.** I didn't expect that many. The "kept simple" comment from my past self cost me roughly that fraction of the reranker's input.
- **Free-tier Groq is enough.** I burned through the 70 B daily TPD cap once and ran the rest of the eval on the 8 B. Total `$` spent across three ablations: **$0**.
- **Static eval baselines age fast.** Every ADR bumped `evals/baseline.json`. The CI gate is now a moving target — which is correct, that's how regression gates should work.

---

## What the repo gives you for free

[`alvarocanoo/claude-docs-rag`](https://github.com/alvarocanoo/claude-docs-rag) — MIT.

- `cdrag eval` — run the full suite locally in ~15 min on Groq free tier.
- `evals/baseline.json` — committed numbers for every ADR.
- `scripts/check_regression.py` — apples-to-apples comparison by question id.
- `.github/workflows/ci.yml` — gate that *actually runs* on PR with secrets gating for fork-safety.
- 12 ADRs in `docs/DECISIONS.md` covering every architectural choice including the ones above.
- Live demo: <https://claude-docs-rag.vercel.app>.

Star it if you found this useful. Issues / PRs welcome on the next ADRs (013, 014).

---

*Want eval-driven dev for your own LLM app? Steal the structure: golden dataset, three metrics, latest-vs-baseline gate, ADR per change. The numbers fall out.*
