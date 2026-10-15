---
title: GAIA Agent (Agents Course Final)
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# 🕵🏻 GAIA Agent — Hugging Face Agents Course Final Assignment

A from-scratch, tool-using agent for the [GAIA](https://huggingface.co/gaia-benchmark)
benchmark (the 20-question Level-1 subset used by the HF Agents Course). It scores
**7/20 (35%)** on the official course `/submit` endpoint — above the 30% target —
with a **deterministic 6/20 floor** that doesn't depend on web luck.

> **No answers are hardcoded.** Every point comes from the agent reasoning, running
> code, reading files, or searching the web. The ground-truth answers were used only
> to *diagnose* format-vs-content failures, never to shortcut them.

This README documents not just the final design but the **full research process** —
including everything that didn't work and the concrete ceiling we hit — because the
negative results are the most useful part of the story.

---

## Results

| Metric | Value |
|---|---|
| Official score | **7 / 20 (35%)** |
| Deterministic floor | 6 / 20 — Q3, Q6, Q7, Q12, Q14, Q19 (reproducible every run) |
| Model | `Qwen/Qwen2.5-72B-Instruct`, pinned to the `deepinfra` provider |
| Score progression | 0 → 1 → 4 → **7** as the agent was hardened |

### Full question-by-question breakdown

| # | Type | Status | How / Why |
|---|------|--------|-----------|
| Q3 | reasoning | ✅ solved | reverse the string via `run_python`, auto-decoded |
| Q6 | logic table | ✅ solved | build table in `run_python`, check all pairs |
| Q7 | video audio | ✅ solved | `yt-dlp` → Groq Whisper transcription |
| Q12 | python file | ✅ solved | run the attachment → deterministic output |
| Q14 | audio file | ✅ solved | Whisper transcription of the recording |
| Q19 | spreadsheet | ✅ solved | sum the food columns (exclude drinks) |
| Q5 | web (Wikipedia FAC) | ⚠️ sometimes | nominator is findable but web is noisy |
| Q16 | web (academic) | ⚠️ sometimes | right city when search lands correctly |
| Q20 | web (table) | ⚠️ sometimes | winner is in a Wikipedia table |
| Q10 | audio | ⚠️ close | transcript correct, exact ingredient wording drifts |
| Q1 | web (counting) | ❌ | "studio album" count is ambiguous (3 vs 4) |
| Q8 | web (precise) | ❌ | surname in one specific LibreText exercise |
| Q9 | reasoning (botany) | ❌ | botanical-fruit classification |
| Q11 | web (multi-hop) | ❌ | actor → role, cross-language |
| Q13 | web (stats table) | ❌ | at-bats of the 1977 walks leader |
| Q15 | web (precise) | ❌ | NASA award buried in a paper's acknowledgments |
| Q17 | web (table) | ❌ | fewest-athletes country at the 1928 Olympics |
| Q18 | web (roster) | ❌ | NPB pitchers by jersey number |
| Q4 | vision (chess) | ❌ | VLMs misread the board position |
| Q2 | video | ❌ | frame-level bird-species counting |

---

## How it works

The agent runs a simple, auditable **JSON tool loop** (`agent.py`): at each step the
model returns strict JSON — a tool call or a final answer — and the loop executes the
tool, appends the (length-bounded) result to history, and repeats until a final answer
or a step budget is reached.

```
question ──▶ ModelClient ──▶ {"type":"tool", ...}  ──▶ execute tool ──┐
   ▲                                                                   │
   └──────────────── bounded history (+ tool result) ◀────────────────┘
                         │
                         ▼
          {"type":"final","answer":...} ──▶ format normalization ──▶ answer
```

### Tools

| Tool | Purpose |
|---|---|
| `web_search` / `visit_webpage` | Wikipedia API + DuckDuckGo search; clean page/table text |
| `run_python` | Code interpreter — the model writes Python for math, logic, tables, string ops |
| `read_attachment_text` / `analyze_spreadsheet` / `execute_python_attachment` | Read/run attached files |
| `transcribe_audio` / `youtube_transcribe` | Audio (and YouTube audio) → text via Groq Whisper |
| `inspect_image` / `inspect_video` / `youtube_research` | Media metadata + research helpers |

### Key design decisions

- **Provider pinning for determinism.** The HF Inference router load-balances a model
  across backend providers and is *non-deterministic even at temperature 0*. Pinning
  `Qwen/Qwen2.5-72B-Instruct:deepinfra` makes runs reproducible.
- **Code over guessing.** Arithmetic, logic tables, and string manipulation go through
  `run_python`, so they're exact instead of hallucinated.
- **Gated-dataset file fallback.** The course scoring API serves *no* files for these
  tasks, so `file_tools.py` fetches them from the gated `gaia-benchmark/GAIA` dataset.
- **Exact-match formatting.** The scorer is near-exact string match — `b, e` ≠ `b,e`.
  Answers are normalized to GAIA's format (space after list commas, no `$`, no trailing
  period, number-only for "how many").
- **Resilient client.** Retries on rate limits (429) and transient gateway errors (5xx),
  and falls back to non-JSON mode if a provider rejects `response_format`.

---

## 🔬 Research journey (what we tried, in order)

This project was as much an infrastructure and evaluation study as an agent build.
The score climbed **0 → 1 → 4 → 7** as each blocker was found and removed:

| Stage | Problem found | Fix | Result |
|---|---|---|---|
| **Local baseline** | `qwen2:7b` via Ollama was weak/slow | tool loop + local dry evaluator | runs, but wrong answers |
| **First hosted run** | HF Inference credits exhausted (HTTP 402) | — | **0/20** |
| **Groq free tier** | 200k tokens/**day** cap — one run exhausts it; 13 questions died | retry/backoff, token-slashing | **1/20** (run couldn't finish) |
| **Determinism** | HF router is non-deterministic at temp 0 (routes to different providers) | **pin `:deepinfra`** | reproducible runs |
| **Reasoning** | model miscomputes logic/strings | `run_python` code interpreter + auto-decode | Q3, Q6 locked |
| **File attachments** | scoring API returns *"No file path"* (404) for every file | **fallback to the gated GAIA dataset** + Groq Whisper | Q10/Q12/Q14/Q19 unlocked |
| **Submission #2** | several right answers scored wrong | discovered the scorer is **exact-match** (`b, e` ≠ `b,e`) → format normalization | **4/20 → 7/20** |
| **Video audio** | YouTube "what did X say" unanswered | `yt-dlp` → Whisper | Q7 added |
| **Web research** | multi-hop questions unreliable | real search API (Tavily, optional), **free Wikipedia API**, bigger context window | marginal — see below |
| **Model study** | is a bigger brain the answer? | A/B'd DeepSeek-V3.2, Kimi-K2, Qwen3-235B | **no** — see below |

### The model A/B study (why a bigger model wasn't the answer)

The hard web questions were fed clean, correct source pages — yet still failed. To
test whether the bottleneck was the *model*, three frontier models were benchmarked on
the 10 failing web questions (vs. `Qwen2.5-72B`'s 0/10):

| Model | Web score | Observation |
|---|---|---|
| Qwen3-235B-A22B-Instruct | **2/10** | best; got Q1 and Q16 |
| DeepSeek-V3.2 | 0/10 | slow; refused ("I don't have access…") |
| Kimi-K2-Instruct | 0/10 | stalled; refused |

Even frontier models couldn't extract the answers — they more often *refused* than
dug deeper. **The bottleneck is precise retrieval, not model size.**

---

## 🚧 Why the ceiling is ~35–40% (the free-tier blockade)

GAIA Level 1 is deliberately broad. Of the 13 questions we don't reliably solve, each
needs a capability that isn't available on a **free** stack:

1. **Precise retrieval (Q8, Q13, Q15, Q17, Q18).** The answer lives in one specific
   page/table/paper (a single LibreText exercise, a stat line, a paper's
   acknowledgments, a jersey-number roster). Free search (DuckDuckGo + the Wikipedia
   API) doesn't reliably surface the *exact* source, so no model can extract it. This
   needs a paid, agent-grade search API — and even then success isn't guaranteed.
2. **Vision (Q4, chess).** Every vision LLM tested — including Qwen3-VL-235B — misreads
   the exact board position. A reliable fix needs a specialized board-recognition model
   plus a chess engine (Stockfish).
3. **Video (Q2, bird count).** Requires frame-level, fine-grained species recognition
   over time — effectively out of scope.
4. **Ambiguity (Q1, Q9).** "Studio album" counts and botanical-fruit classification are
   genuinely contestable; even correct retrieval doesn't pin the expected answer.

So the honest reachable ceiling with **free models + free search** is ~35–40%. Pushing
higher would mean: a paid search API (→ maybe ~45–50%), plus a vision+engine pipeline,
plus video understanding — each with sharply diminishing returns.

**What we deliberately did *not* do:** hardcode answers, scrape the validation labels
into the agent, or overfit prompts to individual questions. The 6/20 floor is real,
general, and reproducible.

---

## Project structure

```
agent.py             # GaiaAgent: the JSON tool loop, prompt, answer normalization
model.py             # ModelClient: ollama / hf / groq providers, retry & fallback
web_tools.py         # Wikipedia API + DuckDuckGo search, page/table reader
file_tools.py        # attachment download (+ gated GAIA dataset fallback), run_python
multimodal_tools.py  # Whisper transcription (file + YouTube), media metadata
app.py               # Gradio UI: runs all 20 questions and submits to the course API
local_eval.py        # local dry-run harness (writes JSONL/CSV/Markdown reports)
tests/               # unit tests (24)
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

| `MODEL_PROVIDER` | Needs | Notes |
|---|---|---|
| `hf` (recommended) | `HF_TOKEN` (Read + gated-repo access) | HF Inference router; required for the file fallback |
| `groq` | `GROQ_API_KEY` | fast; free tier has a tight daily token cap |
| `ollama` | local Ollama | fully local, no keys |

`GROQ_API_KEY` also powers audio/YouTube transcription regardless of the chat provider.

### Local dry run (does not submit)

```bash
# PowerShell
$env:MODEL_PROVIDER="hf"
$env:HF_MODEL="Qwen/Qwen2.5-72B-Instruct:deepinfra"
$env:HF_TOKEN="<token>"; $env:GROQ_API_KEY="<groq key>"
$env:TRUST_GAIA_PYTHON_ATTACHMENTS="true"
python local_eval.py            # all 20 → reports in local_runs/
python local_eval.py --limit 3
```

### Deploy as a Hugging Face Space

This repo doubles as a Gradio Space (see the YAML header). Push it, set the
secrets/variables below, log in, and click **Run Evaluation & Submit All Answers**.

**Secrets:** `HF_TOKEN`, `GROQ_API_KEY`
**Variables:**

```
MODEL_PROVIDER=hf
HF_MODEL=Qwen/Qwen2.5-72B-Instruct:deepinfra
AGENT_MAX_STEPS=6
MODEL_TIMEOUT_SECONDS=120
WEB_SEARCH_ENABLED=true
TRUST_GAIA_PYTHON_ATTACHMENTS=true
WEB_PAGE_CHAR_LIMIT=9000
WEB_SEARCH_MAX_RESULTS=5
```

---

## Testing

```bash
python -m unittest discover -s tests -q   # 24 tests
```

Covers the tool loop, provider selection, file tools, answer normalization, and the
local evaluator.

---

## Acknowledgements

Built for the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course)
final assignment. GAIA benchmark by Mialon et al. Licensed under MIT.
