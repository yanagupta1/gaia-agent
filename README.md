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

A from-scratch tool-using agent that answers exact-match questions from the
[GAIA](https://huggingface.co/gaia-benchmark) benchmark (Level 1 subset used by the
Hugging Face Agents Course). It scores **7/20 (35%)** on the official course
`/submit` endpoint — above the 30% target — with a **deterministic 6/20 floor**
that does not depend on web luck.

> No answers are hardcoded. Every point comes from the agent reasoning, running
> code, reading files, or searching the web.

---

## Results

| Metric | Value |
|---|---|
| Official score | **7 / 20 (35%)** |
| Deterministic floor | 6 / 20 (Q3, Q6, Q12, Q14, Q16, Q19) |
| Model | `Qwen/Qwen2.5-72B-Instruct` (pinned to the `deepinfra` provider) |
| Progression | 0 → 1 → 4 → **7** as the agent was hardened |

**How the questions break down**

| Category | Questions | Status |
|---|---|---|
| Reasoning / compute | reversed-string decode, non-commutativity table | ✅ solved via `run_python` |
| File attachments | Python exec, spreadsheet sum, audio transcription | ✅ solved (dataset fallback + Whisper) |
| Web / multi-hop | discography counts, Wikipedia facts, standings | ⚠️ partial (model research ceiling) |
| Vision / video | chess board image, YouTube bird count | ❌ out of scope (no vision backend) |

---

## How it works

The agent runs a simple, auditable **JSON tool loop** (`agent.py`): at each step the
model returns strict JSON — either a tool call or a final answer — and the loop
executes the tool, appends the result to a bounded history, and continues until a
final answer or a step budget is reached.

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
| `web_search` / `visit_webpage` | DuckDuckGo search + readable page text |
| `run_python` | Code interpreter — the model writes Python; used for math, logic, set/table analysis, string reversal (gated by `TRUST_GAIA_PYTHON_ATTACHMENTS`) |
| `read_attachment_text` / `analyze_spreadsheet` / `execute_python_attachment` | Read/run attached files |
| `transcribe_audio` | Audio → text via Groq `whisper-large-v3` |
| `inspect_image` / `inspect_video` / `youtube_research` | Media metadata + research helpers |

### Key design decisions (the hard-won lessons)

- **Provider pinning for determinism.** The HF Inference router load-balances a
  model across backend providers and is *non-deterministic even at temperature 0*.
  Pinning `Qwen/Qwen2.5-72B-Instruct:deepinfra` makes runs reproducible.
- **Code over guessing.** Arithmetic, logic tables, and string manipulation are
  delegated to `run_python` so they're exact instead of hallucinated.
- **Attachment fallback.** The course scoring API serves no files for these tasks,
  so `file_tools.py` falls back to the gated `gaia-benchmark/GAIA` dataset
  (`2023/validation/<file>`) using `HF_TOKEN`.
- **Exact-match formatting.** The scorer is near-exact string match — `b, e` ≠
  `b,e`. Final answers are normalized to GAIA's format (single space after list
  commas, no leading currency symbol, number-only for "how many").
- **Resilient model client.** Retries on rate limits (429) and transient gateway
  errors (5xx), and falls back to non-JSON mode if a provider rejects
  `response_format`.

---

## Project structure

```
agent.py             # GaiaAgent: the JSON tool loop, prompt, answer normalization
model.py             # ModelClient: ollama / hf / groq providers, retry & fallback
web_tools.py         # DuckDuckGo search + HTML-to-text reader
file_tools.py        # attachment download (+ GAIA dataset fallback), run_python, spreadsheets
multimodal_tools.py  # image/video metadata, Groq Whisper transcription, youtube research
app.py               # Gradio UI that runs all 20 questions and submits to the course API
local_eval.py        # local dry-run harness (writes JSONL/CSV/Markdown reports)
tests/               # unit tests (24)
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

### Providers

| `MODEL_PROVIDER` | Needs | Notes |
|---|---|---|
| `hf` (recommended) | `HF_TOKEN` (Read + gated-repo access) | Uses the HF Inference router; required for the file-attachment fallback |
| `groq` | `GROQ_API_KEY` | Fast; free tier has a tight daily token cap |
| `ollama` | local Ollama | Fully local, no keys |

`GROQ_API_KEY` is also used for audio transcription regardless of the chat provider.

### Run the local dry evaluator

The dry runner executes the agent on all 20 questions and writes reports to
`local_runs/`. It does **not** submit and cannot compute the official score.

```bash
# PowerShell
$env:MODEL_PROVIDER="hf"
$env:HF_MODEL="Qwen/Qwen2.5-72B-Instruct:deepinfra"
$env:HF_TOKEN="<your token>"
$env:GROQ_API_KEY="<your groq key>"
$env:TRUST_GAIA_PYTHON_ATTACHMENTS="true"
python local_eval.py            # all 20
python local_eval.py --limit 3  # first 3
```

### Deploy to a Hugging Face Space

This repo doubles as a Gradio Space (see the YAML header). Push it to a Space, set
the secrets/variables below, then log in and click **Run Evaluation & Submit All
Answers**.

**Secrets:** `HF_TOKEN`, `GROQ_API_KEY`
**Variables:**

```
MODEL_PROVIDER=hf
HF_MODEL=Qwen/Qwen2.5-72B-Instruct:deepinfra
AGENT_MAX_STEPS=6
MODEL_TIMEOUT_SECONDS=120
WEB_SEARCH_ENABLED=true
TRUST_GAIA_PYTHON_ATTACHMENTS=true
WEB_PAGE_CHAR_LIMIT=6000
WEB_SEARCH_MAX_RESULTS=5
```

---

## Testing

```bash
python -m unittest discover -s tests -q
```

24 tests cover the tool loop, provider selection, file tools, answer
normalization, and the local evaluator.

---

## Limitations & next steps

- **Vision / video questions** (chess image, YouTube bird count) are unanswered —
  they'd need a vision backend plus a chess engine, and frame-level video analysis.
- **Web / multi-hop questions** are the noisy frontier: DuckDuckGo scraping and the
  model's multi-hop research are inconsistent. A proper search API would help most.
- `TRUST_GAIA_PYTHON_ATTACHMENTS` runs model/attachment Python on the host; it is
  **off by default** and intended for the trusted benchmark setting only. A sandbox
  would be the production hardening.

---

## Acknowledgements

Built for the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course)
final assignment. GAIA benchmark by Mialon et al.
