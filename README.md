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

# GAIA Agent

A tool-using agent I built from scratch for the Hugging Face Agents Course final
assignment. It answers questions from the GAIA benchmark (the 20-question Level 1
subset the course uses) and scores **7/20 (35%)** on the official `/submit`
endpoint, which clears the 30% pass mark.

Six of those seven are a deterministic floor (Q3, Q6, Q7, Q12, Q14, Q19). They
pass on every run because they come from running code, reading files, or
transcribing audio, not from getting lucky with a web search.

No answers are hardcoded. I used the official answers only to work out whether a
failing question was a formatting problem or a wrong answer, never to shortcut it.

This README also covers what did not work and where the ceiling is, because on a
free stack that turned out to be the most useful part of the story.

## Results

| | |
|---|---|
| Official score | 7/20 (35%) |
| Reliable floor | 6/20 (Q3, Q6, Q7, Q12, Q14, Q19) |
| Model | `Qwen/Qwen2.5-72B-Instruct`, pinned to the `deepinfra` provider |
| Progression | 0 -> 1 -> 4 -> 7 |

### Every question

| # | Type | Status | Notes |
|---|------|--------|-------|
| Q3 | reasoning | pass | reverse the string with `run_python`, then answer |
| Q6 | logic table | pass | build the table in `run_python`, check all pairs |
| Q7 | video audio | pass | `yt-dlp` pulls the audio, Whisper transcribes it |
| Q12 | python file | pass | run the attached script |
| Q14 | audio file | pass | Whisper transcribes the recording |
| Q19 | spreadsheet | pass | sum the food columns, exclude drinks |
| Q5 | web | flaky | nominator is findable, but search is noisy |
| Q16 | web | flaky | right city when the search lands well |
| Q20 | web | flaky | answer sits in a Wikipedia table |
| Q10 | audio | close | transcript is right, exact wording drifts |
| Q1 | web | fail | "studio album" count is ambiguous (3 vs 4) |
| Q8 | web | fail | surname in one specific LibreText exercise |
| Q9 | reasoning | fail | botanical-fruit classification |
| Q11 | web | fail | actor to role, across languages |
| Q13 | web | fail | at-bats of the 1977 walks leader |
| Q15 | web | fail | NASA award buried in a paper's acknowledgments |
| Q17 | web | fail | fewest-athletes country, 1928 Olympics |
| Q18 | web | fail | pitchers by jersey number |
| Q4 | vision | fail | vision models misread the chess board |
| Q2 | video | fail | counting bird species across video frames |

## How it works

`agent.py` runs a plain JSON tool loop. Each step, the model returns JSON that is
either a tool call or a final answer. The loop runs the tool, adds the result to a
length-capped history, and repeats until it gets a final answer or hits the step
limit.

```
  question
     |
     v
  model  ->  tool call  ->  run the tool
     ^                           |
     |___ history + result <_____|
     |
     v
  final answer  ->  format cleanup  ->  submit
```

Tools:

- `web_search` / `visit_webpage`: Wikipedia API plus DuckDuckGo, with clean page
  and table text
- `run_python`: a code interpreter the model writes Python for (math, logic,
  tables, string work)
- `read_attachment_text` / `analyze_spreadsheet` / `execute_python_attachment`:
  read or run attached files
- `transcribe_audio` / `youtube_transcribe`: audio and YouTube audio to text via
  Whisper
- `inspect_image` / `inspect_video` / `youtube_research`: metadata and research
  helpers

A few decisions that mattered:

- Pin the provider. The HF router spreads a model across backends and is not
  deterministic even at temperature 0. Pinning `:deepinfra` makes runs repeatable.
- Prefer code over guessing. Anything numeric or logical goes through `run_python`.
- Fall back to the dataset for files. The scoring API serves no files for these
  tasks (it returns "No file path"), so `file_tools.py` fetches them from the gated
  `gaia-benchmark/GAIA` dataset with an HF token.
- Match the scorer's formatting. It is near-exact string match, so `b, e` is not
  `b,e`. Answers get normalized (space after list commas, no `$`, no trailing
  period, number only for "how many").
- Retry on 429 and 5xx, and drop JSON mode if a provider rejects it.

## What I tried, in order

The score climbed as each blocker got found and removed:

| Stage | Blocker | Fix | Result |
|-------|---------|-----|--------|
| Local baseline | `qwen2:7b` on Ollama was weak and slow | tool loop + local evaluator | runs, wrong answers |
| First hosted run | HF inference credits ran out (402) | none yet | 0/20 |
| Groq free tier | 200k tokens/day cap, one run uses it up | retry/backoff, trim tokens | 1/20 |
| Determinism | HF router not deterministic at temp 0 | pin `:deepinfra` | repeatable runs |
| Reasoning | model miscomputes logic and strings | `run_python` + auto-decode reversed text | Q3, Q6 |
| File questions | scoring API 404s on every file | fall back to the gated dataset + Whisper | Q10, Q12, Q14, Q19 |
| Second submit | right answers scored wrong | scorer is exact match, so normalize formatting | 4/20 -> 7/20 |
| Video audio | "what did X say" unanswered | `yt-dlp` + Whisper | Q7 |
| Web research | multi-hop questions stayed unreliable | Wikipedia API, bigger context | marginal |
| Model study | would a bigger model fix it | A/B DeepSeek, Kimi, Qwen3-235B | no |

On the model study: I fed the failing web questions clean, correct source pages and
they still failed, so I tested three frontier models on the 10 bad ones (the 72B got
0/10 there).

| Model | Web score | Notes |
|-------|-----------|-------|
| Qwen3-235B-A22B-Instruct | 2/10 | best, got Q1 and Q16 |
| DeepSeek-V3.2 | 0/10 | slow, often refused |
| Kimi-K2-Instruct | 0/10 | stalled, refused |

The bigger models refused more than they dug. The bottleneck is finding the exact
source, not model size.

## Why it stops around 35 to 40 percent

The questions I do not solve each need something a free stack does not have:

1. Precise retrieval (Q8, Q13, Q15, Q17, Q18). The answer is in one specific page,
   table, or paper. Free search does not reliably surface that exact source, so no
   model can read it. This needs a paid search API, and even then it is not certain.
2. Vision (Q4). Every vision model I tried, up to Qwen3-VL-235B, misreads the board.
   Doing it properly needs a board-recognition model plus a chess engine.
3. Video (Q2). Counting bird species across frames is out of scope.
4. Ambiguity (Q1, Q9). Album counts and botanical classification are genuinely
   debatable, so even correct retrieval does not pin the expected answer.

Pushing past this means paid search (maybe 45 to 50 percent), plus a vision
pipeline, plus video, for steadily smaller gains.

## Project layout

```
agent.py             GaiaAgent: the tool loop, prompt, answer cleanup
model.py             ModelClient: ollama / hf / groq, retries, fallback
web_tools.py         Wikipedia API + DuckDuckGo search, page reader
file_tools.py        attachment download with dataset fallback, run_python
multimodal_tools.py  Whisper transcription (file and YouTube), media metadata
app.py               Gradio UI that runs all 20 and submits
local_eval.py        local dry-run harness, writes reports to local_runs/
tests/               unit tests (24)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
```

| Provider | Needs | Notes |
|----------|-------|-------|
| `hf` (recommended) | `HF_TOKEN` with gated-repo read | needed for the file fallback |
| `groq` | `GROQ_API_KEY` | fast, but a tight daily token cap on free tier |
| `ollama` | local Ollama | fully local, no keys |

`GROQ_API_KEY` also handles audio and YouTube transcription for any provider.

Local dry run (does not submit):

```bash
# PowerShell
$env:MODEL_PROVIDER="hf"
$env:HF_MODEL="Qwen/Qwen2.5-72B-Instruct:deepinfra"
$env:HF_TOKEN="<token>"; $env:GROQ_API_KEY="<groq key>"
$env:TRUST_GAIA_PYTHON_ATTACHMENTS="true"
python local_eval.py
```

### As a Hugging Face Space

This repo doubles as a Gradio Space (see the header). Push it, set the secrets and
variables, log in, and click Run Evaluation and Submit All Answers.

Secrets: `HF_TOKEN`, `GROQ_API_KEY`

Variables:

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

## Tests

```bash
python -m unittest discover -s tests -q
```

## Certification

Completed the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/),
covering agent architectures, tool integration, orchestration, and evaluation.

📜 [View Certificate](docs/hg_certif.png)

## Notes

Built for the Hugging Face Agents Course final assignment. GAIA benchmark by Mialon
et al. MIT licensed.
