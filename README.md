---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Runtime configuration

Local development uses Ollama by default:

```text
MODEL_PROVIDER=ollama
OLLAMA_MODEL=qwen2:7b
```

Hugging Face Space evaluation (recommended: Groq free tier — no GPU needed, so
the Space can run on free CPU-basic hardware):

```text
MODEL_PROVIDER=groq
GROQ_API_KEY=<space secret>        # free key from console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile
MODEL_TIMEOUT_SECONDS=120
AGENT_MAX_STEPS=4
WEB_SEARCH_ENABLED=true
```

The HF Inference router is also supported when credits are available:

```text
MODEL_PROVIDER=hf
HF_TOKEN=<space secret>
HF_MODEL=Qwen/Qwen2.5-72B-Instruct
MODEL_TIMEOUT_SECONDS=120
AGENT_MAX_STEPS=4
```

Python attachment execution is disabled by default. Enable only for trusted GAIA
benchmark files:

```text
TRUST_GAIA_PYTHON_ATTACHMENTS=true
```
