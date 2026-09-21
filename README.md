# Cognify-AI

**Self-Correcting LLM Framework for Detecting and Reducing Hallucinations**

Cognify-AI generates an answer with a local LLM, breaks that answer into individual factual claims, checks each claim against a trusted local knowledge base, computes a transparent **Hallucination Risk Score**, and automatically rewrites the answer from the retrieved evidence when the risk is too high — then verifies it again.

Everything runs on your own machine: **Ollama** for generation, **Sentence-Transformers** for embeddings, **FAISS** for vector search, a local **NLI** model for entailment checking, **FastAPI** for the backend and plain **HTML/CSS/JS** for the frontend.
**No OpenAI, no Gemini, no Claude, no API key, no cloud service.**

---

## 1. Project title

Cognify-AI — Self-Correcting LLM Framework for Detecting and Reducing Hallucinations.

## 2. Project description

A large language model produces text that is *likely*, not text that is *true*. When it has no reliable knowledge it still answers fluently and confidently — this is called a hallucination. Cognify-AI wraps a local LLM in a verification loop: every factual sentence the model produces is retrieved against, entailment-checked, scored, and — when it fails — regenerated using only the evidence that was actually found.

## 3. Problem statement

Language models cannot tell the difference between a fact they learned and a plausible sentence they invented. In education, healthcare, law and research, a confident wrong answer is worse than "I don't know". There is no signal in a raw LLM answer that tells a user which parts are grounded. Cognify-AI adds that missing signal, makes it explainable claim by claim, and uses it to repair the answer automatically.

## 4. Objectives

1. Generate answers with a local LLM through Ollama, with no API key.
2. Extract the individual factual claims from each answer.
3. Retrieve supporting evidence from a local trusted knowledge base (RAG).
4. Classify each claim as SUPPORTED / CONTRADICTED / INSUFFICIENT using NLI.
5. Combine the signals into a transparent 0–100 Hallucination Risk Score.
6. Automatically self-correct high-risk answers using the retrieved evidence.
7. Re-verify after each correction, with a hard limit on iterations.
8. Present the whole trace — claims, evidence, scores, corrections — in a readable UI.

## 5. Features

- Local LLM generation via Ollama (default `llama3.2:3b`, runs on 16 GB RAM).
- Claim extraction with JSON output and a rule-based fallback for broken JSON.
- Local RAG: `.txt`, `.md`, `.pdf` → chunks → MiniLM embeddings → FAISS.
- NLI verification with `cross-encoder/nli-deberta-v3-base` and a lexical fallback.
- Transparent, configurable risk formula (40/30/20/10).
- Self-correction loop with a maximum-iteration guard and a "never accept a worse rewrite" rule.
- Confidence score, verification status, correction trace, per-claim evidence with sources.
- FastAPI backend with automatic docs at `/docs`, plus a static frontend on the same port.
- Graceful degradation: missing NLI model, missing FAISS, missing embeddings, Ollama down, empty knowledge base — none of these crash the app.
- Separate detection and correction benchmarks.

## 6. Architecture

```
                    User Query
                        |
                 Ollama LLM (local)
                        |
                Initial Response
                        |
                 Claim Extraction
                        |
              Hallucination Detection
        +---------------+----------------+
        |               |                |
   RAG retrieval   NLI entailment   Confidence /
  (FAISS + MiniLM)   verification    uncertainty
        +---------------+----------------+
                        |
             Hallucination Risk Score (0-100)
                        |
                 Is risk > threshold?
                 |                   |
                NO                  YES
                 |                   |
         Final Verified       Correction prompt
            Response        (answer + failed claims
                                 + evidence)
                                     |
                              Ollama LLM (local)
                                     |
                             Corrected Response
                                     |
                              Verify again  --> loop
                                                (max 3)
```

## 7. Workflow

1. The user submits a question.
2. `generator.py` asks Ollama for an answer.
3. `claim_extractor.py` splits the answer into standalone factual claims.
4. `rag.py` retrieves the top-k most similar knowledge-base chunks for each claim.
5. `verifier.py` runs NLI (evidence = premise, claim = hypothesis) and labels the claim.
6. `risk_scorer.py` aggregates the labels and uncertainties into a 0–100 risk score.
7. If risk ≤ threshold → the answer is returned as the final answer.
8. If risk > threshold → `corrector.py` builds an evidence-grounded rewrite prompt and Ollama rewrites the answer.
9. The rewritten answer goes back to step 3. The loop ends when the risk is acceptable or `MAX_ITERATIONS` is reached.
10. The API returns the original answer, the final answer, both scores, every claim, every piece of evidence and the full correction trace.

## 8. Technologies used

| Layer | Technology | Why |
|---|---|---|
| LLM | Ollama + `llama3.2:3b` | Runs locally, no key, ~2 GB model |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, CPU-friendly |
| Vector store | FAISS (`IndexFlatIP`) | Fast exact cosine search; NumPy fallback |
| Verification | `cross-encoder/nli-deberta-v3-base` | Entailment / contradiction / neutral |
| Backend | FastAPI + Uvicorn | Typed validation, auto docs |
| Frontend | HTML + CSS + vanilla JS | No build step, opens instantly |
| Tests | pytest | Runs fully offline |

## 9. System requirements

- Windows 10/11 (the same commands work on Linux/macOS with `source venv/bin/activate`)
- Python 3.10 – 3.12
- 16 GB RAM recommended (8 GB works with `llama3.2:1b` and `ENABLE_NLI=false`)
- ~6 GB free disk (Ollama model ~2 GB, PyTorch ~2.5 GB, NLI model ~0.4 GB)
- Internet **only for the first install** (pip + first model download). The pipeline itself is offline.

## 10. Installation (overview)

```bat
:: put the project in C:\Cognify-AI, then:
cd C:\Cognify-AI
setup_windows.bat
ollama pull llama3.2:3b
run_backend.bat
:: open http://localhost:8000
```

The next sections give the same steps manually.

## 11. Ollama installation (Windows)

1. Download the Windows installer from https://ollama.com/download and run `OllamaSetup.exe`.
2. Ollama starts automatically and listens on `http://localhost:11434`.
3. Check it in a terminal:

```bat
ollama --version
curl http://localhost:11434/api/tags
```

If the server is not running, start it with:

```bat
ollama serve
```

## 12. Model download

```bat
ollama pull llama3.2:3b
ollama list
ollama run llama3.2:3b "Say hello in one sentence"
```

Lighter alternative for a slow or 8 GB machine:

```bat
ollama pull llama3.2:1b
```

Then set `OLLAMA_MODEL=llama3.2:1b` in `.env`.

## 13. Python environment setup

```bat
cd C:\Cognify-AI
python --version
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

Your prompt should now start with `(venv)`.

## 14. Requirements installation

```bat
pip install -r requirements.txt
copy .env.example .env
```

This downloads PyTorch and takes several minutes the first time. CPU-only PyTorch is enough.

## 15. Knowledge-base setup

Five sample documents ship in `knowledge_base/`:

```
geography.txt          capitals, rivers, continents
science.txt            boiling point, light, atmosphere, biology
computer_science.txt   Python, Transformers, RAG, NLI, Ollama, FAISS
history.txt            WWII, Apollo 11, Indian independence, Taj Mahal
cognify_ai_project.txt this project, so the system can describe itself
```

Add your own `.txt`, `.md` or `.pdf` files to the same folder. Then either restart the server or call:

```bat
curl -X POST http://localhost:8000/api/reload-kb
```

Embeddings are cached in `data/vector_index/` and are rebuilt automatically when the documents change.

## 16. Running the backend

```bat
cd C:\Cognify-AI
venv\Scripts\activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The first request loads the embedding model and the NLI model (one-time download, ~400 MB for NLI).
Interactive API docs: http://localhost:8000/docs

## 17. Running the frontend

The backend serves the frontend, so simply open:

```
http://localhost:8000
```

To open the page directly from disk instead (`file:///C:/Cognify-AI/frontend/index.html`), the script automatically falls back to `http://localhost:8000` for API calls, and CORS is already enabled.

## 18. API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Ollama status, installed models, knowledge-base stats |
| GET | `/api/config` | Active models, weights, thresholds |
| POST | `/api/chat` | Full pipeline: generate → detect → correct → verify |
| POST | `/api/verify` | Verify an answer you already have (no generation) |
| POST | `/api/correct` | Run exactly one evidence-grounded correction round |
| POST | `/api/reload-kb` | Rebuild the vector index after adding documents |

Request body for `/api/chat`:

```json
{ "query": "What is the capital of Australia?", "max_iterations": 2 }
```

## 19. Example input / output

**Input**

```bat
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"query\":\"What is the capital of Australia?\"}"
```

**Output (shortened)**

```json
{
  "query": "What is the capital of Australia?",
  "original_answer": "Sydney is the capital of Australia and its largest city.",
  "final_answer": "Canberra is the capital of Australia. Sydney is the largest city of Australia by population, but it is not the capital.",
  "risk_score": 12,
  "confidence_score": 88,
  "status": "LOW_RISK",
  "verification_status": "VERIFIED",
  "iterations": 1,
  "corrected": true,
  "claims": ["Canberra is the capital of Australia.", "Sydney is the largest city of Australia."],
  "evidence": [
    {
      "claim": "Canberra is the capital of Australia.",
      "status": "SUPPORTED",
      "entailment_score": 0.97,
      "best_evidence": "Canberra is the capital city of Australia...",
      "best_evidence_source": "geography.txt"
    }
  ]
}
```

### Test questions for a demo

| # | Question | What it demonstrates |
|---|---|---|
| 1 | What is the capital of France? | Correct answer → SUPPORTED, low risk, no correction |
| 2 | What is the capital of Australia? | Small models often say Sydney → CONTRADICTED → corrected to Canberra |
| 3 | Can the Great Wall of China be seen from space? | A classic myth → contradicted by `science.txt` |
| 4 | At what temperature does water boil at sea level? | Numeric claim verification |
| 5 | Who created Python and when was it released? | Multi-claim answer, each claim checked separately |
| 6 | Who won the under-19 inter-college cricket league in Hyderabad in 2019? | Not in the knowledge base → INSUFFICIENT → the system says it could not be verified instead of inventing |
| 7 | How does Cognify-AI compute its risk score? | The system explains itself from `cognify_ai_project.txt` |

## 20. Hallucination detection process

1. **Claim extraction** — the LLM is asked to return `{"claims": [...]}`. The parser strips markdown fences, finds the first balanced `{...}` block, and falls back to sentence splitting if the JSON is invalid or Ollama is unavailable. Pronouns are resolved by the prompt so each claim stands alone.
2. **Retrieval** — each claim is embedded with MiniLM and matched against the FAISS index. Chunks below `RETRIEVAL_MIN_SCORE` (0.25 cosine) are discarded as irrelevant.
3. **Entailment** — for every surviving chunk the NLI cross-encoder scores `premise = evidence, hypothesis = claim` and returns entailment / neutral / contradiction.
4. **Labelling** —
   - contradiction ≥ 0.50 and greater than entailment → **CONTRADICTED**
   - entailment ≥ 0.60 → **SUPPORTED**
   - otherwise, or no usable evidence → **INSUFFICIENT**
5. **Uncertainty** — verifier uncertainty is `1 − max(probability)` averaged over claims; model uncertainty is a hedging heuristic ("I think", "possibly", very short answers).

**INSUFFICIENT means "not in my knowledge base", not "false".** That distinction is the honest core of the system.

## 21. Self-correction process

When `risk_score > RISK_THRESHOLD` (default 30), `corrector.py` builds this prompt:

```
You generated the following answer to the question: "<query>"

ANSWER: <answer>

The following claims could NOT be verified, or were CONTRADICTED by trusted sources:
- [CONTRADICTED] Sydney is the capital of Australia.

TRUSTED EVIDENCE (from the verified knowledge base):
- (geography.txt) Canberra is the capital city of Australia...

Rewrite the answer using ONLY information supported by the evidence above.
Do not invent facts, numbers, names or dates.
If the evidence is insufficient, explicitly write that this information could not be verified.
Return only the corrected answer.
```

The rewrite is sent back through the entire detection stage. Three guards prevent bad behaviour:

- **Iteration cap** — at most `MAX_ITERATIONS` (default 3) correction rounds, so the loop always ends.
- **No regression** — if a rewrite scores *worse* than the previous answer, the previous answer is kept and a warning is returned.
- **Honest failure** — if the risk is still high after the last round, the response carries a warning instead of a false claim of success.

## 22. Risk-score calculation

```
Risk (0-100) = 100 x ( 0.40 * contradiction_risk
                     + 0.30 * unsupported_risk
                     + 0.20 * nli_uncertainty
                     + 0.10 * model_uncertainty )

contradiction_risk = contradicted claims / total claims
unsupported_risk   = insufficient claims / total claims
nli_uncertainty    = mean(1 - max(entail, neutral, contradict))
model_uncertainty  = hedging heuristic on the answer text

Confidence = 100 - Risk

0-30  LOW RISK      31-60  MEDIUM RISK      61-100  HIGH RISK
```

All weights, thresholds and bands live in `backend/config.py` and can be overridden in `.env`.

> This is a **Hallucination Risk Score**: an engineered indicator produced by this framework. It is **not** a calibrated probability that the answer is false.

## 23. Benchmarking

```bat
python -m evaluation.evaluate_detection
python -m evaluation.evaluate_correction
```

The first measures the detector (precision / recall / F1) and needs no LLM. The second measures the correction loop (success rate, repair rate, average iterations, risk before and after) and needs Ollama running.

**Detection performance** and **correction performance** are reported separately and must not be averaged together.

No dataset is downloaded automatically. `evaluation/README.md` explains how to convert a small slice of TruthfulQA, FEVER or HaluEval into the JSON format used by `evaluation/sample_data/`, and warns that results are only meaningful when the knowledge base actually covers the claims being tested.

## 24. Project structure

```
Cognify-AI/
├── backend/
│   ├── __init__.py
│   ├── main.py             FastAPI app + endpoints + static frontend
│   ├── config.py           all settings, read from .env
│   ├── models.py           pydantic request/response schemas
│   ├── logger.py           shared console + file logger
│   ├── ollama_client.py    local LLM HTTP client with clear errors
│   ├── generator.py        initial answer + model-uncertainty heuristic
│   ├── claim_extractor.py  answer -> list of factual claims (safe JSON parsing)
│   ├── rag.py              load, chunk, embed, FAISS index, retrieve
│   ├── verifier.py         NLI entailment + lexical fallback + claim labelling
│   ├── detector.py         detection stage (claims + evidence + signals)
│   ├── risk_scorer.py      weighted 0-100 risk score, bands, confidence
│   ├── corrector.py        evidence-grounded correction prompt
│   └── pipeline.py         the self-correction loop
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── knowledge_base/         your trusted documents (.txt, .md, .pdf)
├── data/                   generated: vector index, cached embeddings, cognify.log
├── tests/
│   ├── conftest.py
│   ├── test_detector.py
│   ├── test_risk.py
│   └── test_api.py
├── evaluation/
│   ├── evaluate_detection.py
│   ├── evaluate_correction.py
│   ├── README.md
│   └── sample_data/
├── requirements.txt
├── .env.example
├── .gitignore
├── setup_windows.bat
├── run_backend.bat
└── README.md
```

Run the tests with:

```bat
python -m pytest tests -v
```

They pass without Ollama and without downloading any model.

## 25. Limitations

- Cognify-AI **reduces hallucination risk**; it does not eliminate hallucinations.
- Verification is only as good as `knowledge_base/`. A claim that is missing is reported as INSUFFICIENT, which is not the same as false.
- The retriever can miss the right passage; the NLI model can mislabel; the local LLM can ignore the evidence during a rewrite. All three failure modes remain possible.
- The risk score is a weighted heuristic, not a calibrated probability, and the weights were chosen by design, not fitted on a benchmark.
- Small models (1B–3B) are weaker at following the correction instructions than large ones.
- Claim extraction on long, nuanced or opinion-heavy answers is imperfect.
- Each query runs several LLM calls, so a full pipeline takes roughly 10–60 s on a CPU-only laptop.

## 26. Future improvements

- Calibrate the risk score against a labelled dataset and report reliability curves.
- Sentence-level attribution: link every sentence of the final answer to its evidence.
- Self-consistency sampling (generate n answers, measure agreement) as a stronger uncertainty signal.
- Hybrid BM25 + dense retrieval and a cross-encoder re-ranker.
- Allow an optional trusted web source as a second evidence channel.
- Caching of claim-level verdicts, and streaming responses in the UI.
- A human feedback button that stores accepted corrections back into the knowledge base.

## 27. Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama at http://localhost:11434` | Start Ollama: `ollama serve`, or launch the Ollama app |
| `Model 'llama3.2:3b' was not found` | `ollama pull llama3.2:3b`, then check with `ollama list` |
| Backend unreachable in the browser | Start it: `python -m uvicorn backend.main:app --port 8000`, then open http://localhost:8000 |
| `'python' is not recognized` | Install Python from python.org and tick "Add Python to PATH" |
| `'uvicorn' is not recognized` | Activate the venv (`venv\Scripts\activate`) and use `python -m uvicorn ...` |
| Port 8000 already in use | Use `--port 8001`, or find the process with `netstat -ano` and stop it with `taskkill /PID <pid> /F` |
| PowerShell blocks `venv\Scripts\activate` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| First query is very slow | The embedding and NLI models are downloading/loading once; later queries are much faster |
| Out of memory / machine freezes | Use `OLLAMA_MODEL=llama3.2:1b` and `ENABLE_NLI=false` in `.env` |
| NLI model fails to load | The lexical fallback takes over automatically; `pip install sentencepiece` fixes most tokenizer errors |
| `faiss` import fails | The NumPy fallback takes over; optionally reinstall with `pip install faiss-cpu` |
| Everything is INSUFFICIENT | The knowledge base does not cover the topic — add documents and call `POST /api/reload-kb` |
| Answers changed after editing documents | Correct: the embeddings are rebuilt automatically when documents change |

Logs for a viva demo: `data/cognify.log` contains query, initial response, claims, evidence, verification results, risk score, every correction round and the final response. No secrets are logged, because the project has none.

## 28. License

MIT License. Copyright (c) 2026 Cognify-AI contributors.
The models pulled by Ollama and Hugging Face keep their own licences — check them before redistributing.
