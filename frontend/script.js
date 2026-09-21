/* Cognify-AI frontend logic.
   Talks to the FastAPI backend. When the page is served by the backend the API
   base is empty (same origin); when index.html is opened directly from disk we
   fall back to http://localhost:8000. */

const API_BASE = location.protocol === "file:" ? "http://localhost:8000" : "";

const $ = (id) => document.getElementById(id);

const els = {
  query: $("query"),
  runBtn: $("runBtn"),
  maxIterations: $("maxIterations"),
  progress: $("progress"),
  alert: $("alert"),
  results: $("results"),
  riskBadge: $("riskBadge"),
  verificationStatus: $("verificationStatus"),
  meterMarker: $("meterMarker"),
  meterValue: $("meterValue"),
  confidenceValue: $("confidenceValue"),
  riskValue: $("riskValue"),
  iterationsValue: $("iterationsValue"),
  claimsValue: $("claimsValue"),
  breakdown: $("breakdown"),
  formulaNote: $("formulaNote"),
  originalAnswer: $("originalAnswer"),
  finalAnswer: $("finalAnswer"),
  claimList: $("claimList"),
  history: $("history"),
  historyPanel: $("historyPanel"),
  serviceDot: $("serviceDot"),
  serviceText: $("serviceText"),
};

/* ------------------------------------------------------------ helpers */
function showAlert(message, kind = "error") {
  els.alert.textContent = message;
  els.alert.className = kind === "info" ? "alert info" : "alert";
  els.alert.hidden = false;
}
function clearAlert() {
  els.alert.hidden = true;
}
function titleCase(text) {
  return String(text || "").replace(/_/g, " ").toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ------------------------------------------------------ health check */
async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    const data = await response.json();
    const kb = `${data.knowledge_base_documents} docs / ${data.knowledge_base_chunks} chunks`;
    if (data.ollama_running && data.model_available) {
      els.serviceDot.className = "dot ok";
      els.serviceText.textContent = `${data.ollama_model} ready · ${kb}`;
    } else {
      els.serviceDot.className = "dot warn";
      els.serviceText.textContent = data.message;
      showAlert(data.message, "info");
    }
    if (data.knowledge_base_chunks === 0) {
      showAlert("The knowledge base is empty. Add .txt, .md or .pdf files to knowledge_base/ and restart the server.", "info");
    }
  } catch (error) {
    els.serviceDot.className = "dot bad";
    els.serviceText.textContent = "Backend unreachable";
    showAlert("Cannot reach the Cognify-AI backend. Start it with:  uvicorn backend.main:app --reload --port 8000");
  }
}

/* ---------------------------------------------------------- rendering */
function renderMeter(data) {
  const risk = data.risk_score;
  els.meterMarker.style.left = `${Math.min(100, Math.max(0, risk))}%`;
  els.meterValue.textContent = risk;

  let label = "LOW RISK", cls = "badge";
  if (data.status === "MEDIUM_RISK") { label = "MEDIUM RISK"; cls = "badge medium"; }
  if (data.status === "HIGH_RISK")   { label = "HIGH RISK";   cls = "badge high"; }
  els.riskBadge.textContent = label;
  els.riskBadge.className = cls;

  els.verificationStatus.textContent = `Verification: ${titleCase(data.verification_status)}`;
  els.confidenceValue.textContent = `${data.confidence_score}%`;
  els.riskValue.textContent = `${risk}%`;
  els.iterationsValue.textContent = data.iterations ?? 0;
  els.claimsValue.textContent = (data.claims || []).length;
}

function renderBreakdown(breakdown) {
  const rows = [
    ["Contradicted by evidence", breakdown.contradiction_risk, "40%"],
    ["Unsupported by evidence", breakdown.unsupported_risk, "30%"],
    ["Verifier uncertainty", breakdown.nli_uncertainty, "20%"],
    ["Model hedging", breakdown.model_uncertainty, "10%"],
  ];
  els.breakdown.innerHTML = rows.map(([label, value, weight]) => `
    <li>
      <div class="bar-head"><span>${label}</span><span>${Math.round(value * 100)}% · weight ${weight}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.round(value * 100)}%"></div></div>
    </li>`).join("");

  const contributions = breakdown.weighted_contribution || {};
  const parts = Object.entries(contributions)
    .map(([key, value]) => `${titleCase(key)} ${value}`)
    .join(" + ");
  els.formulaNote.textContent = `Points added to the score: ${parts}.`;
}

function renderClaims(evidence) {
  if (!evidence || evidence.length === 0) {
    els.claimList.innerHTML = `<p class="empty">No separate factual claims were extracted from this answer.</p>`;
    return;
  }
  els.claimList.innerHTML = evidence.map((item) => {
    const cls = item.status.toLowerCase();
    const evidenceText = item.best_evidence
      ? `<q>${item.best_evidence}</q><div class="claim-meta">Source: ${item.best_evidence_source} · retrieval similarity ${item.retrieval_score}</div>`
      : `<span class="empty">No sufficiently relevant passage was found in the knowledge base.</span>`;
    return `
      <div class="claim ${cls}">
        <div class="claim-top">
          <span class="claim-text">${item.claim}</span>
          <span class="status ${cls}">${item.status}</span>
        </div>
        <div class="claim-evidence">${evidenceText}</div>
        <div class="claim-meta">
          Verifier: ${item.verifier} · entailment ${item.entailment_score} ·
          contradiction ${item.contradiction_score} · neutral ${item.neutral_score}
        </div>
      </div>`;
  }).join("");
}

function renderHistory(history) {
  if (!history || history.length <= 1) {
    els.historyPanel.hidden = true;
    return;
  }
  els.historyPanel.hidden = false;
  els.history.innerHTML = history.map((step) => `
    <li>
      <strong>${step.iteration === 0 ? "Initial answer" : `Correction ${step.iteration}`}</strong> —
      risk ${step.risk_score}, confidence ${step.confidence_score},
      ${step.supported} supported / ${step.contradicted} contradicted / ${step.insufficient} unsupported
      <div class="trace-answer">${step.answer}</div>
    </li>`).join("");
}

function renderResult(data) {
  renderMeter(data);
  renderBreakdown(data.risk_breakdown);
  els.originalAnswer.textContent = data.original_answer;
  els.finalAnswer.textContent = data.final_answer;
  renderClaims(data.evidence);
  renderHistory(data.history);
  els.results.hidden = false;

  if (data.warnings && data.warnings.length) {
    showAlert(data.warnings.join(" "), "info");
  } else {
    clearAlert();
  }
}

/* ------------------------------------------------------------- submit */
async function run() {
  const query = els.query.value.trim();
  if (!query) {
    showAlert("Type a question first.");
    els.query.focus();
    return;
  }

  clearAlert();
  els.runBtn.disabled = true;
  els.progress.hidden = false;

  const started = Date.now();
  const timer = setInterval(() => {
    const seconds = Math.round((Date.now() - started) / 1000);
    els.progress.textContent = `Generating, verifying and correcting locally… ${seconds}s`;
  }, 500);
  els.progress.textContent = "Generating, verifying and correcting locally… 0s";

  const body = { query };
  if (els.maxIterations.value !== "") body.max_iterations = Number(els.maxIterations.value);

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      showAlert(detail || `Request failed with status ${response.status}.`);
      return;
    }
    renderResult(data);
  } catch (error) {
    showAlert("Cannot reach the backend. Make sure it is running:  uvicorn backend.main:app --reload --port 8000");
  } finally {
    clearInterval(timer);
    els.progress.hidden = true;
    els.runBtn.disabled = false;
  }
}

/* --------------------------------------------------------- listeners */
els.runBtn.addEventListener("click", run);
els.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) run();
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    els.query.value = chip.textContent.trim();
    els.query.focus();
  });
});

checkHealth();
