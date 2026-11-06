
(() => {
  const $ = id => document.getElementById(id);

  const state = {
    chats: [],
    activeChatId: null,
    history: [],
    busy: false,
    highlight: false,         // risk highlighting is OFF by default
    level: Number(localStorage.getItem("cognify.level")) || 2,   // 1 School, 2 College, 3 Postgraduate
    pending: [],              // files added with the + button, not sent yet
    user: null
  };

  const LEVEL_NAME = {1: "School", 2: "College", 3: "Postgraduate"};
  const MAX_FILES = 5, MAX_MB = 8;
  const ALLOWED = /\.(png|jpe?g|webp|gif|pdf|txt|md|csv|json)$/i;
  const IS_IMAGE = /\.(png|jpe?g|webp|gif)$/i;

  function toast(text) {
    const el = $("toast");
    el.textContent = text;
    el.classList.add("show");
    setTimeout(() => el.classList.remove("show"), 3200);
  }

  // ---------------- login ----------------
  let authMode = "login";

  function setAuthMode(mode) {
    authMode = mode;
    document.querySelectorAll(".auth-tab").forEach(b => b.classList.toggle("active", b.dataset.auth === mode));
    $("authSubmit").textContent = mode === "login" ? "Log in" : "Create account";
    $("authSub").textContent = mode === "login" ? "Log in to continue." : "Create your account to get started.";
    $("authHint").hidden = mode !== "signup";
    $("authPassword").autocomplete = mode === "login" ? "current-password" : "new-password";
    $("authError").textContent = "";
  }

  function showAuth() {
    state.user = null;
    $("authScreen").hidden = false;
    $("logoutBtn").hidden = true;
    $("authEmail").focus();
  }

  function setUser(email) {
    state.user = email;
    $("authScreen").hidden = true;
    $("profileName").textContent = email;
    $("profileSub").textContent = "Signed in";
    $("avatarLetter").textContent = (email || "C")[0].toUpperCase();
    $("logoutBtn").hidden = false;
  }

  async function authCall(path, body) {
    const r = await fetch("/api/auth/" + path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const d = data.detail;
      throw new Error(Array.isArray(d) ? "Please check your email and password (10+ characters to sign up)." : (d || "Request failed."));
    }
    return data;
  }

  async function submitAuth(e) {
    e.preventDefault();
    const email = $("authEmail").value.trim(), password = $("authPassword").value;
    $("authError").textContent = "";
    if (!email || !password) { $("authError").textContent = "Enter your email and password."; return; }
    $("authSubmit").disabled = true;
    try {
      if (authMode === "signup") await authCall("signup", {email, password});
      const data = await authCall("login", {email, password});
      $("authPassword").value = "";
      setUser(data.email);
    } catch (err) {
      $("authError").textContent = err.message;
    } finally {
      $("authSubmit").disabled = false;
    }
  }

  async function checkSession() {
    try {
      const r = await fetch("/api/auth/me");
      if (r.ok) { setUser((await r.json()).email); return; }
    } catch { /* server unreachable: fall through to the login screen */ }
    showAuth();
  }

  // ---------------- attachments (+ button) ----------------
  function readAsBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
      reader.onerror = () => reject(new Error("Could not read " + file.name));
      reader.readAsDataURL(file);
    });
  }

  function addFiles(fileList) {
    for (const file of fileList) {
      if (state.pending.length >= MAX_FILES) { toast(`You can attach up to ${MAX_FILES} files.`); break; }
      if (!ALLOWED.test(file.name)) { toast(`"${file.name}" is not supported. Use photos, PDF, TXT, MD, CSV or JSON.`); continue; }
      if (file.size > MAX_MB * 1024 * 1024) { toast(`"${file.name}" is larger than ${MAX_MB} MB.`); continue; }
      state.pending.push({file, name: file.name, image: IS_IMAGE.test(file.name),
                          thumb: IS_IMAGE.test(file.name) ? URL.createObjectURL(file) : ""});
    }
    renderChips();
  }

  function renderChips() {
    const host = $("attachChips");
    host.hidden = !state.pending.length;
    host.innerHTML = state.pending.map((a, i) => `
      <span class="chip">
        ${a.thumb ? `<img src="${a.thumb}" alt="">` : `<i>📄</i>`}
        <span class="chip-name">${esc(a.name)}</span>
        <button type="button" data-remove="${i}" aria-label="Remove ${esc(a.name)}">×</button>
      </span>`).join("");
  }

  function setLevel(n) {
    state.level = n;
    localStorage.setItem("cognify.level", String(n));
    document.querySelectorAll(".level-btn").forEach(b => {
      const on = Number(b.dataset.level) === n;
      b.classList.toggle("active", on);
      b.setAttribute("aria-checked", String(on));
    });
  }


  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);

  const pretty = value => {
    if (value === undefined || value === null || value === "") return "Not reported";
    return typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
  };

  function activeChat() {
    return state.chats.find(c => c.id === state.activeChatId) || null;
  }

  function newChat() {
    const chat = {
      id: String(Date.now()),
      title: "New conversation",
      messages: [],
      report: null
    };

    state.chats.unshift(chat);
    state.activeChatId = chat.id;
    renderRecent();
    renderMessages();
    renderReport(null);
    showView("chat");
    $("messageInput").focus();
  }

  function renderRecent(filter = "") {
    const chats = state.chats.filter(c =>
      c.title.toLowerCase().includes(filter.toLowerCase())
    );

    $("recentChats").innerHTML = chats.length
      ? chats.map(c => `
          <button class="recent-chat ${c.id === state.activeChatId ? "selected" : ""}" data-chat-id="${esc(c.id)}">
            ◌ &nbsp; ${esc(c.title)}
          </button>
        `).join("")
      : `<div class="empty-state">${filter ? "No matching conversations." : "Your conversations will appear here."}</div>`;
  }

  const RISK_LABEL = {
    LOW: "Low risk (green)",
    MEDIUM: "Medium risk (yellow)",
    HIGH: "High risk (red)"
  };
  const RISK_REASON = {
    SUPPORTED: "supported by evidence",
    INSUFFICIENT: "not enough evidence to confirm",
    CONTRADICTED: "evidence contradicts this"
  };

  // Answer text: plain when highlighting is OFF, coloured by risk when ON.
  function answerHtml(m) {
    const segments = Array.isArray(m.segments) ? m.segments : null;
    if (!state.highlight || !segments || !segments.length) return esc(m.content);

    return segments.map(s => {
      const level = String(s.level || "NONE").toUpperCase();
      if (!RISK_LABEL[level] || s.risk == null) return esc(s.text);
      const tip = `${RISK_LABEL[level]} - risk ${s.risk}% - ${RISK_REASON[s.status] || "unverified"}`;
      return `<mark class="risk-${level.toLowerCase()}" title="${esc(tip)}">${esc(s.text)}</mark>`;
    }).join("");
  }

  function riskLegend(m) {
    if (!state.highlight || !Array.isArray(m.segments) || !m.segments.length) return "";
    return `<div class="risk-legend">
      <span><i class="dot low"></i>30% Low</span>
      <span><i class="dot medium"></i>60% Medium</span>
      <span><i class="dot high"></i>80% High</span>
    </div>`;
  }

  function renderMessages() {
    const chat = activeChat();

    $("welcome").style.display =
      chat && chat.messages.length ? "none" : "block";

    $("messages").innerHTML = chat ? chat.messages.map((m, i) => {
      if (m.role === "loading") {
        return `<div class="message"><div class="message-avatar">✳</div><div class="message-content"><div class="message-label">COGNIFY-AI</div><div class="thinking">Verifying response...</div></div></div>`;
      }

      const user = m.role === "user";
      const files = user && Array.isArray(m.attachments) && m.attachments.length
        ? `<div class="msg-files">${m.attachments.map(a => `<span class="chip static">${a.thumb ? `<img src="${a.thumb}" alt="">` : "<i>📄</i>"}<span class="chip-name">${esc(a.name)}</span></span>`).join("")}</div>` : "";
      const levelTag = user && m.level ? `<span class="level-tag">Level ${m.level} · ${LEVEL_NAME[m.level]}</span>` : "";

      return `
        <article class="message ${user ? "user" : "assistant"}">
          <div class="message-avatar">${user ? "Y" : "✳"}</div>
          <div class="message-content">
            <div class="message-label">${user ? "YOU" : "COGNIFY-AI"} ${levelTag}</div>
            ${files}
            <div class="message-bubble">${user ? esc(m.content) : answerHtml(m)}</div>
            ${user ? "" : riskLegend(m)}
            ${!user && m.notes && m.notes.length ? `<div class="msg-notes">${m.notes.map(n => `<div>ⓘ ${esc(n)}</div>`).join("")}</div>` : ""}
            ${!user ? `<div class="message-tools"><button data-copy="${i}">Copy answer</button><button data-again="${i}">Ask again</button></div>` : ""}
          </div>
        </article>
      `;
    }).join("") : "";

    $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
  }

  function showView(name) {
    const ids = {
      chat: "chatView",
      knowledge: "knowledgeView",
      history: "historyView",
      settings: "settingsView"
    };

    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    $(ids[name] || ids.chat).classList.add("active");

    document.querySelectorAll(".nav-item").forEach(b =>
      b.classList.toggle("active", b.dataset.view === name)
    );

    $("pageTitle").textContent = ({
      chat: "Chat",
      knowledge: "Knowledge base",
      history: "Verification history",
      settings: "Settings"
    })[name] || "Chat";

    $("sidebar").classList.remove("mobile-open");

    if (name === "knowledge" || name === "settings") refreshHealth();
    if (name === "history") renderHistory();
  }

  function get(obj, keys) {
    if (!obj || typeof obj !== "object") return undefined;
    for (const key of keys) {
      if (obj[key] !== undefined && obj[key] !== null) return obj[key];
    }
    return undefined;
  }

  function arrayify(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") {
      return Object.entries(value).map(([key, val]) =>
        typeof val === "object" && val ? {label: key, ...val} : {label: key, value: val}
      );
    }
    return typeof value === "string" && value ? [value] : [];
  }

  function verdictResults(data) {
    const ev = data && data.evidence;
    return Array.isArray(ev) && ev.length && typeof ev[0] === "object" && ev[0] && "status" in ev[0] && "claim" in ev[0]
      ? ev : null;
  }

  function reportFrom(data) {
    const nested = get(data, ["verification", "verification_report", "report", "analysis", "metadata"]) || {};

    let answer = get(data, [
      "answer", "verified_answer", "response", "output",
      "final_answer", "corrected_answer", "message"
    ]);

    if (answer && typeof answer === "object") {
      answer = get(answer, ["text", "content", "answer", "output"]);
    }

    if (typeof answer !== "string") answer = "";

    return {
      answer,
      highlights: Array.isArray(data && data.highlights) ? data.highlights : [],
      original: get(data, ["original_answer", "initial_answer", "draft_answer"]) ??
                get(nested, ["original_answer", "initial_answer"]),
      risk: get(data, ["risk_score", "risk", "hallucination_risk"]) ??
            get(nested, ["risk_score", "risk"]),
      confidence: get(data, ["confidence", "confidence_score"]) ??
                  get(nested, ["confidence", "confidence_score"]),
      iterations: get(data, ["iterations", "correction_iterations", "rounds"]) ??
                  get(nested, ["iterations", "rounds"]),
      // This backend returns `claims` as plain strings and the verdicts in `evidence`
      // (one result per claim). Use the verdict objects so claims are classified.
      claims: verdictResults(data) ? verdictResults(data).map(r => ({
                claim: r.claim,
                status: r.status,
                reason: r.best_evidence
                  ? "Evidence: " + r.best_evidence.slice(0, 140) + (r.best_evidence.length > 140 ? "…" : "")
                  : "No supporting evidence found."
              }))
              : arrayify(get(data, ["claims", "claim_analysis", "claim_results"]) ??
                         get(nested, ["claims", "claim_analysis"])),
      evidence: verdictResults(data) ? verdictResults(data)
                  .flatMap(r => (r.evidence || []).map(e => ({title: e.source, text: e.text})))
                  .filter((e, i, all) => all.findIndex(x => x.text === e.text) === i)
              : arrayify(get(data, ["evidence", "retrieved_evidence", "sources", "context"]) ??
                         get(nested, ["evidence", "sources"])),
      history: get(data, ["correction_history", "corrections", "history"]) ??
               get(nested, ["correction_history", "corrections"]),
      status: get(data, ["verification_status", "verdict", "status"]) ??
              get(nested, ["verification_status", "verdict"]),
      raw: data
    };
  }

  function percent(value) {
    if (typeof value === "string") value = Number(value.replace("%", ""));
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    if (value >= 0 && value <= 1) value *= 100;
    return Math.round(Math.max(0, Math.min(100, value)));
  }

  function renderReport(r) {
    const host = $("reportBody");

    if (!r) {
      host.innerHTML = `
        <div class="report-empty">
          <div class="report-empty-icon">◈</div>
          <strong>Nothing to verify yet</strong>
          <p>Ask a question to see verification results and evidence here.</p>
        </div>`;
      return;
    }

    const risk = percent(r.risk);
    const confidence = percent(r.confidence);
    const claims = arrayify(r.claims);
    const evidence = arrayify(r.evidence);

    const counts = {supported: 0, contradicted: 0, unclear: 0};

    claims.forEach(c => {
      const verdict = typeof c === "string" ? c :
        String(get(c, ["verdict", "status", "label", "result", "classification"]) || "");

      if (/support|entailed|correct|verified|true/i.test(verdict)) counts.supported++;
      else if (/contradict|refuted|false|incorrect/i.test(verdict)) counts.contradicted++;
      else counts.unclear++;
    });

    const riskSection = risk === null
      ? `<div class="score-row"><div class="score-info"><strong>Risk score not reported</strong><small>Your backend did not include a risk score.</small></div></div>`
      : `<div class="score-row"><div class="score-ring" style="--score:${risk}%"><strong>${risk}%</strong></div><div class="score-info"><strong>${risk < 30 ? "Lower reported risk" : risk < 65 ? "Moderate reported risk" : "Higher reported risk"}</strong><small>Score reported by your verification backend.</small></div></div>`;

    const claimHtml = claims.length ? claims.map(c => {
      const claim = typeof c === "string" ? c :
        get(c, ["claim", "text", "statement", "content"]) || JSON.stringify(c);
      const verdict = typeof c === "string" ? "Not classified" :
        get(c, ["verdict", "status", "label", "result", "classification"]) || "Not classified";
      const reason = typeof c === "string" ? "" :
        get(c, ["reason", "explanation", "details", "justification"]) || "";

      return `<div class="detail-row"><span>${esc(claim)}${reason ? "<br>" + esc(reason) : ""}</span><strong>${esc(verdict)}</strong></div>`;
    }).join("") : "Claim-level results were not included.";

    const evidenceHtml = evidence.length ? evidence.map(e => {
      if (typeof e === "string") return `<div class="detail-row"><span>${esc(e)}</span></div>`;

      const title = get(e, ["title", "source", "name", "document", "filename"]) || "Retrieved passage";
      const content = get(e, ["text", "content", "snippet", "page_content", "excerpt"]) || JSON.stringify(e);

      return `<div class="detail-row"><span><strong>${esc(title)}</strong><br>${esc(content)}</span></div>`;
    }).join("") : "No retrieved evidence was included in this response.";

    const disclosure = (title, value) => `
      <details class="report-disclosure">
        <summary>${title}</summary>
        <div>${esc(pretty(value))}</div>
      </details>`;

    host.innerHTML = `
      <section class="report-section">
        <div class="report-section-head">
          <strong>Overall assessment</strong>
          <span>${esc(r.status || "Results received")}</span>
        </div>
        ${riskSection}
        ${confidence === null ? "" : `<div class="report-section-content">Confidence reported by backend: <strong>${confidence}%</strong></div>`}
      </section>

      <div class="mini-metrics">
        <div class="mini-metric"><strong>${claims.length ? counts.supported : "—"}</strong><small>Supported</small></div>
        <div class="mini-metric"><strong>${claims.length ? counts.contradicted : "—"}</strong><small>Contradicted</small></div>
        <div class="mini-metric"><strong>${claims.length ? counts.unclear : "—"}</strong><small>Unclear</small></div>
      </div>

      <section class="report-section">
        <div class="report-section-head"><strong>Claim analysis</strong><span>${claims.length} items</span></div>
        <div class="report-section-content">${claimHtml}</div>
      </section>

      <section class="report-section">
        <div class="report-section-head"><strong>Retrieved evidence</strong><span>${evidence.length} items</span></div>
        <div class="report-section-content">${evidenceHtml}</div>
      </section>

      ${disclosure("Original answer", r.original)}
      ${disclosure("Correction history", r.history)}
      ${disclosure("Raw verification response", r.raw)}

      <div class="report-section-content">Correction iterations: ${esc(r.iterations ?? "Not reported")}</div>`;
  }

  async function sendMessage(text) {
    let question = String(text || "").trim();
    if (!question && state.pending.length) question = "Please explain the attached file(s).";
    if (!question || state.busy) return;
  

    if (!activeChat()) newChat();

    const chat = activeChat();

    if (!chat.messages.length) {
      chat.title = question.length > 38 ? question.slice(0, 38) + "…" : question;
    }

    const sending = state.pending.slice();
    state.pending = [];
    renderChips();

    chat.messages.push(
      {role: "user", content: question, level: state.level,
       attachments: sending.map(a => ({name: a.name, thumb: a.thumb}))},
      {role: "loading", content: ""}
    );

    state.busy = true;
    $("sendBtn").disabled = true;
    $("messageInput").value = "";
    $("messageInput").style.height = "auto";
    renderRecent();
    renderMessages();

    try {
      const attachments = await Promise.all(
        sending.map(async a => ({name: a.name, data: await readAsBase64(a.file)}))
      );

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query: question, level: state.level, attachments})
      });

      if (response.status === 401) {          // session expired
        showAuth();
        throw new Error("Your session expired. Please log in again.");
      }

      const textResponse = await response.text();
      let data;

      try { data = JSON.parse(textResponse); }
      catch { data = textResponse; }

      if (!response.ok) {
        const detail = typeof data === "object" ?
          (data.detail || data.error || data.message) : data;
        throw new Error(detail || `Request failed (${response.status})`);
      }

      const report = reportFrom(data);

      if (!report.answer) {
        report.answer = typeof data === "string" ? data :
          get(data, ["detail", "text", "content"]) ||
          "The backend response did not contain a recognized answer field. Inspect the raw response in the verification report.";
      }

      chat.messages = chat.messages.filter(m => m.role !== "loading");
      chat.messages.push({role: "assistant", content: report.answer, segments: report.highlights,
        notes: Array.isArray(data.warnings)
          ? data.warnings.filter(w => /photo|attach|vision|PDF|readable text|knowledge base has no evidence|Correction was skipped/i.test(w)) : []});
      chat.report = report;

      renderReport(report);

      state.history.unshift({
        question,
        answer: report.answer,
        report,
        time: new Date()
      });

      renderHistory();
    } catch (error) {
      chat.messages = chat.messages.filter(m => m.role !== "loading");
      chat.messages.push({
        role: "assistant",
        content: "I couldn't complete this request.\n\n" +
          error.message +
          "\n\nCheck that the FastAPI backend is running, then try again."
      });
      renderReport(null);
    } finally {
      state.busy = false;
      $("sendBtn").disabled = false;
      renderMessages();
      renderRecent();
    }
  }

  async function refreshHealth() {
    try {
      const response = await fetch("/api/health", {cache: "no-store"});
      if (!response.ok) throw new Error("Health endpoint returned " + response.status);

      const h = await response.json();

      $("sidebarOrb").className = "status-orb online";
      $("sidebarStatus").textContent = "Backend connected";
      $("sidebarModel").textContent = h.model || h.ollama_model || "Local model available";

      $("kbDocs").textContent = h.knowledge_base_documents ?? "—";
      $("kbChunks").textContent = h.knowledge_base_chunks ?? "—";
      $("kbEmbedding").textContent = h.embedding_backend ?? "—";

      const details = [
        ["API status", h.status],
        ["Ollama running", h.ollama_running],
        ["Model available", h.model_available],
        ["Model", h.model || h.ollama_model],
        ["Knowledge-base documents", h.knowledge_base_documents],
        ["Knowledge-base chunks", h.knowledge_base_chunks],
        ["Embedding backend", h.embedding_backend],
        ["NLI backend", h.nli_backend]
      ];

      const html = details.map(([label, value]) =>
        `<div class="detail-row"><span>${esc(label)}</span><strong>${esc(pretty(value))}</strong></div>`
      ).join("");

      $("kbDetails").innerHTML = html;
      $("settingsDetails").innerHTML = html;
    } catch (error) {
      $("sidebarOrb").className = "status-orb offline";
      $("sidebarStatus").textContent = "Backend unavailable";
      $("sidebarModel").textContent = "Start FastAPI to connect";

      const html = `<div class="detail-row"><span>Connection</span><strong>Unavailable</strong></div>`;
      $("kbDetails").innerHTML = html;
      $("settingsDetails").innerHTML = html;
    }
  }

  function renderHistory() {
    $("historyList").innerHTML = state.history.length
      ? state.history.map((h, i) => `
          <button class="history-item" data-history="${i}">
            ◈ &nbsp; ${esc(h.question)}
            <br><small>${esc(h.time.toLocaleString())}</small>
          </button>
        `).join("")
      : `<div class="empty-state">No verifications yet. Ask a question in Chat to start.</div>`;
  }

  document.addEventListener("click", e => {
    const prompt = e.target.closest("[data-prompt]");
    if (prompt) {
      showView("chat");
      sendMessage(prompt.dataset.prompt);
      return;
    }

    const nav = e.target.closest("[data-view]");
    if (nav) {
      showView(nav.dataset.view);
      return;
    }

    const recent = e.target.closest("[data-chat-id]");
    if (recent) {
      state.activeChatId = recent.dataset.chatId;
      renderRecent();
      renderMessages();
      renderReport(activeChat()?.report || null);
      showView("chat");
      return;
    }

    const copy = e.target.closest("[data-copy]");
    if (copy) {
      const msg = activeChat()?.messages[Number(copy.dataset.copy)];
      if (msg) navigator.clipboard.writeText(msg.content);
      return;
    }

    const again = e.target.closest("[data-again]");
    if (again) {
      const last = [...(activeChat()?.messages || [])].reverse().find(m => m.role === "user");
      if (last) sendMessage(last.content);
      return;
    }

    const history = e.target.closest("[data-history]");
    if (history) {
      const item = state.history[Number(history.dataset.history)];
      if (item) {
        renderReport(item.report);
        showView("chat");
      }
    }
  });

  $("chatForm").addEventListener("submit", e => {
    e.preventDefault();
    sendMessage($("messageInput").value);
  });

  $("messageInput").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chatForm").requestSubmit();
    }
  });

  $("messageInput").addEventListener("input", () => {
    const el = $("messageInput");
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  });

  $("newChatBtn").addEventListener("click", newChat);
  $("searchChats").addEventListener("input", e => renderRecent(e.target.value));
  $("refreshHealth").addEventListener("click", refreshHealth);
  $("settingsRefresh").addEventListener("click", refreshHealth);
  $("riskToggle").addEventListener("change", e => {
    state.highlight = e.target.checked;
    e.target.setAttribute("aria-checked", String(state.highlight));
    renderMessages();
  });

  $("toggleReport").addEventListener("click", () => $("reportPanel").classList.toggle("open"));
  $("closeReport").addEventListener("click", () => $("reportPanel").classList.remove("open"));
  $("mobileMenu").addEventListener("click", () => $("sidebar").classList.toggle("mobile-open"));

  document.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      newChat();
    }
    if (e.key === "Escape") {
      $("sidebar").classList.remove("mobile-open");
      $("reportPanel").classList.remove("open");
    }
  });

  document.querySelectorAll(".auth-tab").forEach(b => b.addEventListener("click", () => setAuthMode(b.dataset.auth)));
  $("authForm").addEventListener("submit", submitAuth);
  $("logoutBtn").addEventListener("click", async () => {
    try { await fetch("/api/auth/logout", {method: "POST"}); } catch { /* ignore */ }
    state.chats = []; state.history = []; state.activeChatId = null;
    newChat(); renderReport(null);
    $("profileName").textContent = "Cognify workspace";
    $("profileSub").textContent = "Local environment";
    showAuth();
  });

  document.querySelectorAll(".level-btn").forEach(b => b.addEventListener("click", () => setLevel(Number(b.dataset.level))));
  $("attachBtn").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", e => { addFiles(e.target.files); e.target.value = ""; });
  $("attachChips").addEventListener("click", e => {
    const btn = e.target.closest("[data-remove]");
    if (!btn) return;
    const [removed] = state.pending.splice(Number(btn.dataset.remove), 1);
    if (removed && removed.thumb) URL.revokeObjectURL(removed.thumb);
    renderChips();
  });

  setAuthMode("login");
  setLevel(state.level in LEVEL_NAME ? state.level : 2);
  checkSession();
  newChat();
  refreshHealth();
})();
