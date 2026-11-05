const user = FB.requireAuth("user");
if (!user) throw new Error("auth");

const historyEl = document.getElementById("history");
const thread = document.getElementById("thread");
const empty = document.getElementById("empty");
const input = document.getElementById("input");
const send = document.getElementById("send");
const status = document.getElementById("status");
const clearance = document.getElementById("clearance");
const chatTitle = document.getElementById("chat-title");
const graphScope = document.getElementById("graph-scope");
const memoryDrawer = document.getElementById("memory-drawer");
const scrim = document.getElementById("scrim");
const sidebar = document.getElementById("sidebar");
const profilePanel = document.getElementById("profile-panel");
const profileToggle = document.getElementById("profile-toggle");

let conversations = [];
let activeId = null;
let busy = false;

function selectedGraphId() {
  const value = (graphScope && graphScope.value) || "";
  return value || null;
}

const email = user.email || user.id || "user";
document.getElementById("user-email").textContent = email;
document.getElementById("user-role").textContent = user.role || "user";
document.getElementById("user-avatar").textContent = String(email).trim().charAt(0).toUpperCase() || "U";
if (user.clearance) clearance.value = user.clearance;

document.getElementById("logout").addEventListener("click", () => FB.logout());
document.getElementById("new-chat").addEventListener("click", () => {
  activeId = null;
  resetThread();
  closeMobileSidebar();
});
document.getElementById("open-memory").addEventListener("click", openMemory);
document.getElementById("close-memory").addEventListener("click", closeMemory);
profileToggle.addEventListener("click", () => {
  const open = profilePanel.hasAttribute("hidden");
  if (open) {
    profilePanel.removeAttribute("hidden");
    profileToggle.setAttribute("aria-expanded", "true");
  } else {
    profilePanel.setAttribute("hidden", "");
    profileToggle.setAttribute("aria-expanded", "false");
  }
});

scrim.addEventListener("click", () => {
  closeMemory();
  closeMobileSidebar();
});
document.getElementById("menu").addEventListener("click", () => {
  sidebar.classList.add("open");
  scrim.classList.add("show");
});

document.querySelectorAll("[data-prompt]").forEach((btn) => {
  btn.addEventListener("click", () => {
    input.value = btn.getAttribute("data-prompt");
    document.getElementById("composer").requestSubmit();
  });
});

document.getElementById("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || busy) return;
  await ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("composer").requestSubmit();
  }
});

document.getElementById("memory-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = document.getElementById("memory-input").value.trim();
  if (!content) return;
  try {
    await FB.request("/v1/memory", {
      method: "POST",
      body: JSON.stringify({ content }),
      clearance: clearance.value,
    });
    document.getElementById("memory-input").value = "";
    await loadMemories();
  } catch (error) {
    alert(error.message);
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

function closeMobileSidebar() {
  sidebar.classList.remove("open");
  if (!memoryDrawer.classList.contains("open")) scrim.classList.remove("show");
}

function setStatus(text, live) {
  status.textContent = text;
  status.classList.toggle("live", Boolean(live));
}

function renderSessions() {
  historyEl.replaceChildren();
  if (!conversations.length) {
    const note = document.createElement("p");
    note.className = "history-empty";
    note.textContent = "No chats yet — start one below.";
    historyEl.appendChild(note);
    return;
  }
  conversations.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "session" + (item.id === activeId ? " active" : "");
    row.innerHTML =
      '<div class="session-title"></div><div class="session-meta"></div><span class="session-delete" aria-label="Delete">×</span>';
    row.querySelector(".session-title").textContent = item.title || "New chat";
    row.querySelector(".session-meta").textContent = formatWhen(item.updated_at);
    row.addEventListener("click", (event) => {
      if (event.target.classList.contains("session-delete")) return;
      openConversation(item.id);
      closeMobileSidebar();
    });
    row.querySelector(".session-delete").addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!confirm("Delete this chat?")) return;
      await FB.request("/v1/conversations/" + item.id, {
        method: "DELETE",
        clearance: clearance.value,
      });
      if (activeId === item.id) {
        activeId = null;
        resetThread();
      }
      await loadConversations();
    });
    historyEl.appendChild(row);
  });
}

function formatWhen(value) {
  try {
    return new Date(value).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function resetThread() {
  thread.replaceChildren(empty);
  empty.hidden = false;
  chatTitle.textContent = "New chat";
}

function ensureThreadReady() {
  empty.hidden = true;
  if (empty.parentElement === thread) empty.remove();
}

function setAnswerContent(body, text, { streaming = false } = {}) {
  const value = text || "";
  if (!streaming && value && globalThis.FBMarkdown) {
    body.classList.add("md");
    FBMarkdown.paint(body, value);
    return;
  }
  body.classList.remove("md");
  body.textContent = value;
}

function addMessage(role, content, meta) {
  ensureThreadReady();
  const row = document.createElement("div");
  row.className = "msg-row " + role;

  if (role === "user") {
    const bubble = document.createElement("div");
    bubble.className = "bubble user";
    const who = document.createElement("div");
    who.className = "who";
    who.textContent = "You";
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = content || "";
    bubble.append(who, body);
    row.appendChild(bubble);
    thread.appendChild(row);
    thread.scrollTop = thread.scrollHeight;
    return { row, body, bubble };
  }

  // Assistant stack order: Thinking → Reasoning → Response → Sources
  const stack = document.createElement("div");
  stack.className = "answer-stack";

  const thinkingPanel = makeTrace("thinking", "Thinking", { open: true });
  const reasoningPanel = makeTrace("reasoning", "Reasoning", { open: true });

  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = "Fashion Brain";
  const body = document.createElement("div");
  body.className = "body response-body";
  if (content) setAnswerContent(body, content);
  bubble.append(who, body);

  const sources = makeSourcesPanel();

  stack.append(thinkingPanel, reasoningPanel, bubble, sources);
  row.appendChild(stack);
  thread.appendChild(row);

  if (meta) {
    if (meta.thinking) appendTraceText(thinkingPanel, meta.thinking);
    if (meta.reasoning) appendTraceText(reasoningPanel, meta.reasoning);
    if (content) setAnswerContent(body, content);
    paintSources(sources, meta, { open: false });
    collapseTraces(row);
  }

  thread.scrollTop = thread.scrollHeight;
  return { row, body, bubble, stack, thinkingPanel, reasoningPanel, sources };
}

function makeTrace(kind, label, { open = true } = {}) {
  const panel = document.createElement("div");
  panel.className = "trace" + (open ? " streaming" : "");
  panel.dataset.kind = kind;
  panel.hidden = true;
  const head = document.createElement("button");
  head.type = "button";
  head.className = "trace-toggle";
  head.textContent = open ? label : label + " (hidden)";
  const body = document.createElement("div");
  body.className = "trace-body" + (open ? " open" : "");
  head.addEventListener("click", () => {
    const isOpen = body.classList.toggle("open");
    head.textContent = isOpen ? label : label + " (hidden)";
  });
  panel.append(head, body);
  return panel;
}

function appendTraceText(panel, text) {
  if (!panel || !text) return;
  panel.hidden = false;
  const body = panel.querySelector(".trace-body");
  if (!body) return;
  body.textContent += text;
  thread.scrollTop = thread.scrollHeight;
}

function ensureTrace(row, kind, label, { open = true } = {}) {
  let panel = row.querySelector(".trace[data-kind='" + kind + "']");
  if (!panel) {
    panel = makeTrace(kind, label, { open });
    const stack = row.querySelector(".answer-stack");
    const bubble = row.querySelector(".bubble.assistant");
    if (stack && kind === "thinking") {
      stack.insertBefore(panel, stack.firstChild);
    } else if (stack && kind === "reasoning") {
      const thinking = stack.querySelector(".trace[data-kind='thinking']");
      if (thinking) thinking.after(panel);
      else if (bubble) stack.insertBefore(panel, bubble);
      else stack.appendChild(panel);
    } else if (bubble) {
      bubble.before(panel);
    } else {
      row.appendChild(panel);
    }
  }
  panel.hidden = false;
  const head = panel.querySelector(".trace-toggle");
  const body = panel.querySelector(".trace-body");
  if (open && body && !body.classList.contains("open")) {
    body.classList.add("open");
    if (head) head.textContent = label;
  }
  return body;
}

function appendTrace(row, kind, label, text, { open = true } = {}) {
  if (!text) return;
  const body = ensureTrace(row, kind, label, { open });
  body.textContent += text;
  thread.scrollTop = thread.scrollHeight;
}

function collapseTraces(row) {
  row.querySelectorAll(".trace").forEach((panel) => {
    const body = panel.querySelector(".trace-body");
    const head = panel.querySelector(".trace-toggle");
    if (!body || !head || panel.hidden) return;
    body.classList.remove("open");
    panel.classList.remove("streaming");
    const label = panel.dataset.kind === "reasoning" ? "Reasoning" : "Thinking";
    head.textContent = label + " (hidden)";
  });
}

function makeSourcesPanel() {
  const wrap = document.createElement("div");
  wrap.className = "sources-panel";
  wrap.hidden = true;

  const head = document.createElement("button");
  head.type = "button";
  head.className = "trace-toggle sources-toggle";
  head.textContent = "Sources / References";

  const body = document.createElement("div");
  body.className = "sources-body open";

  const list = document.createElement("div");
  list.className = "citations";
  body.appendChild(list);

  head.addEventListener("click", () => {
    const open = body.classList.toggle("open");
    head.textContent = open ? "Sources / References" : "Sources / References (hidden)";
  });

  wrap.append(head, body);
  return wrap;
}

function ensureSourcesPanel(row) {
  let panel = row.querySelector(".sources-panel");
  if (panel) return panel;
  panel = makeSourcesPanel();
  const stack = row.querySelector(".answer-stack");
  if (stack) stack.appendChild(panel);
  else row.appendChild(panel);
  return panel;
}

function sourceKey(item) {
  return [
    item.url || "",
    item.record_id || "",
    item.source_id || "",
    item.graph || "",
    item.title || "",
  ].join("|");
}

function appendSourceCard(panel, item) {
  if (!panel || !item) return;
  const list = panel.querySelector(".citations");
  if (!list) return;
  const key = sourceKey(item);
  if ([...list.children].some((node) => node.dataset.key === key)) return;

  panel.hidden = false;
  panel.classList.add("streaming");

  const url = String(item.url || "").trim();
  const title = String(item.title || item.record_id || item.source_id || "Reference").trim();
  const excerpt = String(item.excerpt || "").trim();
  const metaBits = [item.graph || item.source_id, item.record_id]
    .map((bit) => String(bit || "").trim())
    .filter(Boolean);

  const card = url ? document.createElement("a") : document.createElement("div");
  card.className = "citation streaming-in";
  card.dataset.key = key;
  if (url) {
    card.href = url;
    card.target = "_blank";
    card.rel = "noopener noreferrer";
  }
  card.innerHTML =
    '<div class="cite-title"></div>' +
    (url || metaBits.length ? '<div class="cite-url"></div>' : "") +
    (excerpt ? '<div class="cite-excerpt"></div>' : "");
  card.querySelector(".cite-title").textContent = title;
  const urlEl = card.querySelector(".cite-url");
  if (urlEl) urlEl.textContent = url || metaBits.join(" · ");
  if (excerpt) card.querySelector(".cite-excerpt").textContent = excerpt;
  list.appendChild(card);
  requestAnimationFrame(() => card.classList.add("visible"));
  thread.scrollTop = thread.scrollHeight;
}

function paintSources(panel, meta, { open = true } = {}) {
  if (!panel || !meta) return;
  const refs = normalizeReferences(meta);
  const caveats = Array.isArray(meta.caveats) ? meta.caveats.filter(Boolean) : [];
  const routes = Array.isArray(meta.routes) ? meta.routes : [];
  if (!refs.length && !caveats.length && !routes.length) {
    panel.hidden = true;
    return;
  }

  panel.hidden = false;
  panel.classList.remove("streaming");
  const body = panel.querySelector(".sources-body");
  const head = panel.querySelector(".sources-toggle");
  const list = panel.querySelector(".citations");
  if (!body || !list) return;
  list.replaceChildren();
  body.querySelectorAll(".sources-meta").forEach((node) => node.remove());

  refs.forEach((item) => appendSourceCard(panel, item));

  if (routes.length || caveats.length) {
    const details = document.createElement("div");
    details.className = "sources-meta";
    if (routes.length) {
      const p = document.createElement("p");
      p.className = "caveat";
      p.textContent =
        "Routes: " +
        routes.map((route) => (route === "catalog" ? "plm" : route)).join(" · ");
      details.appendChild(p);
    }
    caveats.forEach((text) => {
      const p = document.createElement("p");
      p.className = "caveat";
      p.textContent = text;
      details.appendChild(p);
    });
    body.appendChild(details);
  }

  if (open) {
    body.classList.add("open");
    if (head) head.textContent = "Sources / References";
  } else {
    body.classList.remove("open");
    if (head) head.textContent = "Sources / References (hidden)";
  }
}

function normalizeReferences(meta) {
  const fromMeta = Array.isArray(meta.url_citations) ? meta.url_citations : [];
  const fromEvidence = Array.isArray(meta.evidence) ? meta.evidence : [];
  const items = [];
  const seen = new Set();

  const push = (raw) => {
    if (!raw) return;
    const item = {
      url: String(raw.url || "").trim(),
      title: String(raw.title || raw.record_id || raw.source_id || "Reference").trim(),
      excerpt: String(raw.excerpt || "").trim(),
      source_id: String(raw.source_id || "").trim(),
      record_id: String(raw.record_id || "").trim(),
      graph: String(raw.graph || "").trim(),
    };
    if (!item.url && !item.title && !item.excerpt && !item.record_id) return;
    const key = sourceKey(item);
    if (seen.has(key)) return;
    seen.add(key);
    items.push(item);
  };

  fromMeta.forEach(push);
  fromEvidence.forEach(push);
  return items;
}

function showThinkingPlaceholder(ui) {
  if (!ui) return;
  ensureTrace(ui.row, "thinking", "Thinking", { open: true });
  const panel = ui.row.querySelector(".trace[data-kind='thinking']");
  if (!panel) return;
  panel.classList.add("streaming");
  const body = panel.querySelector(".trace-body");
  if (body && !body.textContent) {
    const el = document.createElement("span");
    el.className = "thinking";
    el.innerHTML = '<span class="thinking-word">Thinking</span><span class="thinking-dots"></span>';
    body.appendChild(el);
  }
  if (ui.body) {
    ui.body.classList.remove("md");
    ui.body.innerHTML =
      '<span class="response-placeholder">Waiting for response…</span>';
  }
}

function clearThinkingPlaceholder(row) {
  const panel = row.querySelector(".trace[data-kind='thinking']");
  if (!panel) return;
  const body = panel.querySelector(".trace-body");
  if (!body) return;
  const placeholder = body.querySelector(".thinking");
  if (placeholder && !body.textContent.replace(/\s/g, "").replace("Thinking", "")) {
    body.replaceChildren();
  } else if (placeholder) {
    placeholder.remove();
  }
}

function wsAskUrl() {
  const http = FB.apiBase();
  const ws = http.replace(/^http/i, "ws");
  const params = new URLSearchParams();
  const token = FB.getAccess();
  if (token) params.set("token", token);
  if (clearance && clearance.value) params.set("clearance", clearance.value);
  return ws + "/ws/ask?" + params.toString();
}

function askOverSocket(payload) {
  return new Promise((resolve, reject) => {
    let socket;
    try {
      socket = new WebSocket(wsAskUrl());
    } catch (error) {
      reject(error);
      return;
    }
    const result = {
      answer: "",
      thinking: "",
      reasoning: "",
      evidence: [],
      meta: {},
      answerObj: {},
    };
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      try {
        socket.close();
      } catch {
        /* ignore */
      }
      fn(value);
    };
    const ui = payload._ui;
    socket.addEventListener("open", () => {
      socket.send(
        JSON.stringify({
          question: payload.question,
          conversation_id: payload.conversation_id || null,
          graph_id: payload.graph_id || null,
        })
      );
    });
    socket.addEventListener("message", (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      if (data.error) {
        finish(reject, new Error(data.message || data.error));
        return;
      }
      const type = data.type;
      const text = data.text || "";
      if (type === "thinking") {
        clearThinkingPlaceholder(ui && ui.row);
        result.thinking += text;
        if (ui) {
          setStatus("Thinking…", true);
          appendTrace(ui.row, "thinking", "Thinking", text, { open: true });
        }
      } else if (type === "reasoning") {
        result.reasoning += text;
        if (ui) {
          setStatus("Reasoning…", true);
          appendTrace(ui.row, "reasoning", "Reasoning", text, { open: true });
        }
      } else if (type === "answer") {
        result.answer += text;
        if (ui) {
          setStatus("Responding…", true);
          const placeholder = ui.body.querySelector(".response-placeholder");
          if (placeholder) ui.body.replaceChildren();
          setAnswerContent(ui.body, result.answer, { streaming: true });
          thread.scrollTop = thread.scrollHeight;
        }
      } else if (type === "source") {
        const item = data.item || {};
        result.evidence.push(item);
        if (ui) {
          setStatus("Sources…", true);
          const panel = ensureSourcesPanel(ui.row);
          appendSourceCard(panel, item);
        }
      } else if (type === "done") {
        result.answerObj = data.answer || {};
        if (!result.answer && result.answerObj.answer) result.answer = result.answerObj.answer;
        const doneEvidence = result.answerObj.evidence || [];
        if (!result.evidence.length && doneEvidence.length) {
          result.evidence = doneEvidence;
          if (ui) {
            const panel = ensureSourcesPanel(ui.row);
            doneEvidence.forEach((item) => appendSourceCard(panel, item));
          }
        }
        result.meta = {
          routes: result.answerObj.routes || [],
          evidence: result.evidence.length ? result.evidence : doneEvidence,
          caveats: result.answerObj.caveats || [],
          thinking: result.thinking,
          reasoning: result.reasoning,
        };
        finish(resolve, result);
      }
    });
    socket.addEventListener("error", () => finish(reject, new Error("Live stream failed. Is Fashion Brain running?")));
    socket.addEventListener("close", () => {
      if (!settled) finish(reject, new Error("Connection closed before the answer finished."));
    });
  });
}

async function loadConversations() {
  const data = await FB.request("/v1/conversations", { clearance: clearance.value });
  conversations = data.conversations || [];
  renderSessions();
}

async function loadGraphs() {
  if (!graphScope) return;
  try {
    const data = await FB.request("/v1/graphs", { clearance: clearance.value });
    const graphs = (data.graphs || []).filter(
      (g) => g.status !== "deleted" && g.kind !== "interconnect" && g.id !== "shared" && g.id !== "master"
    );
    const current = graphScope.value;
    graphScope.replaceChildren();
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "All graphs";
    graphScope.appendChild(all);
    graphs.forEach((g) => {
      const opt = document.createElement("option");
      opt.value = g.id;
      opt.textContent = g.name || g.id;
      graphScope.appendChild(opt);
    });
    if ([...graphScope.options].some((o) => o.value === current)) graphScope.value = current;
  } catch (_err) {
    /* KG may be down; keep All graphs */
  }
}

async function createConversation(title) {
  const body = { title: title || "New chat" };
  const graphId = selectedGraphId();
  if (graphId) body.graph_id = graphId;
  const item = await FB.request("/v1/conversations", {
    method: "POST",
    body: JSON.stringify(body),
    clearance: clearance.value,
  });
  activeId = item.id;
  if (graphScope && item.graph_id) graphScope.value = item.graph_id;
  await loadConversations();
  resetThread();
  chatTitle.textContent = item.title || "New chat";
  return item;
}

async function openConversation(id) {
  const data = await FB.request("/v1/conversations/" + id, { clearance: clearance.value });
  activeId = id;
  chatTitle.textContent = data.title || "Chat";
  if (graphScope) graphScope.value = data.graph_id || "";
  thread.replaceChildren();
  empty.hidden = true;
  const messages = data.messages || [];
  if (!messages.length) {
    thread.appendChild(empty);
    empty.hidden = false;
  } else {
    messages.forEach((message) => {
      const meta = message.meta || {};
      if (message.role === "assistant") {
        addMessage("assistant", message.content, meta);
      } else {
        addMessage("user", message.content, meta);
      }
    });
  }
  renderSessions();
}

async function ask(question) {
  busy = true;
  send.disabled = true;
  input.value = "";
  input.style.height = "auto";
  try {
    if (!activeId) await createConversation(question.slice(0, 48));
    addMessage("user", question);
    const assistant = addMessage("assistant", "");
    showThinkingPlaceholder(assistant);
    setStatus("Thinking…", true);

    const payload = {
      question,
      conversation_id: activeId,
      _ui: assistant,
    };
    const graphId = selectedGraphId();
    if (graphId) payload.graph_id = graphId;

    let streamed;
    try {
      streamed = await askOverSocket(payload);
    } catch (streamError) {
      const data = await FB.request("/v1/conversations/" + activeId + "/messages", {
        method: "POST",
        body: JSON.stringify({ question, graph_id: graphId || null }),
        clearance: clearance.value,
      });
      const answerObj = data.answer || {};
      const fallback =
        answerObj.answer || (data.message && data.message.content) || "(No answer)";
      const meta = Object.assign({}, (data.message && data.message.meta) || {}, {
        routes: answerObj.routes || [],
        evidence: answerObj.evidence || [],
        caveats: answerObj.caveats || [],
        thinking: (data.message && data.message.meta && data.message.meta.thinking) || "",
        reasoning: (data.message && data.message.meta && data.message.meta.reasoning) || "",
      });
      clearThinkingPlaceholder(assistant.row);
      if (meta.thinking) appendTrace(assistant.row, "thinking", "Thinking", meta.thinking, { open: false });
      if (meta.reasoning) appendTrace(assistant.row, "reasoning", "Reasoning", meta.reasoning, { open: false });
      setAnswerContent(assistant.body, fallback);
      paintSources(ensureSourcesPanel(assistant.row), meta, { open: true });
      collapseTraces(assistant.row);
      streamed = { answer: fallback, answerObj, meta, evidence: meta.evidence || [] };
    }

    const finalAnswer = streamed.answer || streamed.answerObj.answer || "(No answer)";
    setAnswerContent(assistant.body, finalAnswer);

    const meta = Object.assign({}, streamed.meta || {}, {
      routes: (streamed.answerObj && streamed.answerObj.routes) || streamed.meta.routes || [],
      evidence:
        (streamed.evidence && streamed.evidence.length
          ? streamed.evidence
          : null) ||
        (streamed.answerObj && streamed.answerObj.evidence) ||
        streamed.meta.evidence ||
        [],
      caveats: (streamed.answerObj && streamed.answerObj.caveats) || streamed.meta.caveats || [],
      thinking: streamed.thinking || streamed.meta.thinking || "",
      reasoning: streamed.reasoning || streamed.meta.reasoning || "",
    });

    const thinkingBody = assistant.row.querySelector(".trace[data-kind='thinking'] .trace-body");
    const reasoningBody = assistant.row.querySelector(".trace[data-kind='reasoning'] .trace-body");
    if (meta.thinking && thinkingBody && !thinkingBody.textContent.trim()) {
      appendTrace(assistant.row, "thinking", "Thinking", meta.thinking, { open: false });
    }
    if (meta.reasoning && reasoningBody && !reasoningBody.textContent.trim()) {
      appendTrace(assistant.row, "reasoning", "Reasoning", meta.reasoning, { open: false });
    }

    paintSources(ensureSourcesPanel(assistant.row), meta, { open: true });
    collapseTraces(assistant.row);
    assistant.row.querySelectorAll(".trace").forEach((panel) => {
      const body = panel.querySelector(".trace-body");
      if (body && !body.textContent.trim()) panel.hidden = true;
    });

    setStatus("Online", true);
    await loadConversations();
    const active = conversations.find((item) => item.id === activeId);
    if (active && active.title) chatTitle.textContent = active.title;
  } catch (error) {
    addMessage("assistant", "Error: " + error.message);
    setStatus("Error", false);
  } finally {
    busy = false;
    send.disabled = false;
    input.focus();
  }
}

async function loadMemories() {
  const data = await FB.request("/v1/memory", { clearance: clearance.value });
  const list = document.getElementById("memory-list");
  list.replaceChildren();
  (data.memories || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "memory-item";
    const text = document.createElement("div");
    text.textContent = item.content;
    const del = document.createElement("button");
    del.type = "button";
    del.className = "ghost";
    del.style.marginTop = "8px";
    del.textContent = "Delete";
    del.addEventListener("click", async () => {
      await FB.request("/v1/memory/" + item.id, {
        method: "DELETE",
        clearance: clearance.value,
      });
      await loadMemories();
    });
    row.append(text, del);
    list.appendChild(row);
  });
}

function openMemory() {
  memoryDrawer.classList.add("open");
  scrim.classList.add("show");
  loadMemories().catch((error) => alert(error.message));
}

function closeMemory() {
  memoryDrawer.classList.remove("open");
  if (!sidebar.classList.contains("open")) scrim.classList.remove("show");
}

(async function boot() {
  try {
    await FB.request("/health", { noAuth: true });
    setStatus("Online", true);
    await Promise.all([loadConversations(), loadGraphs()]);
  } catch (error) {
    setStatus("Offline", false);
  }
})();
