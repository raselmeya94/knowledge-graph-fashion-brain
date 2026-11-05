const user = FB.requireAuth("admin");
if (!user) throw new Error("auth");
if (user.role !== "admin") {
  location.href = "/user/";
  throw new Error("admin only");
}

const adminEmail = user.email || user.id || "admin";
document.getElementById("admin-email").textContent = adminEmail;
document.getElementById("admin-role").textContent = user.role;
document.getElementById("admin-avatar").textContent = String(adminEmail).trim().charAt(0).toUpperCase() || "A";
document.getElementById("logout").addEventListener("click", () => FB.logout());

const views = ["overview", "ingest", "pipeline", "reviews", "query"];
let selectedReviewId = null;
const scrape = {
  file_id: "",
  review_id: "",
  maxDone: 0,
  extract: null,
  chunks: null,
  propose: null,
  queue: null,
  approve: null,
  commit: null,
};

document.querySelectorAll(".nav-btn[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => showView(btn.getAttribute("data-view")));
});
document.querySelectorAll(".linkish[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => showView(btn.getAttribute("data-view")));
});

const jobPollers = new Map();

function prettyJob(job) {
  if (!job) return "—";
  const bits = [job.kind || "job", job.status || "?", job.id ? "#" + job.id : ""];
  if (job.message) bits.push("· " + job.message);
  if (job.error) bits.push("· " + job.error);
  return bits.filter(Boolean).join(" ");
}

function stopJobPoll(key) {
  const timer = jobPollers.get(key);
  if (timer) {
    clearInterval(timer);
    jobPollers.delete(key);
  }
}

async function pollJobUntilDone(jobId, { msgEl, outEl, onDone, key } = {}) {
  const pollKey = key || jobId;
  stopJobPoll(pollKey);
  const tick = async () => {
    try {
      const data = await FB.request("/v1/admin/jobs/" + encodeURIComponent(jobId));
      const job = data.job || data;
      if (outEl) outEl.textContent = pretty(data);
      if (msgEl) {
        msgEl.textContent = prettyJob(job);
        msgEl.className = job.status === "failed" ? "msg" : job.status === "done" ? "msg ok" : "msg";
      }
      if (job.status === "done" || job.status === "failed" || job.status === "cancelled") {
        stopJobPoll(pollKey);
        if (onDone) await onDone(job);
      }
    } catch (error) {
      if (msgEl) {
        msgEl.className = "msg";
        msgEl.textContent = error.message || "Job poll failed";
      }
      stopJobPoll(pollKey);
    }
  };
  await tick();
  jobPollers.set(pollKey, setInterval(tick, 3000));
}

async function startBackgroundJob(path, body, { msgEl, outEl, onDone, key } = {}) {
  if (msgEl) {
    msgEl.textContent = "Starting job…";
    msgEl.className = "msg";
  }
  const data = await FB.request(path, {
    method: "POST",
    body: JSON.stringify(body == null ? {} : body),
  });
  if (outEl) outEl.textContent = pretty(data);
  const job = data.job || (data.id ? data : null);
  if (!job || !job.id) {
    if (msgEl) {
      msgEl.className = "msg ok";
      msgEl.textContent = "Done (no job id).";
    }
    if (onDone) await onDone(data);
    return data;
  }
  if (msgEl) msgEl.textContent = prettyJob(job) + " — polling…";
  await pollJobUntilDone(job.id, { msgEl, outEl, onDone, key: key || job.id });
  return data;
}

document.getElementById("btn-reload-graphs")?.addEventListener("click", async () => {
  const msg = document.getElementById("reload-msg");
  try {
    await startBackgroundJob("/v1/admin/graphs/reload", {}, {
      msgEl: msg,
      key: "reload",
      onDone: async (job) => {
        if (job.status === "done") await loadOverview();
      },
    });
  } catch (error) {
    msg.className = "msg";
    msg.textContent = error.message || "Reload failed";
  }
});

function showView(name) {
  views.forEach((view) => {
    document.getElementById("view-" + view).classList.toggle("hidden", view !== name);
  });
  document.querySelectorAll(".nav-btn[data-view]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-view") === name);
  });
  if (name === "reviews") loadReviews().catch(showErr);
  if (name === "overview") loadOverview().catch(showErr);
  if (name === "ingest") loadIngestStatus().catch(showErr);
  if (name === "pipeline") {
    fillGraphSelects().catch(showErr);
    fillOntologySelects().catch(showErr);
    gotoStep(Math.max(1, scrape.maxDone || 1));
  }
}

function showErr(error) {
  console.error(error);
  alert(error.message || String(error));
}

function pretty(data) {
  return JSON.stringify(data, null, 2);
}

function setMsg(id, text, ok) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = ok ? "msg ok" : "msg";
  el.textContent = text || "";
}

function markDone(step) {
  scrape.maxDone = Math.max(scrape.maxDone, step);
  document.querySelectorAll(".step-tab").forEach((tab) => {
    const n = Number(tab.getAttribute("data-step"));
    tab.classList.toggle("done", n < scrape.maxDone || (n === scrape.maxDone && n < 5));
  });
  const nextBtn = document.getElementById("btn-scrape-next-" + step);
  if (nextBtn) nextBtn.classList.remove("hidden");
}

function gotoStep(step) {
  const n = Number(step);
  if (n > scrape.maxDone + 1 && n > 1) {
    setMsg("scrape-msg-" + Math.min(scrape.maxDone || 1, 5), "Complete the previous step first.", false);
    return;
  }
  document.querySelectorAll(".step-tab").forEach((tab) => {
    tab.classList.toggle("active", Number(tab.getAttribute("data-step")) === n);
  });
  document.querySelectorAll(".step-panel").forEach((panel) => {
    panel.classList.toggle("active", Number(panel.getAttribute("data-step-panel")) === n);
  });
  if (n === 5 && scrape.review_id) {
    refreshScrapeReview().catch(() => {});
  }
}

document.querySelectorAll(".step-tab").forEach((tab) => {
  tab.addEventListener("click", () => gotoStep(tab.getAttribute("data-step")));
});
document.querySelectorAll("[data-goto]").forEach((btn) => {
  btn.addEventListener("click", () => gotoStep(btn.getAttribute("data-goto")));
});

document.getElementById("scrape-commit-mode").addEventListener("change", () => {
  const mode = document.getElementById("scrape-commit-mode").value;
  document.getElementById("scrape-commit-existing").classList.toggle("hidden", mode !== "existing");
  document.getElementById("scrape-commit-new").classList.toggle("hidden", mode !== "new");
});

async function fillGraphSelects() {
  const graphs = await FB.request("/v1/admin/graphs");
  const options = (graphs.graphs || [])
    .filter((g) => g.id && g.id !== "shared" && g.id !== "master")
    .map((g) => ({ id: g.id, label: g.name || g.id, ontology_id: g.ontology_id || "" }));
  ["scrape-graph", "scrape-commit-graph", "q-graph"].forEach((selectId) => {
    const select = document.getElementById(selectId);
    if (!select) return;
    const current = select.value;
    if (!options.length) return;
    select.innerHTML = options.map((g) => `<option value="${escapeAttr(g.id)}">${escapeHtml(g.label)}</option>`).join("");
    if (current && options.some((g) => g.id === current)) select.value = current;
    else if (options.some((g) => g.id === "news")) select.value = "news";
  });
  return options;
}

async function fillOntologySelects() {
  let options = [];
  try {
    const data = await FB.request("/v1/admin/ontologies");
    options = (data.ontologies || data || [])
      .map((o) => ({ id: o.id, label: o.name || o.id }))
      .filter((o) => o.id);
  } catch {
    options = [{ id: "news", label: "news" }, { id: "documents", label: "documents" }];
  }
  if (!options.length) options = [{ id: "news", label: "news" }];
  ["scrape-ontology", "scrape-commit-ontology"].forEach((selectId) => {
    const select = document.getElementById(selectId);
    if (!select) return;
    const current = select.value;
    select.innerHTML = options.map((o) => `<option value="${escapeAttr(o.id)}">${escapeHtml(o.label)}</option>`).join("");
    if (current && options.some((o) => o.id === current)) select.value = current;
    else if (options.some((o) => o.id === "news")) select.value = "news";
  });
  await loadOntologyRelationships();
}

async function loadOntologyRelationships() {
  const box = document.getElementById("scrape-ontology-rels");
  if (!box) return;
  const ontologyId = document.getElementById("scrape-ontology").value || "news";
  box.textContent = "Loading " + ontologyId + "…";
  try {
    const data = await FB.request("/v1/admin/ontologies/" + encodeURIComponent(ontologyId));
    const definition = data.definition || data.ontology || data;
    const allowlist = definition.relationship_allowlist || {};
    const types = Object.keys(allowlist).sort();
    if (!types.length) {
      box.textContent = "No relationship types on this ontology yet. Propose will invent new ones for review.";
      return;
    }
    box.replaceChildren();
    types.forEach((relType) => {
      const pairs = allowlist[relType] || [];
      const div = document.createElement("div");
      div.className = "chunk-item";
      const pairText = pairs.length
        ? pairs.map((p) => (Array.isArray(p) ? p.join(" → ") : String(p))).join(" · ")
        : "(no pairs)";
      div.innerHTML = `<strong>${escapeHtml(relType)}</strong>${escapeHtml(pairText)}`;
      box.appendChild(div);
    });
  } catch (error) {
    box.textContent = "Could not load ontology: " + (error.message || error);
  }
}

document.getElementById("scrape-ontology").addEventListener("change", () => {
  loadOntologyRelationships().catch(showErr);
});

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/'/g, "&#39;");
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(new Error("Could not read file."));
    reader.readAsDataURL(file);
  });
}

function guessKindFromName(name) {
  const lower = String(name || "").toLowerCase();
  if (lower.endsWith(".pdf")) return "pdf";
  if (lower.endsWith(".docx")) return "docx";
  if (lower.endsWith(".doc")) return "doc";
  if (lower.endsWith(".json")) return "json";
  return "text";
}

function renderChunks(chunks) {
  const box = document.getElementById("scrape-chunks");
  if (!chunks || !chunks.length) {
    box.textContent = "No chunks yet.";
    return;
  }
  box.replaceChildren();
  chunks.forEach((chunk, index) => {
    const div = document.createElement("div");
    div.className = "chunk-item";
    const id = chunk.chunk_id || chunk.id || "p" + (index + 1);
    const text = chunk.text || chunk.content || "";
    div.innerHTML = `<strong>${escapeHtml(id)}</strong>${escapeHtml(text)}`;
    box.appendChild(div);
  });
}

function renderRelationships(relationships) {
  const body = document.getElementById("scrape-rels-body");
  if (!relationships || !relationships.length) {
    body.innerHTML = '<tr><td colspan="6">No proposals yet.</td></tr>';
    return;
  }
  body.replaceChildren();
  relationships.forEach((rel) => {
    const novelty = rel.novelty || (rel.properties && rel.properties.novelty) || "—";
    const method = rel.method || "—";
    const tr = document.createElement("tr");
    const noveltyClass = novelty === "new" ? "tag-new" : novelty === "existing" ? "tag-existing" : "";
    const methodClass = method === "ollama" ? "tag-existing" : "tag-new";
    tr.innerHTML = `
      <td class="${methodClass}">${escapeHtml(method)}</td>
      <td class="${noveltyClass}">${escapeHtml(novelty)}</td>
      <td>${escapeHtml(rel.type || "—")}</td>
      <td>${escapeHtml(rel.start_name || rel.from || "—")}</td>
      <td>${escapeHtml(rel.end_name || rel.to || "—")}</td>
      <td class="mono">${escapeHtml(rel.chunk_id || (rel.properties && rel.properties.chunk_id) || "—")}</td>`;
    body.appendChild(tr);
  });
}

function renderNameList(targetId, items, emptyLabel) {
  const box = document.getElementById(targetId);
  box.replaceChildren();
  if (!items.length) {
    box.textContent = emptyLabel;
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "list-row";
    const left = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = item.title;
    const meta = document.createElement("span");
    meta.textContent = item.meta || "";
    left.append(title, document.createElement("br"), meta);
    row.appendChild(left);
    if (item.actions && item.actions.length) {
      const actions = document.createElement("div");
      actions.className = "row-actions";
      item.actions.forEach((action) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "btn secondary";
        btn.textContent = action.label;
        btn.addEventListener("click", action.onClick);
        actions.appendChild(btn);
      });
      row.appendChild(actions);
    } else if (item.badge) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = item.badge;
      row.appendChild(badge);
    }
    box.appendChild(row);
  });
}

async function loadOverview() {
  const fb = await FB.request("/health", { noAuth: true });
  const pillFb = document.getElementById("pill-fb");
  pillFb.textContent = "FB " + (fb.status || "?");
  pillFb.className = "pill " + (fb.status === "ok" ? "ok" : "bad");
  document.getElementById("stat-backend").textContent =
    fb.graph_backend || fb.graph || fb.service || "—";

  try {
    const kg = await FB.request("/v1/admin/kg/health");
    const pillKg = document.getElementById("pill-kg");
    pillKg.textContent = "KG " + (kg.status || "ok");
    pillKg.className = "pill " + ((kg.status || "ok") === "ok" ? "ok" : "bad");
    if (kg.graph_backend) document.getElementById("stat-backend").textContent = kg.graph_backend;
  } catch (error) {
    const pillKg = document.getElementById("pill-kg");
    pillKg.textContent = "KG unreachable";
    pillKg.className = "pill bad";
  }

  const sources = await FB.request("/v1/admin/sources");
  const sourceItems = (sources.sources || sources || [])
    .map((item) => {
      if (typeof item === "string") return { title: item, meta: "source", badge: "src" };
      return {
        title: item.id || item.name || item.source_id || "source",
        meta: item.name && item.id ? item.name : item.kind || item.type || "registered",
        badge: item.kind || item.type || "src",
      };
    })
    .filter((item) => item.title);
  document.getElementById("stat-sources").textContent = String(sourceItems.length);
  renderNameList("sources-box", sourceItems, "No sources registered.");

  const graphs = await FB.request("/v1/admin/graphs");
  const graphItems = (graphs.graphs || graphs || [])
    .map((item) => {
      if (typeof item === "string") {
        return { title: item, meta: "graph", badge: "graph" };
      }
      const id = item.id || item.name || "";
      const skipBuild = id === "shared" || id === "master";
      const row = {
        title: id || "graph",
        meta:
          (item.node_count != null ? item.node_count + " nodes · " : "") +
          (item.ontology_id ? "ontology · " + item.ontology_id : item.name || item.kind || "graph"),
        badge: item.kind || "graph",
      };
      if (id && !skipBuild) {
        row.actions = [
          {
            label: "Rebuild",
            onClick: () => runGraphJob(id, "rebuild"),
          },
          {
            label: "Sync",
            onClick: () => runGraphJob(id, "sync"),
          },
        ];
      }
      return row;
    })
    .filter((item) => item.title);
  document.getElementById("stat-graphs").textContent = String(graphItems.length);
  renderNameList("graphs-box", graphItems, "No graphs yet.");

  await fillGraphSelects();
  await fillOntologySelects();
}

async function runGraphJob(graphId, action) {
  const msg = document.getElementById("reload-msg");
  try {
    await startBackgroundJob("/v1/admin/graphs/" + encodeURIComponent(graphId) + "/" + action, {}, {
      msgEl: msg,
      key: action + ":" + graphId,
      onDone: async (job) => {
        if (job.status === "done") await loadOverview();
      },
    });
  } catch (error) {
    msg.className = "msg";
    msg.textContent = error.message || action + " failed";
  }
}

async function loadIngestStatus() {
  const data = await FB.request("/v1/admin/ingestion/status");
  document.getElementById("ingest-out").textContent = pretty(data);
  const job = data.job;
  const msg = document.getElementById("ingest-msg");
  if (job) {
    msg.textContent = prettyJob(job);
    msg.className = job.status === "failed" ? "msg" : job.status === "done" ? "msg ok" : "msg";
    if (job.status === "queued" || job.status === "running") {
      await pollJobUntilDone(job.id, {
        msgEl: msg,
        outEl: document.getElementById("ingest-out"),
        key: "ingest",
        onDone: async () => loadIngestStatus().catch(() => {}),
      });
    }
  }
  return data;
}

document.getElementById("btn-ingest-status").addEventListener("click", () => {
  document.getElementById("ingest-msg").textContent = "";
  loadIngestStatus().catch((error) => {
    document.getElementById("ingest-msg").textContent = error.message;
  });
});

document.getElementById("btn-ingest-run").addEventListener("click", async () => {
  const msg = document.getElementById("ingest-msg");
  const out = document.getElementById("ingest-out");
  try {
    await startBackgroundJob(
      "/v1/admin/ingestion/run",
      { sources: ["own_products"] },
      {
        msgEl: msg,
        outEl: out,
        key: "ingest",
        onDone: async () => loadIngestStatus().catch(() => {}),
      }
    );
  } catch (error) {
    msg.className = "msg";
    msg.textContent = error.message;
  }
});

document.getElementById("btn-scrape-extract").addEventListener("click", async () => {
  setMsg("scrape-msg-1", "Extracting…", false);
  try {
    const kindSelect = document.getElementById("scrape-kind").value;
    const fileInput = document.getElementById("scrape-file");
    const text = document.getElementById("scrape-text").value;
    const payload = { clean: false };
    if (fileInput.files && fileInput.files[0]) {
      const file = fileInput.files[0];
      payload.filename = file.name;
      payload.media_type = file.type || "";
      payload.kind = kindSelect || guessKindFromName(file.name);
      payload.data_base64 = await fileToBase64(file);
    } else if (text.trim()) {
      payload.kind = kindSelect || "text";
      payload.filename = kindSelect === "json" ? "paste.json" : "paste.txt";
      if (kindSelect === "json") {
        payload.json_data = JSON.parse(text);
      } else {
        payload.text = text;
      }
    } else {
      throw new Error("Paste text or choose a file.");
    }
    const data = await FB.request("/v1/admin/scraping/extract", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    scrape.file_id = data.file_id;
    scrape.extract = data;
    scrape.review_id = "";
    document.getElementById("scrape-out-1").textContent = pretty(data);
    document.getElementById("scrape-file-id-label").textContent = "file_id: " + data.file_id;
    setMsg("scrape-msg-1", "Extracted " + (data.char_count || 0) + " chars. Continue to chunk.", true);
    markDone(1);
  } catch (error) {
    setMsg("scrape-msg-1", error.message, false);
  }
});

document.getElementById("btn-scrape-chunk").addEventListener("click", async () => {
  setMsg("scrape-msg-2", "Chunking…", false);
  try {
    if (!scrape.file_id) throw new Error("Run extract first.");
    const data = await FB.request("/v1/admin/scraping/chunk", {
      method: "POST",
      body: JSON.stringify({ file_id: scrape.file_id }),
    });
    scrape.chunks = data;
    document.getElementById("scrape-out-2").textContent = pretty({
      file_id: data.file_id,
      filename: data.filename,
      chunk_count: data.chunk_count,
    });
    renderChunks(data.chunks || []);
    setMsg("scrape-msg-2", (data.chunk_count || 0) + " paragraph chunk(s). Continue to propose.", true);
    markDone(2);
  } catch (error) {
    setMsg("scrape-msg-2", error.message, false);
  }
});

document.getElementById("btn-scrape-propose").addEventListener("click", async () => {
  setMsg("scrape-msg-3", "Proposing…", false);
  try {
    if (!scrape.file_id) throw new Error("Run extract + chunk first.");
    const data = await FB.request("/v1/admin/scraping/propose", {
      method: "POST",
      body: JSON.stringify({
        file_id: scrape.file_id,
        ontology_id: document.getElementById("scrape-ontology").value || "news",
        graph_id: document.getElementById("scrape-graph").value || undefined,
        use_model: document.getElementById("scrape-model").checked,
      }),
    });
    scrape.propose = data;
    const rels = data.relationships || [];
    const existingCount = rels.filter((r) => r.novelty === "existing").length;
    const newCount = rels.filter((r) => r.novelty === "new").length;
    const ollamaCount = rels.filter((r) => r.method === "ollama").length;
    const fallbackOnly = rels.length > 0 && ollamaCount === 0;
    document.getElementById("scrape-out-3").textContent = pretty({
      file_id: data.file_id,
      ontology_id: data.ontology_id,
      graph_id: data.graph_id,
      relationship_count: data.relationship_count,
      from_ollama: ollamaCount,
      existing: existingCount,
      new_for_ontology_review: newCount,
      warnings: data.warnings || [],
    });
    renderRelationships(rels);
    if (!rels.length) {
      setMsg(
        "scrape-msg-3",
        "0 relationships from Ollama. Keep “Use local Ollama LLM” checked and wait — multi-paragraph propose can take ~10s per chunk.",
        false
      );
    } else if (fallbackOnly) {
      setMsg(
        "scrape-msg-3",
        "Only MENTIONS fallback (method=deterministic) — Ollama did not run or failed. Check the box, confirm Ollama is up, then Run propose again. Warnings: " +
          ((data.warnings || []).slice(0, 2).join(" · ") || "none"),
        false
      );
    } else {
      setMsg(
        "scrape-msg-3",
        ollamaCount +
          " from Ollama / " +
          rels.length +
          " total (" +
          existingCount +
          " existing, " +
          newCount +
          " new for ontology review).",
        true
      );
      markDone(3);
    }
  } catch (error) {
    setMsg("scrape-msg-3", error.message, false);
  }
});

document.getElementById("btn-scrape-queue").addEventListener("click", async () => {
  setMsg("scrape-msg-4", "Queuing…", false);
  try {
    if (!scrape.file_id) throw new Error("Run propose first.");
    const queuedRels = (scrape.propose && scrape.propose.relationships) || [];
    const body = {
      file_id: scrape.file_id,
      ontology_id: (scrape.propose && scrape.propose.ontology_id) || document.getElementById("scrape-ontology").value,
      graph_id: (scrape.propose && scrape.propose.graph_id) || document.getElementById("scrape-graph").value,
      warnings: (scrape.propose && scrape.propose.warnings) || [],
    };
    if (queuedRels.length) body.relationships = queuedRels;
    const data = await FB.request("/v1/admin/scraping/queue", {
      method: "POST",
      body: JSON.stringify(body),
    });
    scrape.queue = data;
    const review = data.review || data;
    scrape.review_id = review.id || data.review_id || "";
    selectedReviewId = scrape.review_id || selectedReviewId;
    document.getElementById("scrape-out-4").textContent = pretty(data);
    document.getElementById("scrape-review-label").textContent = "review_id: " + (scrape.review_id || "—");
    setMsg("scrape-msg-4", "Queued review " + scrape.review_id + ". Continue to approve & commit.", true);
    markDone(4);
  } catch (error) {
    setMsg("scrape-msg-4", error.message, false);
  }
});

async function refreshScrapeReview() {
  if (!scrape.review_id) return;
  const data = await FB.request("/v1/admin/reviews/" + scrape.review_id);
  document.getElementById("scrape-out-5").textContent = pretty(data);
  return data;
}

document.getElementById("btn-scrape-approve").addEventListener("click", async () => {
  setMsg("scrape-msg-5", "Approving…", false);
  try {
    if (!scrape.review_id) throw new Error("Queue a review first.");
    const data = await FB.request("/v1/admin/reviews/" + scrape.review_id + "/approve", {
      method: "POST",
      body: "{}",
    });
    scrape.approve = data;
    document.getElementById("scrape-out-5").textContent = pretty(data);
    setMsg("scrape-msg-5", "Approved. Choose ontology mode and commit.", true);
    markDone(5);
  } catch (error) {
    setMsg("scrape-msg-5", error.message, false);
  }
});

document.getElementById("btn-scrape-commit").addEventListener("click", async () => {
  setMsg("scrape-msg-5", "Committing…", false);
  try {
    if (!scrape.review_id) throw new Error("Queue and approve a review first.");
    const mode = document.getElementById("scrape-commit-mode").value;
    let body;
    if (mode === "new") {
      const ontologyId = document.getElementById("scrape-new-ontology-id").value.trim();
      const graphId = document.getElementById("scrape-new-graph-id").value.trim();
      if (!ontologyId || !graphId) throw new Error("New ontology id and graph id are required.");
      body = {
        review_id: scrape.review_id,
        ontology: {
          mode: "new",
          id: ontologyId,
          name: document.getElementById("scrape-new-ontology-name").value.trim() || ontologyId,
          clone_from: document.getElementById("scrape-clone-from").value.trim() || "news",
        },
        graph: {
          id: graphId,
          name: document.getElementById("scrape-new-graph-name").value.trim() || graphId,
        },
      };
    } else {
      body = {
        review_id: scrape.review_id,
        ontology: {
          mode: "existing",
          ontology_id: document.getElementById("scrape-commit-ontology").value || "news",
        },
        graph: {
          id: document.getElementById("scrape-commit-graph").value || undefined,
        },
        graph_id: document.getElementById("scrape-commit-graph").value || undefined,
      };
    }
    const data = await FB.request("/v1/admin/scraping/commit", {
      method: "POST",
      body: JSON.stringify(body),
    });
    scrape.commit = data;
    document.getElementById("scrape-out-5").textContent = pretty(data);
    setMsg(
      "scrape-msg-5",
      "Committed to graph " + (data.graph_id || "?") + " / ontology " + (data.ontology_id || "?") + ".",
      true
    );
    await fillGraphSelects();
    await fillOntologySelects();
  } catch (error) {
    setMsg("scrape-msg-5", error.message, false);
  }
});

async function loadReviews() {
  const data = await FB.request("/v1/admin/reviews");
  const body = document.getElementById("reviews-body");
  const reviews = data.reviews || data || [];
  if (!reviews.length) {
    body.innerHTML = '<tr><td colspan="5">No reviews.</td></tr>';
    return;
  }
  body.replaceChildren();
  reviews.forEach((review) => {
    const id = review.id || review.review_id;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${escapeHtml(id)}</td>
      <td>${escapeHtml(review.status || review.decision || "—")}</td>
      <td>${escapeHtml(review.source_path || review.filename || review.kind || "—")}</td>
      <td>${(review.relationships && review.relationships.length) || review.rel_count || "—"}</td>
      <td class="row-actions"></td>`;
    const actions = tr.querySelector(".row-actions");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "btn secondary";
    open.textContent = "Open";
    open.addEventListener("click", () => openReview(id));
    actions.appendChild(open);
    body.appendChild(tr);
  });
}

async function openReview(id) {
  selectedReviewId = id;
  const data = await FB.request("/v1/admin/reviews/" + id);
  document.getElementById("review-detail").textContent = pretty(data);
}

document.getElementById("btn-refresh-reviews").addEventListener("click", () => loadReviews().catch(showErr));

document.getElementById("btn-approve").addEventListener("click", async () => {
  const msg = document.getElementById("review-msg");
  if (!selectedReviewId) {
    msg.textContent = "Select a review first.";
    return;
  }
  try {
    const data = await FB.request("/v1/admin/reviews/" + selectedReviewId + "/approve", { method: "POST", body: "{}" });
    document.getElementById("review-detail").textContent = pretty(data);
    msg.className = "msg ok";
    msg.textContent = "Approved.";
    await loadReviews();
  } catch (error) {
    msg.className = "msg";
    msg.textContent = error.message;
  }
});

document.getElementById("btn-commit").addEventListener("click", async () => {
  const msg = document.getElementById("review-msg");
  if (!selectedReviewId) {
    msg.textContent = "Select a review first.";
    return;
  }
  try {
    const data = await FB.request("/v1/admin/reviews/" + selectedReviewId + "/commit", { method: "POST", body: "{}" });
    document.getElementById("review-detail").textContent = pretty(data);
    msg.className = "msg ok";
    msg.textContent = "Committed into the graph.";
    await loadReviews();
  } catch (error) {
    msg.className = "msg";
    msg.textContent = error.message;
  }
});

document.getElementById("btn-query").addEventListener("click", async () => {
  const msg = document.getElementById("query-msg");
  msg.textContent = "";
  try {
    const params = JSON.parse(document.getElementById("q-params").value || "{}");
    const graph = document.getElementById("q-graph").value;
    const template = document.getElementById("q-template").value.trim();
    const data = await FB.request("/v1/admin/graphs/" + graph + "/query", {
      method: "POST",
      body: JSON.stringify({ template, params, limit: 20, offset: 0 }),
    });
    document.getElementById("query-out").textContent = pretty(data);
    msg.className = "msg ok";
    msg.textContent = "OK";
  } catch (error) {
    msg.className = "msg";
    msg.textContent = error.message;
  }
});

loadOverview().catch(showErr);
