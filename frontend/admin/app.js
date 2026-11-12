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

const views = ["kg-builder", "overview"];

document.querySelectorAll(".nav-btn[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => showView(btn.getAttribute("data-view")));
});
document.querySelectorAll(".linkish[data-view], .btn[data-view]").forEach((btn) => {
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
  if (name === "overview") loadOverview().catch(showErr);
  if (name === "kg-builder") window.KGBuilder.load();
}

function showErr(error) {
  console.error(error);
  alert(error.message || String(error));
}

function pretty(data) {
  return JSON.stringify(data, null, 2);
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

showView("kg-builder");
