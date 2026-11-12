(function (global) {
  const $ = (id) => document.getElementById(id);
  const state = {
    loaded: false,
    ontologies: [],
    sources: [],
    graphs: [],
    checklist: null,
    pollTimer: null,
  };

  function setMessage(id, text, ok) {
    const node = $(id);
    if (!node) return;
    node.textContent = text || "";
    node.className = ok ? "msg ok" : "msg";
  }

  function pretty(value) {
    return JSON.stringify(value, null, 2);
  }

  async function request(path, options) {
    return global.FB.request("/v1/admin/kg-api" + path, Object.assign({}, options || {}, { clearance: "internal" }));
  }

  function fillSelect(id, rows, emptyLabel, getValue, getLabel) {
    const select = $(id);
    const current = select.value;
    select.replaceChildren();
    if (!rows.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = emptyLabel;
      select.append(option);
      return;
    }
    rows.forEach((row) => {
      const option = document.createElement("option");
      option.value = getValue(row);
      option.textContent = getLabel(row);
      select.append(option);
    });
    if (rows.some((row) => getValue(row) === current)) select.value = current;
  }

  function showGraphs() {
    const box = $("kg-graphs-list");
    box.replaceChildren();
    if (!state.graphs.length) {
      box.textContent = "No graphs registered yet.";
      return;
    }
    state.graphs.forEach((graph) => {
      const row = document.createElement("div");
      row.className = "kg-resource-row";
      const title = document.createElement("strong");
      title.textContent = graph.name || graph.id || "Graph";
      const details = document.createElement("span");
      details.textContent = [
        graph.id,
        graph.ontology_id ? "ontology: " + graph.ontology_id : "",
        Number.isFinite(graph.node_count) ? graph.node_count + " nodes" : "",
        Number.isFinite(graph.edge_count) ? graph.edge_count + " edges" : "",
      ].filter(Boolean).join(" · ");
      row.append(title, details);
      box.append(row);
    });
  }

  function showSources() {
    const box = $("kg-sources-list");
    box.replaceChildren();
    if (!state.sources.length) {
      box.textContent = "No sources registered yet.";
      return;
    }
    state.sources.forEach((source) => {
      const row = document.createElement("div");
      row.className = "kg-resource-row";
      const title = document.createElement("strong");
      title.textContent = source.source_id || "Source";
      const details = document.createElement("span");
      details.textContent = [source.source_type, source.connector, source.enabled ? "enabled" : "disabled"].filter(Boolean).join(" · ");
      row.append(title, details);
      box.append(row);
    });
  }

  async function inspect() {
    setMessage("kg-inspect-msg", "Checking KG API and registered resources…", false);
    $("kg-service-pill").textContent = "KG API · checking";
    $("kg-service-pill").className = "pill";
    $("kg-inspect-state").textContent = "Checking";
    $("kg-inspect-state").className = "kg-step-state";

    const paths = [
      ["/health", "health"],
      ["/ontologies", "ontologies"],
      ["/api/sources", "sources"],
      ["/api/graphs", "graphs"],
      ["/api/bootstrap", "bootstrap"],
    ];
    const results = await Promise.all(paths.map(async ([path, name]) => [name, await request(path)]));
    const data = Object.fromEntries(results);
    state.ontologies = data.ontologies.ontologies || [];
    state.sources = data.sources.sources || [];
    state.graphs = data.graphs.graphs || [];
    fillSelect("kg-ontology-existing", state.ontologies, "No ontologies registered", (row) => row.id, (row) => `${row.name || row.id} (${row.id})`);
    $("kg-count-ontologies").textContent = state.ontologies.length;
    $("kg-count-sources").textContent = state.sources.length;
    $("kg-count-graphs").textContent = state.graphs.length;
    showSources();
    showGraphs();
    $("kg-inspect-output").textContent = pretty({
      health: data.health,
      counts: data.bootstrap.counts,
      active_jobs: data.bootstrap.active_jobs || [],
    });
    const healthy = data.health.status === "ok";
    $("kg-service-pill").textContent = `KG API · ${data.health.status || "connected"} · ${data.health.graph_backend || "graph store"}`;
    $("kg-service-pill").className = healthy ? "pill ok" : "pill bad";
    $("kg-inspect-state").textContent = healthy ? "Connected" : "Degraded";
    $("kg-inspect-state").className = healthy ? "kg-step-state ok" : "kg-step-state bad";
    setMessage("kg-inspect-msg", "Inspection complete.", healthy);
    state.loaded = true;
    invalidateChecklist();
    return data;
  }

  function showInspectError(error) {
    $("kg-service-pill").textContent = "KG API · unavailable";
    $("kg-service-pill").className = "pill bad";
    $("kg-inspect-state").textContent = "Connection failed";
    $("kg-inspect-state").className = "kg-step-state bad";
    setMessage("kg-inspect-msg", error.message || String(error), false);
  }

  function ontologyId() {
    return $("kg-ontology-id").value.trim() || $("kg-ontology-existing").value.trim();
  }

  function sourceId() {
    return $("kg-source-id").value.trim();
  }

  function graphId() {
    return $("kg-graph-id").value.trim();
  }

  function invalidateChecklist() {
    state.checklist = null;
    $("kg-start-build").disabled = true;
    $("kg-checklist-summary").textContent = "Not checked";
    $("kg-checklist-summary").className = "kg-step-state";
  }

  function useOntology() {
    const selected = $("kg-ontology-existing").value;
    if (!selected) {
      setMessage("kg-ontology-msg", "Choose a registered ontology first.", false);
      return;
    }
    $("kg-ontology-id").value = selected;
    $("kg-ontology-name").value = state.ontologies.find((row) => row.id === selected)?.name || selected;
    invalidateChecklist();
    setMessage("kg-ontology-msg", `Using ontology "${selected}".`, true);
  }

  async function createOntology() {
    const entityTypes = $("kg-entity-types").value.split(/[\n,]/).map((value) => value.trim()).filter(Boolean);
    const allowlist = JSON.parse($("kg-relationship-allowlist").value || "{}");
    const body = {
      name: $("kg-ontology-name").value.trim(),
      description: $("kg-ontology-description").value.trim(),
      entity_types: entityTypes,
      relationship_allowlist: allowlist,
      version: "1.0.0",
    };
    const id = $("kg-ontology-id").value.trim();
    if (id) body.id = id;
    if (!body.name) throw new Error("Enter an ontology name.");
    const result = await request("/ontologies", { method: "POST", body: JSON.stringify(body) });
    if (result.ontology && result.ontology.id) $("kg-ontology-id").value = result.ontology.id;
    await inspect();
    setMessage("kg-ontology-msg", `Ontology ${result.ontology?.id || id || "created"}.`, true);
  }

  async function loadMapping() {
    const id = encodeURIComponent(sourceId());
    if (!id) throw new Error("Enter a source ID first.");
    const result = await request("/api/sources/" + id + "/mapping");
    $("kg-mapping-json").value = pretty(result.mapping || result);
    setMessage("kg-mapping-msg", `Loaded mapping for ${sourceId()}.`, true);
  }

  async function saveMapping() {
    const id = sourceId();
    if (!id) throw new Error("Enter a source ID first.");
    const mapping = JSON.parse($("kg-mapping-json").value || "{}");
    if (!mapping || Array.isArray(mapping) || typeof mapping !== "object" || !Object.keys(mapping).length) {
      throw new Error("Mapping must be a non-empty JSON object.");
    }
    mapping.source_id = id;
    mapping.graph_id = graphId() || mapping.graph_id || id;
    mapping.source_table = $("kg-db-table").value.trim() || mapping.source_table;
    mapping.connection_id = $("kg-db-id").value.trim() || mapping.connection_id;
    const ontology = ontologyId();
    if (ontology) mapping.ontology_id = ontology;
    const result = await request("/api/sources/" + encodeURIComponent(id) + "/mapping", {
      method: "PUT",
      body: JSON.stringify(mapping),
    });
    setMessage("kg-mapping-msg", `Mapping ${result.status || "saved"} for ${id}.`, true);
    invalidateChecklist();
  }

  function dbPayload() {
    const port = Number($("kg-db-port").value);
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("Database port must be between 1 and 65535.");
    return {
      id: $("kg-db-id").value.trim(),
      host: $("kg-db-host").value.trim(),
      port,
      database: $("kg-db-database").value.trim(),
      user: $("kg-db-user").value.trim(),
      password_env: $("kg-db-password-env").value.trim() || null,
      table_default: $("kg-db-table").value.trim(),
      schema_default: $("kg-db-schema").value.trim(),
      driver: "postgresql",
      description: "Configured from the admin KG builder.",
    };
  }

  async function saveAndTestDb() {
    const payload = dbPayload();
    if (!payload.id || !payload.database) throw new Error("Connection ID and database name are required.");
    await request("/api/db/connections", { method: "POST", body: JSON.stringify(payload) });
    const query = new URLSearchParams({
      table: payload.table_default,
      schema: payload.schema_default,
    });
    const result = await request("/api/db/connections/" + encodeURIComponent(payload.id) + "/test?" + query, {
      method: "POST",
      body: "{}",
    });
    $("kg-db-output").textContent = pretty(result);
    setMessage("kg-db-msg", result.ok ? "Connection tested successfully." : "Connection test completed with an issue.", Boolean(result.ok));
    invalidateChecklist();
  }

  async function registerSource() {
    const body = {
      source_id: sourceId(),
      source_type: $("kg-source-type").value.trim(),
      connector: $("kg-source-connector").value.trim(),
      supported_formats: $("kg-source-format").value.split(",").map((value) => value.trim()).filter(Boolean),
      ingestion_mode: $("kg-source-mode").value.trim(),
      update_strategy: "content_hash",
      permissions_model: "classification",
      enabled: true,
      description: $("kg-source-description").value.trim(),
      future_connectors: [],
    };
    if (!body.source_id || !body.source_type || !body.connector) throw new Error("Source ID, type and connector are required.");
    const result = await request("/api/sources", { method: "POST", body: JSON.stringify(body) });
    setMessage("kg-source-msg", `Registered source "${result.source?.source_id || body.source_id}".`, true);
    await inspect();
    invalidateChecklist();
  }

  async function registerGraph() {
    const body = {
      id: graphId(),
      name: $("kg-graph-name").value.trim(),
      ontology_id: ontologyId(),
      kind: $("kg-graph-kind").value,
      description: $("kg-graph-description").value.trim(),
      source_ids: sourceId() ? [sourceId()] : [],
    };
    if (!body.id || !body.name || !body.ontology_id) throw new Error("Graph ID, name and ontology are required.");
    const result = await request("/graphs", { method: "POST", body: JSON.stringify(body) });
    setMessage("kg-graph-msg", `Registered graph "${result.graph?.id || body.id}".`, true);
    await inspect();
    invalidateChecklist();
  }

  function checklistLabels(checks) {
    return [
      ["health", "KG service and graph backend"],
      ["ontology", "Ontology registered"],
      ["mapping", "Source mapping saved"],
      ["db", "Database credentials available on the KG server"],
      ["source", "Source registered"],
      ["graph", "Graph registered"],
    ].map(([key, label]) => ({ key, label, ok: Boolean(checks && checks[key]) }));
  }

  async function refreshChecklist() {
    const params = new URLSearchParams({
      graph_id: graphId(),
      source_id: sourceId(),
    });
    if (ontologyId()) params.set("ontology_id", ontologyId());
    const result = await request("/api/setup/checklist?" + params);
    state.checklist = result;
    const list = $("kg-checklist");
    list.replaceChildren();
    checklistLabels(result.checks).forEach(({ key, label, ok }) => {
      const item = document.createElement("li");
      item.className = ok ? "done" : "failed";
      item.textContent = `${label}: ${ok ? "ready" : "missing"}`;
      if (Array.isArray(result.missing) && result.missing.includes(key)) item.title = `The API reports "${key}" is required before building.`;
      list.append(item);
    });
    const ready = Boolean(result.ready_to_build);
    $("kg-checklist-summary").textContent = ready ? "Ready to build" : `${(result.missing || []).length} item(s) missing`;
    $("kg-checklist-summary").className = ready ? "kg-step-state ok" : "kg-step-state bad";
    $("kg-start-build").disabled = !ready;
    return result;
  }

  async function pollBuild(graph, jobId) {
    try {
      const query = new URLSearchParams({ job_id: jobId });
      const [jobData, status] = await Promise.all([
        request("/jobs/" + encodeURIComponent(jobId)),
        request("/api/graphs/" + encodeURIComponent(graph) + "/build/status?" + query),
      ]);
      const job = jobData.job || jobData;
      const data = { job, build: status };
      $("kg-build-output").textContent = pretty(data);
      const progress = Number(status.progress_percent ?? job.progress_percent ?? job.progress?.percent);
      if (Number.isFinite(progress)) $("kg-build-progress").style.width = Math.max(0, Math.min(100, progress)) + "%";
      const currentStatus = String(job.status || status.status || "running").toLowerCase();
      if (["done", "completed"].includes(currentStatus)) {
        $("kg-build-progress").style.width = "100%";
        setMessage("kg-build-msg", "Build complete. Refresh the checklist or inspect graph counts above.", true);
        state.pollTimer = null;
        await inspect();
        return;
      }
      if (["failed", "cancelled", "error"].includes(currentStatus)) {
        setMessage("kg-build-msg", job.error || status.message || `Build ${currentStatus}. See the job details below.`, false);
        state.pollTimer = null;
        return;
      }
      setMessage("kg-build-msg", status.message || job.message || `Build ${currentStatus}… polling every 3 seconds.`, false);
      state.pollTimer = global.setTimeout(() => pollBuild(graph, jobId), 3000);
    } catch (error) {
      state.pollTimer = null;
      setMessage("kg-build-msg", error.message, false);
    }
  }

  async function startBuild() {
    if (!state.checklist || !state.checklist.ready_to_build) throw new Error("Refresh the checklist and resolve every missing item before building.");
    const graph = graphId();
    const limitText = $("kg-build-limit").value.trim();
    const limit = limitText ? Number(limitText) : null;
    if (limitText && (!Number.isInteger(limit) || limit < 1)) throw new Error("Record limit must be a positive whole number.");
    const body = {
      source_id: sourceId(),
      brand: $("kg-build-brand").value.trim() || null,
      full: $("kg-build-full").checked,
      limit,
    };
    const result = await request("/api/graphs/" + encodeURIComponent(graph) + "/build", {
      method: "POST",
      body: JSON.stringify(body),
    });
    if (state.pollTimer) global.clearTimeout(state.pollTimer);
    $("kg-build-progress").style.width = "0";
    $("kg-build-output").textContent = pretty(result);
    const job = result.job || result;
    if (!job.id) {
      setMessage("kg-build-msg", "Build request returned without a job ID.", false);
      return;
    }
    setMessage("kg-build-msg", `${job.message || "Build queued"} — tracking job ${job.id}.`, true);
    await pollBuild(graph, job.id);
  }

  async function runQuery() {
    const params = JSON.parse($("kg-query-params").value || "{}");
    const result = await request("/api/graphs/query", {
      method: "POST",
      body: JSON.stringify({
        graph_id: graphId(),
        template: $("kg-query-template").value.trim(),
        params,
        limit: 20,
        offset: 0,
      }),
    });
    $("kg-query-output").textContent = pretty(result);
    setMessage("kg-query-msg", "Query complete.", true);
  }

  function bindAction(buttonId, messageId, action) {
    $(buttonId).addEventListener("click", async () => {
      setMessage(messageId, "", false);
      try {
        await action();
      } catch (error) {
        setMessage(messageId, error.message || String(error), false);
      }
    });
  }

  $("kg-inspect").addEventListener("click", () => inspect().catch(showInspectError));
  bindAction("kg-use-ontology", "kg-ontology-msg", useOntology);
  bindAction("kg-create-ontology", "kg-ontology-msg", createOntology);
  bindAction("kg-load-mapping", "kg-mapping-msg", loadMapping);
  bindAction("kg-save-mapping", "kg-mapping-msg", saveMapping);
  bindAction("kg-save-test-db", "kg-db-msg", saveAndTestDb);
  bindAction("kg-register-source", "kg-source-msg", registerSource);
  bindAction("kg-register-graph", "kg-graph-msg", registerGraph);
  bindAction("kg-refresh-checklist", "kg-build-msg", refreshChecklist);
  bindAction("kg-start-build", "kg-build-msg", startBuild);
  bindAction("kg-run-query", "kg-query-msg", runQuery);

  ["kg-source-id", "kg-graph-id", "kg-ontology-id"].forEach((id) => {
    $(id).addEventListener("input", invalidateChecklist);
  });
  $("kg-ontology-existing").addEventListener("change", invalidateChecklist);

  global.KGBuilder = {
    load() {
      if (!state.loaded) inspect().catch(showInspectError);
    },
  };
})(window);
