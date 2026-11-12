/** Lightweight markdown + ```chart fence renderer for Fashion Brain answers. */
(function (global) {
  const CHART_COLORS = [
    "#8b1e3f",
    "#2a2e38",
    "#1f7a4d",
    "#5b6c8f",
    "#b0893c",
    "#8b4d63",
    "#3d6b8a",
    "#6b7280",
  ];

  function apiBase() {
    return String(global.FB_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");
  }

  function safeUrl(value) {
    const url = String(value || "").trim();
    if (/^https?:\/\//i.test(url) || /^mailto:/i.test(url)) return url;
    // Sample images live on the API host, not the static frontend (e.g. :5173).
    if (/^\/samples\/[a-f0-9]{32}\.(png|jpe?g|webp)$/i.test(url)) return apiBase() + url;
    return "";
  }

  function looksLikeTableRow(line) {
    return line.includes("|") && splitTableRow(line).length >= 2;
  }

  function splitTableRow(line) {
    let trimmed = line.trim();
    if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
    if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
    return trimmed.split("|").map((cell) => cell.trim());
  }

  function isDividerRow(line) {
    if (!looksLikeTableRow(line)) return false;
    return splitTableRow(line).every((cell) => /^:?-+:?$/.test(cell));
  }

  function isTableStart(lines, index) {
    return index + 1 < lines.length && looksLikeTableRow(lines[index]) && isDividerRow(lines[index + 1]);
  }

  function isBlockStart(lines, index) {
    const line = lines[index];
    if (!line || !line.trim()) return false;
    if (/^```/.test(line)) return true;
    if (/^#{1,3}\s+/.test(line)) return true;
    if (/^\s*>/.test(line)) return true;
    if (/^\s*(?:[-*+]|\d+[.)])\s+/.test(line)) return true;
    if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line.trim())) return true;
    return isTableStart(lines, index);
  }

  /** Extract a balanced `{…}` object starting at index. */
  function readJsonObject(text, start) {
    if (text[start] !== "{") return null;
    let depth = 0;
    let inString = false;
    let escape = false;
    for (let index = start; index < text.length; index += 1) {
      const char = text[index];
      if (inString) {
        if (escape) escape = false;
        else if (char === "\\") escape = true;
        else if (char === "\"") inString = false;
        continue;
      }
      if (char === "\"") {
        inString = true;
        continue;
      }
      if (char === "{") depth += 1;
      else if (char === "}") {
        depth -= 1;
        if (depth === 0) return text.slice(start, index + 1);
      }
    }
    return null;
  }

  /** Pull markdown table rows (2+ cells) out of a flattened blob. */
  function extractTableRows(blob) {
    const rows = [];
    const rowRe = /\|(?:[^|\n]+\|){2,}/g;
    let match;
    while ((match = rowRe.exec(String(blob || "")))) {
      rows.push(match[0].trim());
    }
    return rows;
  }

  function repairFlatTables(text) {
    // "| Name | Count | | --- | ---: | | a | 1 |" → real markdown rows
    const dividerRe = /\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|/;
    const parts = [];
    let cursor = 0;
    let searchFrom = 0;
    while (searchFrom < text.length) {
      const dividerMatch = text.slice(searchFrom).match(dividerRe);
      if (!dividerMatch || dividerMatch.index == null) break;
      const dividerAt = searchFrom + dividerMatch.index;
      const divider = dividerMatch[0];
      const before = text.slice(cursor, dividerAt);
      const headers = extractTableRows(before);
      if (!headers.length) {
        searchFrom = dividerAt + divider.length;
        continue;
      }
      const header = headers[headers.length - 1];
      const headerAt = dividerAt - before.length + before.lastIndexOf(header);
      const after = text.slice(dividerAt + divider.length);
      const firstLineEnd = after.indexOf("\n");
      const firstLine = firstLineEnd < 0 ? after : after.slice(0, firstLineEnd);
      let bodyRows = extractTableRows(firstLine);
      let bodyEnd = dividerAt + divider.length + firstLine.length;
      if (!bodyRows.length) {
        const afterLines = after.split("\n");
        bodyRows = [];
        bodyEnd = dividerAt + divider.length;
        let consumedNewline = false;
        for (const rowLine of afterLines) {
          if (!rowLine.trim()) {
            if (bodyRows.length) break;
            bodyEnd += rowLine.length + (consumedNewline ? 1 : 0);
            consumedNewline = true;
            continue;
          }
          const row = rowLine.trim();
          if (!row.startsWith("|") || !looksLikeTableRow(row) || isDividerRow(row)) break;
          bodyRows.push(row);
          bodyEnd += rowLine.length + (consumedNewline ? 1 : 0);
          consumedNewline = true;
        }
      }
      if (!bodyRows.length) {
        searchFrom = dividerAt + divider.length;
        continue;
      }
      parts.push(text.slice(cursor, headerAt).replace(/[ \t]+$/g, ""));
      parts.push("\n" + header + "\n" + divider + "\n" + bodyRows.join("\n") + "\n");
      cursor = bodyEnd;
      searchFrom = bodyEnd;
    }
    parts.push(text.slice(cursor));
    return parts.join("");
  }

  /** Repair space-collapsed ```chart / sampleboard {…} ``` blobs so fences parse again. */
  function normalizeVizMarkdown(source) {
    let text = String(source || "")
      .replace(/\r\n/g, "\n")
      // Internal analysis markers must never show in chat.
      .replace(/<!--\s*fb:viz:[^>]*-->/gi, "");
    const fence = /```(chart|fbchart|sampleboard)\s*/gi;
    let match;
    const chunks = [];
    let cursor = 0;
    while ((match = fence.exec(text))) {
      const lang = match[1].toLowerCase();
      const jsonStart = match.index + match[0].length;
      if (text[jsonStart] !== "{") {
        fence.lastIndex = match.index + 3;
        continue;
      }
      const json = readJsonObject(text, jsonStart);
      if (!json) {
        fence.lastIndex = match.index + 3;
        continue;
      }
      let close = jsonStart + json.length;
      const after = text.slice(close).match(/^\s*```/);
      if (after) close += after[0].length;
      const before = text.slice(cursor, match.index).replace(/[ \t]+$/g, "");
      chunks.push(before.endsWith("\n") ? before : before + "\n");
      chunks.push("```" + lang + "\n" + json + "\n```\n");
      cursor = close;
      fence.lastIndex = close;
    }
    chunks.push(text.slice(cursor).replace(/^[ \t]+/g, ""));
    return repairFlatTables(chunks.join(""));
  }

  function parseMarkdown(source) {
    const lines = normalizeVizMarkdown(source).split("\n");
    const blocks = [];
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }
      const fence = line.match(/^```([^\s`]*)\s*$/);
      if (fence) {
        const body = [];
        index += 1;
        while (index < lines.length && !/^```\s*$/.test(lines[index])) {
          body.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        const lang = String(fence[1] || "").toLowerCase();
        const text = body.join("\n");
        if (lang === "chart" || lang === "fbchart") {
          try {
            blocks.push({ type: "chart", data: JSON.parse(text) });
          } catch (_err) {
            blocks.push({ type: "code", lang: fence[1], text });
          }
        } else if (lang === "sampleboard") {
          try {
            blocks.push({ type: "sampleboard", data: JSON.parse(text) });
          } catch (_err) {
            blocks.push({ type: "code", lang: fence[1], text });
          }
        } else {
          blocks.push({ type: "code", lang: fence[1], text });
        }
        continue;
      }
      if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line.trim())) {
        blocks.push({ type: "hr" });
        index += 1;
        continue;
      }
      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        blocks.push({ type: "heading", level: heading[1].length, text: heading[2].trim() });
        index += 1;
        continue;
      }
      if (/^\s*>/.test(line)) {
        const body = [];
        while (index < lines.length && /^\s*>/.test(lines[index])) {
          body.push(lines[index].replace(/^\s*>\s?/, ""));
          index += 1;
        }
        blocks.push({ type: "quote", text: body.join("\n") });
        continue;
      }
      if (isTableStart(lines, index)) {
        const header = splitTableRow(lines[index]);
        index += 2;
        const rows = [];
        while (index < lines.length && looksLikeTableRow(lines[index]) && !isDividerRow(lines[index])) {
          rows.push(splitTableRow(lines[index]));
          index += 1;
        }
        blocks.push({ type: "table", header, rows });
        continue;
      }
      if (/^\s*(?:[-*+]|\d+[.)])\s+/.test(line)) {
        const ordered = /^\s*\d+[.)]\s+/.test(line);
        const items = [];
        while (index < lines.length) {
          const item = lines[index].match(/^\s*(?:[-*+]|\d+[.)])\s+(.*)$/);
          if (item) {
            items.push(item[1]);
            index += 1;
            continue;
          }
          if (items.length && /^\s+\S/.test(lines[index])) {
            items[items.length - 1] += "\n" + lines[index].trim();
            index += 1;
            continue;
          }
          break;
        }
        blocks.push({ type: "list", ordered, items });
        continue;
      }
      const paragraph = [];
      while (index < lines.length && lines[index].trim() && !isBlockStart(lines, index)) {
        paragraph.push(lines[index]);
        index += 1;
      }
      if (paragraph.length) blocks.push({ type: "paragraph", text: paragraph.join("\n") });
      else index += 1;
    }
    return blocks;
  }

  function appendInline(parent, source) {
    const text = String(source || "");
    let index = 0;
    while (index < text.length) {
      if (text[index] === "\\" && index + 1 < text.length) {
        parent.appendChild(document.createTextNode(text[index + 1]));
        index += 2;
        continue;
      }
      if (text[index] === "`") {
        const end = text.indexOf("`", index + 1);
        if (end > index) {
          const code = document.createElement("code");
          code.textContent = text.slice(index + 1, end);
          parent.appendChild(code);
          index = end + 1;
          continue;
        }
      }
      if (text.startsWith("![", index)) {
        const close = text.indexOf("]", index + 2);
        if (close > index && text[close + 1] === "(") {
          const end = text.indexOf(")", close + 2);
          const href = end > close ? safeUrl(text.slice(close + 2, end)) : "";
          if (href && (/^https?:\/\//i.test(href) || /\/samples\//i.test(href))) {
            const image = document.createElement("img");
            image.alt = text.slice(index + 2, close);
            image.src = href;
            image.loading = "lazy";
            image.referrerPolicy = "no-referrer";
            parent.appendChild(image);
            index = end + 1;
            continue;
          }
        }
      }
      if (text.startsWith("**", index) || text.startsWith("__", index)) {
        const marker = text.slice(index, index + 2);
        const end = text.indexOf(marker, index + 2);
        if (end > index) {
          const strong = document.createElement("strong");
          appendInline(strong, text.slice(index + 2, end));
          parent.appendChild(strong);
          index = end + 2;
          continue;
        }
      }
      if (text.startsWith("~~", index)) {
        const end = text.indexOf("~~", index + 2);
        if (end > index) {
          const strike = document.createElement("s");
          appendInline(strike, text.slice(index + 2, end));
          parent.appendChild(strike);
          index = end + 2;
          continue;
        }
      }
      if (text[index] === "[") {
        const close = text.indexOf("]", index + 1);
        if (close > index && text[close + 1] === "(") {
          const end = text.indexOf(")", close + 2);
          const href = end > close ? safeUrl(text.slice(close + 2, end)) : "";
          if (href) {
            const link = document.createElement("a");
            link.href = href;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            appendInline(link, text.slice(index + 1, close));
            parent.appendChild(link);
            index = end + 1;
            continue;
          }
        }
      }
      if ((text[index] === "*" || text[index] === "_") && text[index + 1] !== text[index]) {
        const marker = text[index];
        const end = text.indexOf(marker, index + 1);
        if (end > index && text[end + 1] !== marker) {
          const emphasis = document.createElement("em");
          appendInline(emphasis, text.slice(index + 1, end));
          parent.appendChild(emphasis);
          index = end + 1;
          continue;
        }
      }
      let next = index + 1;
      while (next < text.length && !"\\`*_~[".includes(text[next])) next += 1;
      parent.appendChild(document.createTextNode(text.slice(index, next)));
      index = next;
    }
  }

  function fillLines(parent, text) {
    String(text || "")
      .split("\n")
      .forEach((line, lineIndex) => {
        if (lineIndex) parent.appendChild(document.createElement("br"));
        appendInline(parent, line);
      });
  }

  function chartSeries(data) {
    const labels = Array.isArray(data && data.labels) ? data.labels.map((item) => String(item)) : [];
    const values = Array.isArray(data && data.values)
      ? data.values.map((item) => Number(item)).map((item) => (Number.isFinite(item) ? item : 0))
      : [];
    const size = Math.min(labels.length, values.length);
    const rawType = String((data && data.type) || "bar").toLowerCase();
    const kind = rawType === "pie" ? "pie" : rawType === "line" ? "line" : "bar";
    const sliced = values.slice(0, size);
    const avgRaw = data && data.avg != null ? Number(data.avg) : NaN;
    const minRaw = data && data.min != null ? Number(data.min) : NaN;
    const maxRaw = data && data.max != null ? Number(data.max) : NaN;
    return {
      title: String((data && data.title) || "").trim(),
      kind,
      smooth: data && data.smooth !== false,
      analytics: Boolean(data && data.analytics) || kind === "line",
      labels: labels.slice(0, size),
      values: sliced,
      avg: Number.isFinite(avgRaw) ? avgRaw : sliced.length ? sliced.reduce((a, b) => a + b, 0) / sliced.length : 0,
      min: Number.isFinite(minRaw) ? minRaw : sliced.length ? Math.min(...sliced) : 0,
      max: Number.isFinite(maxRaw) ? maxRaw : sliced.length ? Math.max(...sliced) : 0,
    };
  }

  function renderBarChart(svg, labels, values) {
    const width = 760;
    const height = Math.max(160, 28 + labels.length * 34);
    const left = 240;
    const right = 56;
    const top = 8;
    const bottom = 8;
    const minValue = Math.min(...values, 0);
    const maxValue = Math.max(...values, 0);
    const valueRange = maxValue - minValue || 1;
    const plotWidth = width - left - right;
    const zeroX = left + (plotWidth * (0 - minValue)) / valueRange;
    const barH = Math.min(22, (height - top - bottom) / Math.max(labels.length, 1) - 8);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const zeroAxis = document.createElementNS("http://www.w3.org/2000/svg", "line");
    zeroAxis.setAttribute("x1", String(zeroX));
    zeroAxis.setAttribute("x2", String(zeroX));
    zeroAxis.setAttribute("y1", String(top));
    zeroAxis.setAttribute("y2", String(height - bottom));
    zeroAxis.setAttribute("stroke", "#aeb4bf");
    zeroAxis.setAttribute("stroke-width", "1");
    svg.appendChild(zeroAxis);
    labels.forEach((label, index) => {
      const value = values[index];
      const y = top + index * ((height - top - bottom) / labels.length) + 6;
      const valueX = left + (plotWidth * (value - minValue)) / valueRange;
      const barX = Math.min(zeroX, valueX);
      const barW = Math.abs(valueX - zeroX);
      const name = document.createElementNS("http://www.w3.org/2000/svg", "text");
      name.setAttribute("x", String(left - 8));
      name.setAttribute("y", String(y + barH * 0.72));
      name.setAttribute("text-anchor", "end");
      name.setAttribute("font-size", "12");
      name.setAttribute("fill", "#6b7280");
      name.textContent = label.length > 30 ? label.slice(0, 29) + "…" : label;
      name.setAttribute("aria-label", label);
      name.setAttribute("title", label);
      svg.appendChild(name);
      const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      bar.setAttribute("x", String(barX));
      bar.setAttribute("y", String(y));
      bar.setAttribute("width", String(Math.max(barW, 2)));
      bar.setAttribute("height", String(barH));
      bar.setAttribute("rx", "4");
      bar.setAttribute("fill", CHART_COLORS[index % CHART_COLORS.length]);
      svg.appendChild(bar);
      const count = document.createElementNS("http://www.w3.org/2000/svg", "text");
      count.setAttribute("x", String(value < 0 ? barX + 6 : valueX + 6));
      count.setAttribute("text-anchor", "start");
      count.setAttribute("y", String(y + barH * 0.72));
      count.setAttribute("font-size", "12");
      count.setAttribute("fill", "#12141a");
      count.textContent = String(value);
      svg.appendChild(count);
    });
  }

  function renderPieChart(svg, labels, values) {
    const width = 760;
    const height = Math.max(240, 38 + labels.length * 24);
    const cx = 120;
    const cy = 120;
    const radius = 88;
    const weights = values.map((value) => Math.abs(value));
    const total = weights.reduce((sum, value) => sum + value, 0) || 1;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    let angle = -Math.PI / 2;
    values.forEach((value, index) => {
      const slice = (weights[index] / total) * Math.PI * 2;
      const x1 = cx + radius * Math.cos(angle);
      const y1 = cy + radius * Math.sin(angle);
      angle += slice;
      const x2 = cx + radius * Math.cos(angle);
      const y2 = cy + radius * Math.sin(angle);
      const large = slice > Math.PI ? 1 : 0;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute(
        "d",
        `M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2} Z`
      );
      path.setAttribute("fill", CHART_COLORS[index % CHART_COLORS.length]);
      svg.appendChild(path);
      const legendY = 30 + index * 24;
      const swatch = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      swatch.setAttribute("x", "250");
      swatch.setAttribute("y", String(legendY - 10));
      swatch.setAttribute("width", "12");
      swatch.setAttribute("height", "12");
      swatch.setAttribute("rx", "2");
      swatch.setAttribute("fill", CHART_COLORS[index % CHART_COLORS.length]);
      svg.appendChild(swatch);
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", "270");
      text.setAttribute("y", String(legendY));
      text.setAttribute("font-size", "12");
      text.setAttribute("fill", "#6b7280");
      const pct = Math.round((1000 * weights[index]) / total) / 10;
      text.textContent = `${labels[index]} — ${value} (${pct}%)`;
      text.setAttribute("aria-label", `${labels[index]} — ${value} (${pct}%)`);
      text.setAttribute("title", `${labels[index]} — ${value} (${pct}%)`);
      svg.appendChild(text);
    });
  }

  function smoothLinePath(points) {
    if (!points.length) return "";
    if (points.length === 1) return `M ${points[0][0]} ${points[0][1]}`;
    if (points.length === 2) {
      return `M ${points[0][0]} ${points[0][1]} L ${points[1][0]} ${points[1][1]}`;
    }
    let d = `M ${points[0][0]} ${points[0][1]}`;
    for (let i = 0; i < points.length - 1; i += 1) {
      const p0 = points[Math.max(i - 1, 0)];
      const p1 = points[i];
      const p2 = points[i + 1];
      const p3 = points[Math.min(i + 2, points.length - 1)];
      const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
      const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
      const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
      const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2[0]} ${p2[1]}`;
    }
    return d;
  }

  function renderLineChart(svg, labels, values, options) {
    const smooth = !options || options.smooth !== false;
    const analytics = !options || options.analytics !== false;
    const width = 560;
    const height = analytics ? 260 : 220;
    const left = 52;
    const right = 18;
    const top = 28;
    const bottom = 44;
    const seriesMin = Math.min(...values);
    const seriesMax = Math.max(...values);
    const avg = options && Number.isFinite(options.avg) ? options.avg : values.reduce((a, b) => a + b, 0) / values.length;
    const extentMin = Math.min(seriesMin, Number.isFinite(options && options.min) ? options.min : seriesMin, avg);
    const extentMax = Math.max(seriesMax, Number.isFinite(options && options.max) ? options.max : seriesMax, avg);
    const span = Math.max(extentMax - extentMin, Math.abs(extentMax) * 0.08, 1);
    const yMin = extentMin - span * 0.12;
    const yMax = extentMax + span * 0.16;
    const plotW = width - left - right;
    const plotH = height - top - bottom;
    const n = Math.max(labels.length - 1, 1);
    const yOf = (value) => top + plotH * (1 - (value - yMin) / (yMax - yMin));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const points = values.map((value, index) => [left + (plotW * index) / n, yOf(value)]);

    // Y-axis guide ticks.
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      const value = yMin + (yMax - yMin) * (1 - t);
      const y = top + plotH * t;
      const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
      grid.setAttribute("x1", String(left));
      grid.setAttribute("x2", String(left + plotW));
      grid.setAttribute("y1", String(y));
      grid.setAttribute("y2", String(y));
      grid.setAttribute("stroke", "#eef0f3");
      svg.appendChild(grid);
      const tick = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tick.setAttribute("x", String(left - 6));
      tick.setAttribute("y", String(y + 3));
      tick.setAttribute("text-anchor", "end");
      tick.setAttribute("font-size", "10");
      tick.setAttribute("fill", "#9ca3af");
      tick.textContent = `${Math.round(value * 10) / 10}%`;
      svg.appendChild(tick);
    });

    if (yMin < 0 && yMax > 0) {
      const zeroY = yOf(0);
      const zero = document.createElementNS("http://www.w3.org/2000/svg", "line");
      zero.setAttribute("x1", String(left));
      zero.setAttribute("x2", String(left + plotW));
      zero.setAttribute("y1", String(zeroY));
      zero.setAttribute("y2", String(zeroY));
      zero.setAttribute("stroke", "#d1d5db");
      zero.setAttribute("stroke-dasharray", "4 4");
      svg.appendChild(zero);
    }

    const lineD = smooth
      ? smoothLinePath(points)
      : points.map((pt, i) => `${i ? "L" : "M"} ${pt[0]} ${pt[1]}`).join(" ");

    if (analytics && points.length) {
      const area = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const baseY = yOf(Math.max(0, yMin));
      area.setAttribute(
        "d",
        `${lineD} L ${points[points.length - 1][0]} ${baseY} L ${points[0][0]} ${baseY} Z`
      );
      area.setAttribute("fill", CHART_COLORS[0]);
      area.setAttribute("fill-opacity", "0.12");
      area.setAttribute("stroke", "none");
      svg.appendChild(area);
    }

    // Average reference line.
    if (analytics && Number.isFinite(avg)) {
      const avgY = yOf(avg);
      const avgLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      avgLine.setAttribute("x1", String(left));
      avgLine.setAttribute("x2", String(left + plotW));
      avgLine.setAttribute("y1", String(avgY));
      avgLine.setAttribute("y2", String(avgY));
      avgLine.setAttribute("stroke", "#f59e0b");
      avgLine.setAttribute("stroke-width", "1.5");
      avgLine.setAttribute("stroke-dasharray", "5 4");
      svg.appendChild(avgLine);
      const avgLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      avgLabel.setAttribute("x", String(left + plotW - 2));
      avgLabel.setAttribute("y", String(avgY - 5));
      avgLabel.setAttribute("text-anchor", "end");
      avgLabel.setAttribute("font-size", "10");
      avgLabel.setAttribute("fill", "#b45309");
      avgLabel.textContent = `avg ${Math.round(avg * 10) / 10}%`;
      svg.appendChild(avgLabel);
    }

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", lineD);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", CHART_COLORS[0]);
    path.setAttribute("stroke-width", "2.4");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    svg.appendChild(path);

    const minIdx = values.indexOf(seriesMin);
    const maxIdx = values.indexOf(seriesMax);
    const labelStep = Math.max(1, Math.ceil(points.length / 8));
    points.forEach((pt, index) => {
      const isExtreme = index === minIdx || index === maxIdx;
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", String(pt[0]));
      dot.setAttribute("cy", String(pt[1]));
      dot.setAttribute("r", isExtreme ? "4.4" : "2.6");
      dot.setAttribute("fill", isExtreme ? (index === maxIdx ? "#059669" : "#dc2626") : CHART_COLORS[0]);
      svg.appendChild(dot);
      if (isExtreme) {
        const callout = document.createElementNS("http://www.w3.org/2000/svg", "text");
        callout.setAttribute("x", String(pt[0]));
        callout.setAttribute("y", String(index === maxIdx ? pt[1] - 8 : pt[1] + 14));
        callout.setAttribute("text-anchor", "middle");
        callout.setAttribute("font-size", "10");
        callout.setAttribute("font-weight", "600");
        callout.setAttribute("fill", index === maxIdx ? "#047857" : "#b91c1c");
        callout.textContent = `${index === maxIdx ? "max" : "min"} ${Math.round(values[index] * 10) / 10}%`;
        svg.appendChild(callout);
      }
      if (index === 0 || index === points.length - 1 || index % labelStep === 0) {
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", String(pt[0]));
        label.setAttribute("y", String(height - 12));
        label.setAttribute("text-anchor", "middle");
        label.setAttribute("font-size", "9");
        label.setAttribute("fill", "#6b7280");
        const raw = labels[index] || "";
        label.textContent = raw.length > 11 ? raw.slice(0, 10) + "…" : raw;
        svg.appendChild(label);
      }
    });
  }

  function renderChart(data) {
    const wrap = document.createElement("div");
    wrap.className = "chart";
    const series = chartSeries(data);
    if (series.title) {
      const title = document.createElement("div");
      title.className = "chart-title";
      title.textContent = series.title;
      wrap.appendChild(title);
    }
    if (!series.labels.length || !series.values.length) {
      const empty = document.createElement("div");
      empty.className = "chart-empty";
      empty.textContent = "No chart values were provided.";
      wrap.appendChild(empty);
      return wrap;
    }
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("role", "img");
    svg.setAttribute(
      "aria-label",
      series.title || (series.kind === "pie" ? "Pie chart" : series.kind === "line" ? "Line chart" : "Bar chart")
    );
    if (series.kind === "pie") renderPieChart(svg, series.labels, series.values);
    else if (series.kind === "line") {
      renderLineChart(svg, series.labels, series.values, {
        smooth: series.smooth,
        analytics: series.analytics,
        avg: series.avg,
        min: series.min,
        max: series.max,
      });
    } else renderBarChart(svg, series.labels, series.values);
    wrap.appendChild(svg);
    return wrap;
  }

  /** Advanced sample-board tile: **Title** + image or missing-image line. */
  function sampleShotFromParagraph(text) {
    const raw = String(text || "").trim();
    const titleMatch = raw.match(/^\*\*(.+?)\*\*\s*\n([\s\S]*)$/);
    if (!titleMatch) return null;
    const title = titleMatch[1].trim();
    const rest = titleMatch[2].trim();
    const img = rest.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (img) {
      const href = safeUrl(img[2]);
      if (href && /\/samples\//i.test(href)) return { title, href, alt: img[1] || title };
      return null;
    }
    if (/The sample image was not generated/i.test(rest)) return { title, error: rest };
    return null;
  }

  function shotByKey(shots, keys) {
    const list = Array.isArray(shots) ? shots : [];
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      const hit = list.find((shot) => String(shot.key || "").toLowerCase() === key);
      if (hit) return hit;
    }
    return null;
  }

  function boardMissLabel(shot) {
    const err = String((shot && shot.error) || "");
    if (/402/.test(err)) return "Credits / in-flight limit";
    if (err) return "Not generated";
    return "Generating…";
  }

  function fillBoardMedia(node, shot) {
    if (shot && shot.url) {
      const href = safeUrl(shot.url);
      if (href) {
        const image = document.createElement("img");
        image.src = href;
        image.alt = shot.title || "";
        image.loading = "lazy";
        image.referrerPolicy = "no-referrer";
        node.appendChild(image);
        return;
      }
    }
    const miss = document.createElement("div");
    miss.className = "sample-board-miss";
    miss.textContent = boardMissLabel(shot);
    node.appendChild(miss);
  }

  function renderSampleBoard(data) {
    const shots = Array.isArray(data && data.shots) ? data.shots : [];
    const board = document.createElement("article");
    board.className = "sample-board";

    const header = document.createElement("header");
    header.className = "sample-board-header";
    const brand = document.createElement("div");
    brand.className = "sample-board-brand";
    brand.textContent = String((data && data.buyer) || "FASHION BRAIN");
    const heading = document.createElement("div");
    heading.className = "sample-board-heading";
    const badge = document.createElement("div");
    badge.className = "sample-board-badge";
    badge.textContent = String((data && data.title) || "Garment Sample – Suggested Common Style");
    heading.appendChild(badge);
    if (data && data.subtitle) {
      const sub = document.createElement("div");
      sub.className = "sample-board-subtitle";
      sub.textContent = String(data.subtitle);
      heading.appendChild(sub);
    }
    header.append(brand, heading);
    board.appendChild(header);

    const hero = document.createElement("figure");
    hero.className = "sample-board-hero";
    fillBoardMedia(hero, shotByKey(shots, ["on_form"]));
    board.appendChild(hero);

    // Right rail: product views → optional back/fabric → spec table (Primark-style).
    const rail = document.createElement("div");
    rail.className = "sample-board-rail";

    const views = document.createElement("div");
    views.className = "sample-board-views";
    const product = document.createElement("figure");
    product.className = "sample-board-product";
    fillBoardMedia(product, shotByKey(shots, ["front"]));
    const details = document.createElement("div");
    details.className = "sample-board-details";
    const detailA = document.createElement("figure");
    detailA.className = "sample-board-detail";
    fillBoardMedia(detailA, shotByKey(shots, ["neck_label"]));
    const detailB = document.createElement("figure");
    detailB.className = "sample-board-detail";
    fillBoardMedia(detailB, shotByKey(shots, ["hem_cuff"]));
    details.append(detailA, detailB);
    views.append(product, details);
    rail.appendChild(views);

    const stripShots = ["back", "fabric"]
      .map((key) => shotByKey(shots, [key]))
      .filter((shot) => shot && (shot.url || shot.error));
    if (stripShots.length) {
      const strip = document.createElement("div");
      strip.className = "sample-board-strip";
      stripShots.forEach((shot) => {
        const tile = document.createElement("figure");
        tile.className = "sample-board-strip-tile";
        fillBoardMedia(tile, shot);
        strip.appendChild(tile);
      });
      rail.appendChild(strip);
    }

    const rows = Array.isArray(data && data.rows) ? data.rows : [];
    if (rows.length) {
      const wrap = document.createElement("div");
      wrap.className = "sample-board-spec";
      const table = document.createElement("table");
      const head = document.createElement("thead");
      const headRow = document.createElement("tr");
      const th = document.createElement("th");
      th.colSpan = 2;
      th.textContent = "Product Specification Summary";
      headRow.appendChild(th);
      head.appendChild(headRow);
      const body = document.createElement("tbody");
      rows.forEach((row) => {
        const tr = document.createElement("tr");
        const key = document.createElement("th");
        key.scope = "row";
        key.textContent = String((row && row[0]) || "");
        const val = document.createElement("td");
        val.textContent = String((row && row[1]) || "");
        tr.append(key, val);
        body.appendChild(tr);
      });
      table.append(head, body);
      wrap.appendChild(table);
      rail.appendChild(wrap);
    }

    board.appendChild(rail);

    const footer = document.createElement("footer");
    footer.className = "sample-board-footer";
    const chips = document.createElement("div");
    chips.className = "sample-board-chips";
    (Array.isArray(data && data.chips) ? data.chips : []).forEach((label) => {
      const chip = document.createElement("div");
      chip.className = "sample-board-chip";
      const mark = document.createElement("span");
      mark.className = "sample-board-chip-mark";
      mark.setAttribute("aria-hidden", "true");
      const text = document.createElement("span");
      text.textContent = String(label || "");
      chip.append(mark, text);
      chips.appendChild(chip);
    });
    const tagline = document.createElement("div");
    tagline.className = "sample-board-tagline";
    const line = document.createElement("span");
    line.textContent = String((data && data.tagline) || "");
    const mark = document.createElement("strong");
    mark.textContent = String((data && data.buyer) || "");
    tagline.append(line, mark);
    footer.append(chips, tagline);
    board.appendChild(footer);
    return board;
  }

  function renderSampleBoardFromShots(shots) {
    const keyFor = (title) => {
      const t = String(title || "").toLowerCase();
      if (t.includes("three-quarter") || t.includes("on form")) return "on_form";
      if (t.includes("neck")) return "neck_label";
      if (t.includes("hem") || t.includes("cuff")) return "hem_cuff";
      if (t.includes("fabric")) return "fabric";
      if (t.includes("back")) return "back";
      if (t.includes("front")) return "front";
      return "front";
    };
    return renderSampleBoard({
      title: "Garment Sample – Suggested Common Style",
      subtitle: "Advanced garment sample board",
      buyer: "FASHION BRAIN",
      shots: shots.map((shot) => ({
        key: keyFor(shot.title),
        title: shot.title,
        url: shot.href || "",
        error: shot.error || "",
      })),
      rows: [],
      chips: [],
      tagline: "",
    });
  }

  function renderBlock(block) {
    if (block.type === "heading") {
      const heading = document.createElement("h" + block.level);
      appendInline(heading, block.text);
      return heading;
    }
    if (block.type === "code") {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = block.text;
      pre.appendChild(code);
      return pre;
    }
    if (block.type === "chart") return renderChart(block.data);
    if (block.type === "sampleboard") return renderSampleBoard(block.data || {});
    if (block.type === "hr") return document.createElement("hr");
    if (block.type === "quote") {
      const quote = document.createElement("blockquote");
      parseMarkdown(block.text).forEach((child) => quote.appendChild(renderBlock(child)));
      return quote;
    }
    if (block.type === "list") {
      const list = document.createElement(block.ordered ? "ol" : "ul");
      block.items.forEach((item) => {
        const li = document.createElement("li");
        fillLines(li, item);
        list.appendChild(li);
      });
      return list;
    }
    if (block.type === "table") {
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      const table = document.createElement("table");
      const head = document.createElement("thead");
      const headRow = document.createElement("tr");
      block.header.forEach((cell) => {
        const th = document.createElement("th");
        appendInline(th, cell);
        headRow.appendChild(th);
      });
      head.appendChild(headRow);
      const body = document.createElement("tbody");
      block.rows.forEach((row) => {
        const tr = document.createElement("tr");
        const width = Math.max(block.header.length, row.length);
        for (let cellIndex = 0; cellIndex < width; cellIndex += 1) {
          const td = document.createElement("td");
          appendInline(td, row[cellIndex] || "");
          tr.appendChild(td);
        }
        body.appendChild(tr);
      });
      table.append(head, body);
      wrap.appendChild(table);
      return wrap;
    }
    const paragraph = document.createElement("p");
    fillLines(paragraph, block.text);
    return paragraph;
  }

  function paint(node, text) {
    node.replaceChildren();
    const blocks = parseMarkdown(text);
    const hasBoard = blocks.some((block) => block.type === "sampleboard");
    let index = 0;
    while (index < blocks.length) {
      const block = blocks[index];
      if (block.type === "sampleboard") {
        node.appendChild(renderSampleBoard(block.data || {}));
        index += 1;
        continue;
      }
      // Collage only for multi-view boards. Single /garment_design stays a normal image.
      if (!hasBoard && block.type === "paragraph") {
        const first = sampleShotFromParagraph(block.text);
        if (first) {
          const shots = [];
          let cursor = index;
          while (cursor < blocks.length && blocks[cursor].type === "paragraph") {
            const shot = sampleShotFromParagraph(blocks[cursor].text);
            if (!shot) break;
            shots.push(shot);
            cursor += 1;
          }
          if (shots.length >= 2) {
            node.appendChild(renderSampleBoardFromShots(shots));
            index = cursor;
            continue;
          }
        }
      }
      node.appendChild(renderBlock(block));
      index += 1;
    }
  }

  global.FBMarkdown = { paint, parseMarkdown };
})(typeof window !== "undefined" ? window : globalThis);
