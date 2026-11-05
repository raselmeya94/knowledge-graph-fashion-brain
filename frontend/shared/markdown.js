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

  function safeUrl(value) {
    const url = String(value || "").trim();
    if (/^https?:\/\//i.test(url) || /^mailto:/i.test(url)) return url;
    if (/^\/samples\/[a-f0-9]{32}\.(png|jpe?g|webp)$/i.test(url)) return url;
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

  function parseMarkdown(source) {
    const lines = String(source || "").replace(/\r\n/g, "\n").split("\n");
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
          if (href && (/^https?:\/\//i.test(href) || href.startsWith("/samples/"))) {
            const image = document.createElement("img");
            image.alt = text.slice(index + 2, close);
            image.src = href;
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
    return {
      title: String((data && data.title) || "").trim(),
      kind: String((data && data.type) || "bar").toLowerCase() === "pie" ? "pie" : "bar",
      labels: labels.slice(0, size),
      values: values.slice(0, size),
    };
  }

  function renderBarChart(svg, labels, values) {
    const width = 520;
    const height = Math.max(160, 28 + labels.length * 34);
    const left = 120;
    const right = 56;
    const top = 8;
    const bottom = 8;
    const max = Math.max(...values, 1);
    const barH = Math.min(22, (height - top - bottom) / Math.max(labels.length, 1) - 8);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    labels.forEach((label, index) => {
      const value = values[index];
      const y = top + index * ((height - top - bottom) / labels.length) + 6;
      const barW = ((width - left - right) * value) / max;
      const name = document.createElementNS("http://www.w3.org/2000/svg", "text");
      name.setAttribute("x", String(left - 8));
      name.setAttribute("y", String(y + barH * 0.72));
      name.setAttribute("text-anchor", "end");
      name.setAttribute("font-size", "12");
      name.setAttribute("fill", "#6b7280");
      name.textContent = label.length > 16 ? label.slice(0, 15) + "…" : label;
      svg.appendChild(name);
      const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      bar.setAttribute("x", String(left));
      bar.setAttribute("y", String(y));
      bar.setAttribute("width", String(Math.max(barW, 2)));
      bar.setAttribute("height", String(barH));
      bar.setAttribute("rx", "4");
      bar.setAttribute("fill", CHART_COLORS[index % CHART_COLORS.length]);
      svg.appendChild(bar);
      const count = document.createElementNS("http://www.w3.org/2000/svg", "text");
      count.setAttribute("x", String(left + Math.max(barW, 2) + 6));
      count.setAttribute("y", String(y + barH * 0.72));
      count.setAttribute("font-size", "12");
      count.setAttribute("fill", "#12141a");
      count.textContent = String(value);
      svg.appendChild(count);
    });
  }

  function renderPieChart(svg, labels, values) {
    const width = 520;
    const height = 240;
    const cx = 120;
    const cy = 120;
    const radius = 88;
    const total = values.reduce((sum, value) => sum + value, 0) || 1;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    let angle = -Math.PI / 2;
    values.forEach((value, index) => {
      const slice = (value / total) * Math.PI * 2;
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
      const legendY = 28 + index * 22;
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
      const pct = Math.round((1000 * value) / total) / 10;
      text.textContent = `${labels[index]} — ${value} (${pct}%)`;
      svg.appendChild(text);
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
    svg.setAttribute("aria-label", series.title || (series.kind === "pie" ? "Pie chart" : "Bar chart"));
    if (series.kind === "pie") renderPieChart(svg, series.labels, series.values);
    else renderBarChart(svg, series.labels, series.values);
    wrap.appendChild(svg);
    return wrap;
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
      return table;
    }
    const paragraph = document.createElement("p");
    fillLines(paragraph, block.text);
    return paragraph;
  }

  function paint(node, text) {
    node.replaceChildren();
    parseMarkdown(text).forEach((block) => node.appendChild(renderBlock(block)));
  }

  global.FBMarkdown = { paint, parseMarkdown };
})(typeof window !== "undefined" ? window : globalThis);
