/* RAG Codebase Q&A -- vanilla JS client, no build step. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const chat = $("#chat");

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

/* ---------- status + repos ---------- */

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    $("#status-line").textContent =
      `${s.num_repos} repo(s) indexed - embedder: ${s.embedder} - default answerer: ${s.answer_provider_default}`;
  } catch (e) {
    $("#status-line").textContent = "backend unreachable";
  }
}

async function refreshRepos(preferred) {
  const select = $("#repo-select");
  const { repos } = await api("/api/repos");
  select.innerHTML = "";
  for (const r of repos) {
    const opt = el("option", null, `${r.name} (${r.num_chunks} chunks)`);
    opt.value = r.name;
    select.appendChild(opt);
  }
  if (preferred) select.value = preferred;
  return repos;
}

/* ---------- indexing ---------- */

$("#index-btn").addEventListener("click", async () => {
  const source = $("#index-source").value.trim();
  if (!source) return;
  const btn = $("#index-btn");
  btn.disabled = true;
  $("#index-status").textContent = "indexing... (sync, may take a moment)";
  try {
    const { stats } = await api("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    $("#index-status").textContent =
      `indexed '${stats.repo}': ${stats.num_files} files -> ${stats.num_chunks} chunks in ${stats.seconds}s`;
    await refreshRepos(stats.repo);
    await refreshStatus();
  } catch (e) {
    $("#index-status").textContent = `error: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
});

/* ---------- asking ---------- */

function addUserMsg(text) {
  const node = el("div", "msg user", text);
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
}

function scoreBar(value, max) {
  const wrap = el("span", "score-bar");
  const fill = el("i");
  fill.style.width = `${max > 0 ? Math.round((value / max) * 100) : 0}%`;
  wrap.appendChild(fill);
  return wrap;
}

function renderChunks(chunks) {
  const details = el("details", "context");
  const summary = el("summary", null, `Retrieved context (${chunks.length} chunks with scores)`);
  details.appendChild(summary);
  const maxFused = Math.max(...chunks.map((c) => c.fused_score), 0);
  for (const c of chunks) {
    const box = el("div", "chunk");
    const head = el("div", "chunk-head");
    const left = el("span", "chunk-path", `#${c.rank}  ${c.path}:${c.start_line}-${c.end_line}`);
    if (c.symbol) {
      left.appendChild(document.createTextNode("  "));
      left.appendChild(el("span", "chunk-sym", `${c.symbol} (${c.kind})`));
    }
    const scores = el("span", "scores");
    scores.append("fused ");
    scores.appendChild(el("b", null, c.fused_score.toFixed(3)));
    scores.appendChild(scoreBar(c.fused_score, maxFused));
    scores.append(` vec ${c.vector_score.toFixed(3)} - bm25 ${c.bm25_score.toFixed(2)}`);
    head.append(left, scores);
    box.appendChild(head);
    const pre = el("pre", null, c.text + (c.text_truncated ? "\n..." : ""));
    box.appendChild(pre);
    details.appendChild(box);
  }
  return details;
}

function addAnswerMsg(resp) {
  const node = el("div", "msg bot");
  node.appendChild(el("div", "answer-text", resp.answer.text));

  if (resp.answer.citations.length) {
    const cites = el("div", "citations");
    for (const c of resp.answer.citations) {
      cites.appendChild(el("span", "citation-chip", c.label));
    }
    node.appendChild(cites);
  }
  node.appendChild(
    el(
      "div",
      "answer-meta",
      `repo: ${resp.repo} - retrieval: ${resp.mode} - answerer: ${resp.answer.provider}` +
        (resp.answer.grounded ? "" : " - (insufficient context)")
    )
  );
  if (resp.chunks.length) node.appendChild(renderChunks(resp.chunks));
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
}

function addErrorMsg(text) {
  chat.appendChild(el("div", "msg bot error", `Error: ${text}`));
  chat.scrollTop = chat.scrollHeight;
}

$("#ask-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("#question").value.trim();
  if (!q) return;
  $("#question").value = "";
  addUserMsg(q);

  const thinking = el("div", "msg bot spinner", "retrieving + answering...");
  chat.appendChild(thinking);
  chat.scrollTop = chat.scrollHeight;
  $("#ask-btn").disabled = true;

  const payload = {
    question: q,
    mode: $("#mode-select").value,
    top_k: 8,
  };
  const repo = $("#repo-select").value;
  if (repo) payload.repo = repo;
  const provider = $("#provider-select").value;
  if (provider) payload.provider = provider;

  try {
    const resp = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    thinking.remove();
    addAnswerMsg(resp);
  } catch (e) {
    thinking.remove();
    addErrorMsg(e.message);
  } finally {
    $("#ask-btn").disabled = false;
  }
});

/* ---------- boot ---------- */
(async function boot() {
  await refreshStatus();
  try {
    await refreshRepos();
  } catch (e) {
    /* no repos yet */
  }
})();
