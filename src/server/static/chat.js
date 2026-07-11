const log = document.getElementById("log"), form = document.getElementById("form");
const input = document.getElementById("input"), sendBtn = document.getElementById("send");
const pill = document.getElementById("pill"), empty = document.getElementById("empty");
const messages = [];

function bubble(role, text, cls) {
  const row = document.createElement("div"); row.className = "row " + role;
  const b = document.createElement("div"); b.className = "bubble " + (cls || ""); b.textContent = text;
  row.appendChild(b); log.appendChild(row); log.scrollTop = log.scrollHeight; return b;
}
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; }
input.addEventListener("input", autosize);
input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });

const mem = document.getElementById("mem"), facts = document.getElementById("facts");
const memsub = document.getElementById("memsub"), bgstatus = document.getElementById("bgstatus");
let lastBrain = null;

function renderDrawerStatus(i) {
  if (!i) { bgstatus.style.display = "none"; return; }
  if (!i.enabled) { bgstatus.style.display = "none"; return; }
  bgstatus.style.display = "block";
  const loop = i.loop;
  if (!loop) {
    bgstatus.innerHTML = "Background: paused (manual — click Consolidate to learn).";
    return;
  }
  let label, live = false;
  if (loop.state === "dreaming") { label = "learning new facts…"; live = true; }
  else if (loop.state === "repolishing") { label = "polishing old facts…"; live = true; }
  else { label = loop.next_check_in_s > 0 ? ("idle · next check in ~" + loop.next_check_in_s + "s") : "about to check"; }
  let line = "Background: " + (live ? '<span class="pulse"></span>' : "") + '<span class="live">continuously learning</span> · ' + label;
  if (loop.cycles > 0) line += " · " + loop.cycles + " cycle" + (loop.cycles === 1 ? "" : "s") + " · last " + loop.last_outcome.replace("_committed", " ✓").replace("_reverted", " ✗");
  line += ' · <span class="ph">activity ▾</span>';
  bgstatus.innerHTML = line;
  const ph = bgstatus.querySelector(".ph");
  if (ph) ph.addEventListener("click", () => { mem.classList.remove("open"); actPanel.classList.add("open"); if (!actLoaded) loadActivity(); });
}

// ---- Activity feed ----
// NOTE: named actPanel (not act) — the pre-existing button-action helper below is
// already named `act`, and a const/function name collision is a fatal redeclaration
// that silently breaks the ENTIRE script (no handlers attach, send/return go dead)
const actPanel = document.getElementById("act"), actfeed = document.getElementById("actfeed");
const actlive = document.getElementById("actlive"), actfilt = document.getElementById("actfilt");
let actLoaded = false, actFilter = "all", actTimer = null, actSeen = 0;

const EVT = {
  candidate:  { cls: "note", ico: "•", lab: "scored", cat: "note",
    d: e => `surprise ${fmt(e.surprise)} · ${e.fired ? '<span style="color:var(--accent)">flagged</span>' : "skipped"} <span class="quote">${esc(e.text||"")}</span>` },
  experience: { cls: "note", ico: "★", lab: "noticed", cat: "note",
    d: e => `surprise ${fmt(e.surprise)} — queued for the next dream` },
  update:     { cls: "acc",  ico: "↻", lab: "trained", cat: "train",
    d: e => `${e.kind} · ${e.span_tokens}tok · KL ${fmt(e.span_kl)} · Δ ${fmt(e.delta_norm)} · ${Math.round(e.wall_ms)}ms` },
  rejected_update: { cls: "rej", ico: "↻", lab: "rejected", cat: "train",
    d: e => `${e.kind} · KL ${fmt(e.span_kl)} > budget (rolled back)` },
  absorb:     { cls: "acc", ico: "↻", lab: "absorbed", cat: "train", d: e => `user tokens` },
  atomize:    { cls: "note", ico: "⇶", lab: "split", cat: "note", d: e => `${e.count} facts from one turn` },
  dream:      { cls: "learn", ico: "✓", lab: "learned", cat: "learn",
    d: e => `${e.facts} fact${e.facts===1?"":"s"} · recall ${pct(e.recall)} · entropy ${fmt(e.entropy)} · sycophancy ${pct(e.sycophancy)}` },
  dream_reverted: { cls: "rej", ico: "✗", lab: "reverted", cat: "learn",
    d: e => `recall ${pct(e.recall)} below target (${e.reason}) — overlay restored` },
  repolish:   { cls: "learn", ico: "↻", lab: "polished", cat: "learn",
    d: e => `${e.facts} fact${e.facts===1?"":"s"} · recall ${pct(e.recall)}` },
  repolish_reverted: { cls: "rej", ico: "✗", lab: "polish reverted", cat: "learn",
    d: e => `recall ${pct(e.recall)} (${e.reason})` },
  checkpoint: { cls: "", ico: "⤓", lab: "checkpoint", cat: "op", d: e => e.reason || e.checkpoint_id },
  rollback:   { cls: "rej", ico: "⤴", lab: "rollback", cat: "op", d: e => e.checkpoint_id },
  consolidate:{ cls: "", ico: "⇪", lab: "consolidated base", cat: "op", d: e => e.serve_path },
  consolidate_reverted: { cls: "rej", ico: "⇪", lab: "base revert", cat: "op", d: e => e.serve_path },
  canary:     { cls: "", ico: "○", lab: "canary", cat: "op", d: e => `KL ${fmt(e.mean_kl)} · ${e.match_failures} fails` },
  dream_loop_error: { cls: "rej", ico: "!", lab: "loop error", cat: "op", d: e => esc(String(e.error||"").slice(0,80)) },
  worker_error: { cls: "rej", ico: "!", lab: "worker error", cat: "op", d: e => esc(String(e.error||"").slice(0,80)) },
};

function esc(s){ return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function fmt(x){ return x == null ? "—" : (typeof x === "number" ? (Math.abs(x) < 0.01 && x !== 0 ? x.toExponential(1) : x.toFixed(2)) : x); }
function pct(x){ return x == null ? "—" : Math.round(x*100) + "%"; }
function tstr(ts){ const d = new Date(ts*1000); return d.getHours().toString().padStart(2,"0") + ":" + d.getMinutes().toString().padStart(2,"0"); }

async function loadActivity() {
  actLoaded = true;
  try {
    const r = await (await fetch("/v1/brain/journal?limit=200")).json();
    renderFeed(r.events || []);
  } catch (e) { actfeed.innerHTML = '<div class="ev empty">could not load activity</div>'; }
  if (actPanel.classList.contains("open") && !actTimer) actTimer = setInterval(loadActivity, 2500);
}

function renderFeed(events) {
  // newest first; track the newest ts seen so the "live" dot shows when fresh
  const ordered = events.slice().reverse();
  if (ordered.length && ordered[0].at > actSeen) { actSeen = ordered[0].at; }
  actlive.style.display = ordered.length && Date.now()/1000 - ordered[0].at < 10 ? "flex" : "none";
  const filtered = ordered.filter(e => {
    const m = EVT[e.type];
    if (!m) return actFilter === "all";
    return actFilter === "all" || m.cat === actFilter;
  });
  if (!filtered.length) { actfeed.innerHTML = '<div class="ev empty">no ' + actFilter + ' events yet</div>'; return; }
  actfeed.innerHTML = filtered.map(e => {
    const m = EVT[e.type] || { cls: "", ico: "·", lab: e.type, cat: "op", d: () => "" };
    return `<div class="ev ${m.cls}"><span class="ico">${m.ico}</span><span class="t">${tstr(e.at)}</span><span class="body"><span class="lab">${m.lab}</span> <span class="d">${m.d(e)}</span></span></div>`;
  }).join("");
}

actfilt.addEventListener("click", e => {
  const s = e.target.closest("span[data-f]"); if (!s) return;
  actFilter = s.dataset.f;
  actfilt.querySelectorAll("span").forEach(x => x.classList.toggle("on", x === s));
  if (actLoaded) loadActivity();
});

async function refreshPill() {
  try {
    const b = await (await fetch("/v1/brain")).json();
    const i = b.individuation || {};
    const loop = i.loop || null;
    lastBrain = b;
    if (i.enabled && loop) {
      const st = loop.state;
      if (st === "dreaming") pill.textContent = "🧠 learning…";
      else if (st === "repolishing") pill.textContent = "🧠 polishing…";
      else pill.textContent = "🧠 " + (i.noted || 0) + " noticed · " + (i.learned || 0) + " learned";
    } else {
      pill.textContent = i.enabled ? ("🧠 " + (i.noted || 0) + " noticed · " + (i.learned || 0) + " learned") : "chat";
    }
    if (mem.classList.contains("open")) renderDrawerStatus(i);
  } catch (e) { pill.textContent = "offline"; }
}

function renderFacts(items) {
  facts.innerHTML = "";
  if (!items.length) { facts.innerHTML = '<div class="fact" style="color:var(--muted)">Nothing learned yet — chat, then hit “Consolidate”.</div>'; return; }
  for (const it of items) {
    const row = document.createElement("div"); row.className = "fact";
    const q = document.createElement("span"); q.className = "q"; q.textContent = it.question;
    row.appendChild(q);
    if ("recalled" in it) { const m = document.createElement("span"); m.className = "mark " + (it.recalled ? "ok" : "no"); m.textContent = it.recalled ? "✓ recalls" : "✗ forgot"; row.appendChild(m); }
    facts.appendChild(row);
  }
}

async function loadMemory() {
  const m = await (await fetch("/v1/brain/memory")).json();
  memsub.textContent = m.learned.length + " learned · " + m.noted + " noticed";
  renderFacts(m.learned);
  const i = lastBrain ? (lastBrain.individuation || {}) : {};
  if (i.loop !== undefined) renderDrawerStatus(i);
}

pill.addEventListener("click", () => {
  const opening = !mem.classList.contains("open");
  mem.classList.toggle("open", opening);
  if (opening) { actPanel.classList.remove("open"); loadMemory(); }
});
// pause the activity poll when its drawer is closed
const actObserver = new MutationObserver(() => {
  if (!actPanel.classList.contains("open") && actTimer) { clearInterval(actTimer); actTimer = null; }
  if (actPanel.classList.contains("open") && actLoaded && !actTimer) actTimer = setInterval(loadActivity, 2500);
});
actObserver.observe(actPanel, { attributes: true, attributeFilter: ["class"] });

async function act(btn, url, label, done) {
  const t = btn.textContent; btn.disabled = true; btn.textContent = label;
  try { done(await (await fetch(url, { method: "POST" })).json()); }
  catch (e) { memsub.textContent = "error: " + e.message; }
  finally { btn.disabled = false; btn.textContent = t; refreshPill(); }
}
document.getElementById("consolidate").addEventListener("click", e =>
  act(e.target, "/v1/brain/dream", "consolidating…", r => {
    memsub.textContent = r.committed ? ("learned " + r.facts_learned + " new (recall " + Math.round(r.recall * 100) + "%)") : ("nothing durable to learn (dropped " + r.dropped + ")");
    loadMemory();
  }));
document.getElementById("prove").addEventListener("click", e =>
  act(e.target, "/v1/brain/verify", "checking…", r => {
    memsub.textContent = "recalls " + Math.round(r.recall * 100) + "% of learned facts (live, cold)";
    renderFacts(r.items);
  }));

form.addEventListener("submit", async e => {
  e.preventDefault();
  const text = input.value.trim(); if (!text) return;
  if (empty) empty.remove();
  input.value = ""; autosize(); input.disabled = sendBtn.disabled = true;
  bubble("user", text); messages.push({ role: "user", content: text });
  const out = bubble("bot", "", "thinking blink");
  const reply = { role: "assistant", content: "" };
  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, stream: true, enable_thinking: false, max_tokens: 1024 })
    });
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n"); buf = parts.pop();
      for (const p of parts) {
        const line = p.trim(); if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim(); if (data === "[DONE]") continue;
        const delta = JSON.parse(data).choices?.[0]?.delta || {};
        if (delta.content) { reply.content += delta.content; out.className = "bubble blink"; out.textContent = reply.content; log.scrollTop = log.scrollHeight; }
      }
    }
    out.className = "bubble"; out.textContent = reply.content || "…";
    messages.push(reply);
  } catch (err) {
    out.className = "bubble"; out.textContent = "⚠︎ could not reach engram: " + err.message;
  } finally {
    input.disabled = sendBtn.disabled = false; input.focus(); refreshPill();
  }
});
refreshPill();
// poll the brain while the page is open so the pill reflects the background
// learner as it moves through learning/idle (cheap GET; 5s cadence)
setInterval(refreshPill, 5000);
