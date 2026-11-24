'use strict';

const $ = (id) => document.getElementById(id);
const LANG_NAME = { vi: 'Tiếng Việt', zh: 'Tiếng Trung' };
const OTHER = { vi: 'zh', zh: 'vi' };
const fmt = (x) => x.toLocaleString('vi-VN', { maximumFractionDigits: 1 });

const state = { models: [], limits: { max_chars: 5000, max_batch_lines: 64 } };

// ---------- tiện ích ----------
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    let msg = data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join('; ');
    throw new Error(msg || `Lỗi ${res.status}`);
  }
  return data;
}

function el(tag, props = {}, children = []) {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of [].concat(children)) if (child != null) node.append(child);
  return node;
}

let toastTimer;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2200);
}

function decodeParams() {
  return { beam_size: +$('s-beam').value, max_new_tokens: +$('s-max').value };
}

function modelById(id) { return state.models.find((m) => m.id === id); }

function scoreBadges(model, direction) {
  const s = model && model.scores && model.scores[direction];
  if (!s) return [];
  const out = [el('span', { className: 'badge score', textContent: `BLEU ${fmt(s.bleu)}` }),
               el('span', { className: 'badge score', textContent: `chrF ${fmt(s.chrf)}` })];
  if (s.comet != null) out.push(el('span', { className: 'badge score', textContent: `COMET ${fmt(s.comet)}` }));
  return out;
}

function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
  const a = el('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ---------- tab ----------
document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((t) => {
    t.classList.toggle('active', t === tab);
    t.setAttribute('aria-selected', t === tab);
  });
  document.querySelectorAll('.view').forEach((v) => { v.hidden = v.id !== `view-${tab.dataset.view}`; });
}));

// ---------- mô hình ----------
async function loadModels() {
  const data = await api('/api/models');
  state.models = data.models;
  state.limits = data.limits;
  renderModelStatus();
  renderModelSelects();
  renderCompareChecks();
}

function renderModelStatus() {
  $('m-status').replaceChildren(...state.models.map((m) => {
    const cls = !m.available ? 'off' : m.loaded ? 'loaded' : 'ready';
    const status = !m.available ? (m.reason || 'không dùng được') : m.loaded ? 'đã nạp' : 'chưa nạp (nạp ở lần dùng đầu)';
    return el('li', {}, [el('span', { className: `dot ${cls}` }), el('strong', { textContent: m.name }),
      el('span', { className: 'desc', textContent: `— ${status}` })]);
  }));
}

function renderModelSelects() {
  for (const select of document.querySelectorAll('.model-select')) {
    const current = select.value;
    select.replaceChildren(...state.models.map((m) => el('option', {
      value: m.id, disabled: !m.available,
      textContent: m.available ? m.name : `${m.name} (chưa có)`,
      title: m.available ? m.description : m.reason,
    })));
    const fallback = state.models.find((m) => m.available);
    select.value = current && modelById(current)?.available ? current : (fallback ? fallback.id : '');
  }
}

function renderCompareChecks() {
  const checked = new Set([...document.querySelectorAll('#c-models input:checked')].map((i) => i.value));
  const first = checked.size === 0;
  $('c-models').replaceChildren(...state.models.map((m) => {
    const input = el('input', { type: 'checkbox', value: m.id, disabled: !m.available,
      checked: m.available && (first || checked.has(m.id)) });
    return el('label', { className: m.available ? '' : 'disabled', title: m.available ? m.description : m.reason },
      [input, m.name]);
  }));
}

// Sau mỗi lần dịch, lấy lại trạng thái nạp (một lần nạp NLLB có thể bật nhiều mô hình cùng lúc)
function markLoaded(id) {
  const m = modelById(id);
  if (!m || m.loaded) return;
  m.loaded = true;
  renderModelStatus();
  api('/api/models').then((data) => {
    for (const fresh of data.models) Object.assign(modelById(fresh.id) || {}, { loaded: fresh.loaded });
    renderModelStatus();
  }).catch(() => {});
}

// ---------- tab Dịch ----------
const tInput = $('t-input');
let seq = 0;
let debounce;
let lastResult = null;

function updateTargetLabel(detected) {
  const src = $('t-source').value === 'auto' ? detected : $('t-source').value;
  $('t-target').textContent = src ? LANG_NAME[OTHER[src]] : 'Tiếng Trung';
}

function updateCounter() {
  const n = tInput.value.length;
  $('t-count').textContent = `${n.toLocaleString('vi-VN')} / ${state.limits.max_chars.toLocaleString('vi-VN')}`;
  $('t-count').classList.toggle('over', n > state.limits.max_chars);
}

function setOutput(content, loading = false) {
  const out = $('t-output');
  out.classList.toggle('loading', loading);
  out.replaceChildren(content);
}

async function translateNow() {
  const text = tInput.value;
  const id = ++seq;
  if (!text.trim()) {
    lastResult = null;
    setOutput(el('span', { className: 'placeholder', textContent: 'Bản dịch' }));
    $('t-meta').textContent = $('t-detected').textContent = '';
    $('t-pivot').hidden = true;
    updateTargetLabel(null);
    return;
  }
  const model = $('t-model').value;
  if (!model) return;
  const firstLoad = !modelById(model)?.loaded;
  setOutput(lastResult ? lastResult.translation : el('span', { className: 'placeholder', textContent: 'Đang dịch…' }), true);
  $('t-meta').textContent = firstLoad ? 'Đang nạp mô hình lần đầu, có thể mất vài chục giây…' : 'Đang dịch…';
  $('t-go').disabled = true;
  try {
    const r = await api('/api/translate', { text, model, source: $('t-source').value, ...decodeParams() });
    if (id !== seq) return;   // đã có yêu cầu mới hơn
    lastResult = r;
    markLoaded(model);
    setOutput(r.translation);
    $('t-detected').textContent = r.detected ? `Đã nhận diện: ${LANG_NAME[r.source]}` : '';
    updateTargetLabel(r.source);
    $('t-meta').textContent = `${modelById(model).name} · ${r.sentences} câu · ${fmt(r.elapsed_ms / 1000)} giây`;
    $('t-pivot').hidden = !r.pivot;
    $('t-pivot-text').textContent = r.pivot || '';
  } catch (err) {
    if (id !== seq) return;
    lastResult = null;
    setOutput(el('span', { className: 'error', textContent: err.message }));
    $('t-meta').textContent = '';
  } finally {
    if (id === seq) $('t-go').disabled = false;
  }
}

tInput.addEventListener('input', () => {
  updateCounter();
  clearTimeout(debounce);
  if ($('t-auto').checked) debounce = setTimeout(translateNow, 700);
});
tInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); clearTimeout(debounce); translateNow(); }
});
$('t-go').addEventListener('click', () => { clearTimeout(debounce); translateNow(); });
$('t-model').addEventListener('change', () => { lastResult = null; if (tInput.value.trim()) translateNow(); });
$('t-source').addEventListener('change', () => { updateTargetLabel(lastResult?.source); if (tInput.value.trim()) translateNow(); });
$('t-clear').addEventListener('click', () => { tInput.value = ''; updateCounter(); translateNow(); tInput.focus(); });
$('t-copy').addEventListener('click', async () => {
  if (!lastResult) return;
  await navigator.clipboard.writeText(lastResult.translation);
  toast('Đã sao chép bản dịch');
});
$('t-swap').addEventListener('click', () => {
  const current = $('t-source').value === 'auto' ? lastResult?.source : $('t-source').value;
  if (!current) { toast('Nhập câu trước rồi mới đảo chiều'); return; }
  $('t-source').value = OTHER[current];
  if (lastResult) tInput.value = lastResult.translation;
  updateCounter();
  lastResult = null;
  translateNow();
});
document.querySelectorAll('[data-example]').forEach((chip) => chip.addEventListener('click', () => {
  tInput.value = chip.dataset.example;
  $('t-source').value = 'auto';
  updateCounter();
  translateNow();
}));

// ---------- tab So sánh ----------
async function compareNow() {
  const text = $('c-input').value;
  const models = [...document.querySelectorAll('#c-models input:checked')].map((i) => i.value);
  if (!text.trim()) { toast('Nhập câu cần so sánh'); return; }
  if (!models.length) { toast('Chọn ít nhất một mô hình'); return; }
  const cards = new Map(models.map((id) => {
    const m = modelById(id);
    const card = el('div', { className: 'card pending' }, [
      el('h3', { textContent: m.name }),
      el('div', { className: 'text', textContent: m.loaded ? 'Đang dịch…' : 'Đang nạp mô hình…' })]);
    return [id, card];
  }));
  $('c-results').replaceChildren(...cards.values());
  $('c-go').disabled = true;
  try {
    // Gọi lần lượt từng mô hình để kết quả hiện dần (server cũng chỉ chạy một mô hình mỗi lúc)
    for (const id of models) {
      let r;
      try {
        [r] = (await api('/api/compare', { text, models: [id], source: $('c-source').value, ...decodeParams() })).results;
      } catch (err) {
        r = { model: id, error: err.message };
      }
      cards.get(id).replaceWith(renderCompareCard(r));
      cards.set(id, null);
      if (!r.error) markLoaded(id);
    }
  } finally {
    $('c-go').disabled = false;
  }
}

function renderCompareCard(r) {
  const m = modelById(r.model);
  if (r.error) {
    return el('div', { className: 'card' }, [el('h3', { textContent: m ? m.name : r.model }),
      el('div', { className: 'text error', textContent: r.error })]);
  }
  const direction = `${r.source}-${r.target}`;
  return el('div', { className: 'card' }, [
    el('h3', { textContent: m.name }),
    el('div', { className: 'text', textContent: r.translation }),
    r.pivot ? el('div', { className: 'pivot-line', textContent: `EN: ${r.pivot}` }) : null,
    el('div', { className: 'foot' }, [
      el('span', { className: 'badge', textContent: `${r.source.toUpperCase()} → ${r.target.toUpperCase()}` }),
      el('span', { className: 'badge', textContent: `${fmt(r.elapsed_ms / 1000)} giây` }),
      ...scoreBadges(m, direction)]),
  ]);
}

$('c-go').addEventListener('click', compareNow);
$('c-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); compareNow(); }
});

// ---------- tab Dịch tệp ----------
let batchAbort = null;
let batchData = null;

async function readFile(file) {
  if (!file) return;
  $('b-input').value = await file.text();
  $('b-file-name').textContent = `${file.name} · ${$('b-input').value.split(/\r?\n/).length.toLocaleString('vi-VN')} dòng`;
}

$('b-file').addEventListener('change', (e) => readFile(e.target.files[0]));
const drop = $('b-drop');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); readFile(e.dataTransfer.files[0]); });

async function batchNow() {
  let lines = $('b-input').value.replace(/\r\n?/g, '\n').split('\n');
  while (lines.length && !lines[lines.length - 1].trim()) lines.pop();   // bỏ dòng trống cuối tệp
  if (!lines.some((l) => l.trim())) { toast('Chưa có nội dung để dịch'); return; }
  const model = $('b-model').value;
  const chunk = Math.min(16, state.limits.max_batch_lines);
  const translations = [];
  const controller = new AbortController();
  batchAbort = controller;
  $('b-go').disabled = true;
  $('b-cancel').hidden = false;
  $('b-progress').hidden = false;
  $('b-bar').style.width = '0';
  const started = performance.now();
  try {
    for (let i = 0; i < lines.length; i += chunk) {
      $('b-status').textContent = `Đang dịch ${Math.min(i + chunk, lines.length).toLocaleString('vi-VN')} / ${lines.length.toLocaleString('vi-VN')} dòng…`;
      const res = await fetch('/api/batch', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ lines: lines.slice(i, i + chunk), model, source: $('b-source').value, ...decodeParams() }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Lỗi ${res.status}`);
      translations.push(...data.translations);
      markLoaded(model);
      $('b-bar').style.width = `${(100 * translations.length) / lines.length}%`;
    }
    $('b-status').textContent = `Xong ${lines.length.toLocaleString('vi-VN')} dòng trong ${fmt((performance.now() - started) / 1000)} giây`;
  } catch (err) {
    $('b-status').textContent = err.name === 'AbortError'
      ? `Đã dừng sau ${translations.length.toLocaleString('vi-VN')} dòng` : `Lỗi: ${err.message}`;
  } finally {
    batchAbort = null;
    $('b-go').disabled = false;
    $('b-cancel').hidden = true;
  }
  if (translations.length) {
    batchData = { lines: lines.slice(0, translations.length), translations };
    renderBatchTable();
  }
}

function renderBatchTable() {
  const { lines, translations } = batchData;
  const shown = Math.min(lines.length, 500);
  const rows = [];
  for (let i = 0; i < shown; i++) {
    rows.push(el('tr', {}, [el('td', { textContent: i + 1 }), el('td', { textContent: lines[i] }),
      el('td', { textContent: translations[i] })]));
  }
  if (lines.length > shown) {
    rows.push(el('tr', {}, [el('td', {}), el('td', { colSpan: 2, className: 'hint',
      textContent: `… còn ${(lines.length - shown).toLocaleString('vi-VN')} dòng nữa, tải tệp để xem đủ` })]));
  }
  $('b-rows').replaceChildren(...rows);
  $('b-result').hidden = false;
}

$('b-go').addEventListener('click', batchNow);
$('b-cancel').addEventListener('click', () => batchAbort && batchAbort.abort());
$('b-dl-txt').addEventListener('click', () => batchData && download('ban_dich.txt', batchData.translations.join('\n') + '\n'));
$('b-dl-tsv').addEventListener('click', () => {
  if (!batchData) return;
  const clean = (s) => s.replace(/\t/g, ' ');
  const body = batchData.lines.map((l, i) => `${clean(l)}\t${clean(batchData.translations[i])}`).join('\n');
  download('song_ngu.tsv', `nguon\tban_dich\n${body}\n`);
});

// ---------- tùy chọn ----------
for (const [input, output] of [['s-beam', 'o-beam'], ['s-max', 'o-max']]) {
  $(input).addEventListener('input', () => { $(output).textContent = $(input).value; });
}

loadModels().then(updateCounter).catch((err) => toast(`Không kết nối được server: ${err.message}`));
