/* IDX Intelligence frontend application logic.
 * Talks only to the backend REST API; the AI agent runs server-side (Qwen).
 */
'use strict';

/* ---------------- helpers ---------------- */
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

async function api(path, opts = {}) {
  const res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, opts));
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) throw new Error((body && body.error) || ('HTTP ' + res.status));
  return body;
}
function qs(obj) {
  const p = new URLSearchParams();
  Object.entries(obj || {}).forEach(([k, v]) => { if (v !== null && v !== undefined && v !== '') p.set(k, v); });
  const s = p.toString(); return s ? '?' + s : '';
}
const rp = (n) => (n == null || isNaN(n)) ? '—' : 'Rp ' + Math.round(n).toLocaleString('en-US');
const rp2 = (n) => (n == null || isNaN(n)) ? '—' : 'Rp ' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct = (n, d = 2) => (n == null || isNaN(n)) ? '—' : (n >= 0 ? '+' : '') + Number(n).toFixed(d) + '%';
const num = (n, d = 2) => (n == null || isNaN(n)) ? '—' : Number(n).toLocaleString('en-US', { maximumFractionDigits: d });
const cls = (n) => (n == null || isNaN(n)) ? 'mut' : (n >= 0 ? 'pos' : 'neg');
const arrow = (n) => (n == null || isNaN(n)) ? '' : (n >= 0 ? 'arrow_drop_up' : 'arrow_drop_down');
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* Minimal, XSS-safe markdown renderer for agent replies.
 * Escapes first, then applies a small whitelist of formatting. */
function mdRender(src) {
  const inline = (t) => esc(t)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  const lines = String(src == null ? '' : src).split(/\r?\n/);
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push('</' + list + '>'); list = null; } };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    if (!line.trim()) { closeList(); continue; }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (h) { closeList(); out.push('<div class="md-h">' + inline(h[2]) + '</div>'); continue; }
    if (ul) { if (list !== 'ul') { closeList(); out.push('<ul class="md-ul">'); list = 'ul'; } out.push('<li>' + inline(ul[1]) + '</li>'); continue; }
    if (ol) { if (list !== 'ol') { closeList(); out.push('<ol class="md-ol">'); list = 'ol'; } out.push('<li>' + inline(ol[1]) + '</li>'); continue; }
    closeList(); out.push('<div>' + inline(line) + '</div>');
  }
  closeList();
  return out.join('');
}

/* ---------------- state ---------------- */
const state = {
  stocks: [], selected: 'PGEO', view: 'terminal',
  asOf: null, latestDate: null, range: '6M',
  chatHistory: [], currentReportId: null,
  charts: { price: null, pva: null },
  qwenConfigured: false,
};

/* ---------------- boot ---------------- */
document.addEventListener('DOMContentLoaded', async () => {
  wireNav(); wireReplay(); wireReports(); wireHistory(); wireChat(); wireWatchSearch(); wireRangeTabs();
  try {
    const h = await api('/api/health');
    state.latestDate = h.latest_dataset_date;
    state.qwenConfigured = h.qwen_configured;
    $('#data-date-text').textContent = 'Latest data: ' + (h.latest_dataset_date || '—');
    $('#watch-run').textContent = 'Run: ' + (h.latest_dataset_date || '—');
    $('#agent-dot').className = 'h-2 w-2 rounded-full ' + (h.qwen_configured ? 'bg-emerald-500' : 'bg-slate-300');
    $('#agent-model').textContent = h.qwen_configured ? 'Qwen Agent • connected' : 'Qwen Agent • not configured';
    const rd = $('#replay-date'); rd.max = h.latest_dataset_date || '';
  } catch (e) { $('#data-date-text').textContent = 'backend offline'; }
  await loadStocks();
  renderChatQuick();
  const hv = (location.hash || '').replace(/^#\/?/, '');
  setView(['terminal', 'reports', 'history'].includes(hv) ? hv : 'terminal');
  window.addEventListener('hashchange', () => {
    const v = (location.hash || '').replace(/^#\/?/, '');
    if (['terminal', 'reports', 'history'].includes(v) && v !== state.view) setView(v);
  });
});

/* ---------------- view switching ---------------- */
function wireNav() {
  $$('.nav-tab').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
}
function setView(v) {
  state.view = v;
  $$('.nav-tab').forEach((b) => {
    const on = b.dataset.view === v;
    b.className = 'nav-tab px-3 py-1 rounded ' + (on ? 'bg-white text-emerald-800 font-bold shadow-sm' : 'text-slate-600 hover:text-slate-900');
  });
  ['terminal', 'reports', 'history'].forEach((id) => $('#view-' + id).classList.toggle('hidden', id !== v));
  try { if (location.hash !== '#/' + v) history.replaceState(null, '', '#/' + v); } catch (e) { /* ignore */ }
  if (v === 'terminal') loadTerminal();
  if (v === 'history') loadHistory();
}

/* ---------------- replay ---------------- */
function wireReplay() {
  const tog = $('#replay-toggle'), dt = $('#replay-date');
  tog.addEventListener('change', () => {
    if (tog.checked) {
      dt.disabled = false;
      if (!dt.value) dt.value = shiftMonths(state.latestDate, -6);
      state.asOf = dt.value;
    } else { dt.disabled = true; state.asOf = null; }
    applyReplay();
  });
  dt.addEventListener('change', () => { if (tog.checked) { state.asOf = dt.value || null; applyReplay(); } });
  $('#replay-exit').addEventListener('click', () => { tog.checked = false; dt.disabled = true; state.asOf = null; applyReplay(); });
}
function applyReplay() {
  const on = !!state.asOf;
  $('#replay-banner').classList.toggle('hidden', !on);
  if (on) $('#replay-label').textContent = state.asOf;
  refreshCurrentView();
}
function refreshCurrentView() {
  if (state.view === 'terminal') loadTerminal();
  else if (state.view === 'history') loadHistory();
  // reports view keeps its preview until regenerated
}
function shiftMonths(iso, m) {
  if (!iso) return '';
  const d = new Date(iso); d.setMonth(d.getMonth() + m); return d.toISOString().slice(0, 10);
}

/* ---------------- watchlist ---------------- */
async function loadStocks() {
  try {
    const j = await api('/api/stocks');
    state.stocks = j.stocks || [];
    renderWatchlist();
    populateReportStocks();
  } catch (e) { $('#watchlist').innerHTML = '<div class="p-3 text-body-sm text-rose-600">Failed to load stocks.</div>'; }
}
function renderWatchlist(filter) {
  const f = (filter || '').toLowerCase();
  const rows = state.stocks.filter((s) => !f || s.code.toLowerCase().includes(f) || (s.name || '').toLowerCase().includes(f) || (s.energy_type || '').toLowerCase().includes(f));
  $('#watchlist').innerHTML = rows.map((s) => {
    const on = s.code === state.selected;
    const ch = null; // change not in list payload; show last close only
    return `<div data-code="${s.code}" role="button" tabindex="0" aria-label="Select ${s.code}" class="stock-row p-3 cursor-pointer transition-colors ${on ? 'bg-emerald-50/50 border-l-[3px] border-primary' : 'hover:bg-slate-50'}">
      <div class="flex items-center justify-between">
        <div>
          <div class="flex items-center gap-1.5">
            <span class="font-label-numeric-md font-bold ${on ? 'text-emerald-800' : 'text-slate-800'}">${s.code}</span>
            ${on ? '<span class="px-1 py-0.2 rounded bg-emerald-100 text-emerald-800 border border-emerald-300 text-[9px] font-label-tag font-semibold">ACTIVE</span>' : ''}
          </div>
          <p class="text-[11px] text-slate-500 truncate max-w-[130px]">${esc(s.name)}</p>
        </div>
        <div class="text-right">
          <span class="font-label-numeric-md font-bold text-slate-900 block">${rp(s.last_close)}</span>
          <span class="font-label-numeric-sm text-slate-400 text-[10px]">${esc(s.energy_type || '')}</span>
        </div>
      </div>
    </div>`;
  }).join('');
  const pick = (el) => { state.selected = el.dataset.code; renderWatchlist($('#watch-search').value); if (state.view !== 'terminal') setView('terminal'); else loadTerminal(); };
  $$('.stock-row').forEach((el) => {
    el.addEventListener('click', () => pick(el));
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(el); } });
  });
}
function wireWatchSearch() { $('#watch-search').addEventListener('input', (e) => renderWatchlist(e.target.value)); }

/* ---------------- terminal ---------------- */
async function loadTerminal() {
  const code = state.selected;
  // skeletons
  $('#t-lastclose').innerHTML = '<span class="skeleton inline-block h-6 w-24"></span>';
  try {
    const snap = await api('/api/stocks/' + code + qs({ as_of: state.asOf }));
    renderSnapshot(snap);
  } catch (e) {
    $('#t-subtitle').textContent = 'Error: ' + e.message;
  }
  await loadCharts();
}

function renderSnapshot(s) {
  $('#t-ticker').textContent = s.stock;
  const en = $('#t-energy');
  en.textContent = 'EBT STUDY: ' + (s.energy_type || 'RENEWABLE').toUpperCase();
  en.title = 'Project EBT study classification (Stats Description sheet), not a live sector feed';
  const cl = $('#t-cluster');
  cl.textContent = s.cluster ? ('GROUP ' + s.cluster) : '';
  cl.title = en.title;
  $('#t-subtitle').textContent = `${s.name || ''} • Exclusively Daily Close ML Engine`;
  $('#t-lastclose').textContent = rp(s.last_close);
  const ch = $('#t-change');
  ch.className = 'font-label-numeric-sm text-label-numeric-sm flex items-center justify-end font-bold ' + cls(s.change_pct);
  ch.innerHTML = s.change_pct == null ? '—' : `<span class="material-symbols-outlined text-xs">${arrow(s.change_pct)}</span>${rp(s.change)} (${pct(s.change_pct)}) Previous Close`;
  $('#t-eod').textContent = 'EOD Close Market: ' + (s.as_of || '—');
}

/* ---------------- charts ---------------- */
function wireRangeTabs() {
  $$('#t-range-tabs button').forEach((b) => b.addEventListener('click', () => {
    state.range = b.dataset.range;
    $$('#t-range-tabs button').forEach((x) => x.className = 'px-2 py-0.5 ' + (x === b ? 'bg-white text-emerald-800 font-bold rounded shadow-sm' : 'text-slate-600 hover:text-slate-900'));
    loadCharts();
  }));
}
function rangeStart(latestISO, range) {
  if (!latestISO || range === 'ALL') return '';
  const d = new Date(latestISO);
  if (range === '1M') d.setMonth(d.getMonth() - 1);
  else if (range === '3M') d.setMonth(d.getMonth() - 3);
  else if (range === '6M') d.setMonth(d.getMonth() - 6);
  else if (range === '1Y') d.setFullYear(d.getFullYear() - 1);
  return d.toISOString().slice(0, 10);
}
async function loadCharts() {
  const code = state.selected;
  const latest = state.asOf || state.latestDate;
  const start = rangeStart(latest, state.range);
  try {
    const [h, pva] = await Promise.all([
      api('/api/stocks/' + code + '/history' + qs({ start, as_of: state.asOf })),
      api('/api/stocks/' + code + '/prediction-vs-actual' + qs({ start, limit: 500, as_of: state.asOf })),
    ]);
    renderKpis(h.series || []);
    renderPriceChart(h.series || [], pva.series || []);
  } catch (e) { /* keep prior charts */ }
}
function renderKpis(series) {
  const closes = series.map((r) => r.close).filter((c) => c != null);
  if (!closes.length) { $('#t-kpis').innerHTML = ''; return; }
  const last = closes[closes.length - 1];
  const avg5 = closes.slice(-5).reduce((a, b) => a + b, 0) / Math.min(5, closes.length);
  const rets = [];
  for (let i = 1; i < closes.length; i++) rets.push((closes[i] - closes[i - 1]) / closes[i - 1] * 100);
  const vol30 = rets.slice(-30);
  const sd = vol30.length > 1 ? Math.sqrt(vol30.reduce((a, b) => a + b * b, 0) / vol30.length) : null;
  const hi = Math.max(...closes), lo = Math.min(...closes);
  const d5 = avg5 ? (last - avg5) / avg5 * 100 : null;
  const card = (t, v, sub) => `<div class="p-3 rounded bg-white border border-slate-200 shadow-sm"><div class="text-[10px] text-slate-400 uppercase font-label-tag font-semibold tracking-wider">${t}</div><div class="font-bold text-slate-800 text-base mt-1">${v} ${sub || ''}</div></div>`;
  $('#t-kpis').innerHTML =
    card('5-Day Close Avg', rp(avg5), `<span class="${cls(d5)} font-medium text-xs ml-1">(${pct(d5, 1)})</span>`) +
    card('30D Close Volatility (σ)', sd != null ? sd.toFixed(2) + '%' : '—', `<span class="text-slate-400 font-normal text-xs ml-1">${sd != null && sd < 1.5 ? 'Low' : sd != null && sd < 3 ? 'Moderate' : 'High'}</span>`) +
    card('Range High', rp(hi)) +
    card('Range Low', rp(lo));
}
function destroyChart(k) { if (state.charts[k]) { state.charts[k].destroy(); state.charts[k] = null; } }
function renderPriceChart(series, pvaSeries) {
  if (typeof Chart === 'undefined') return;
  destroyChart('price');
  const predByDate = {}; (pvaSeries || []).forEach((r) => { if (r.date) predByDate[r.date] = r.predicted; });
  const labels = series.map((r) => r.date);
  const close = series.map((r) => r.close);
  const pred = labels.map((d) => (predByDate[d] != null ? predByDate[d] : null));
  const ctx = $('#chart-price').getContext('2d');
  state.charts.price = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [
      { label: 'Daily Close (EOD)', data: close, borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.08)', fill: true, tension: 0.25, pointRadius: 0, borderWidth: 2 },
      { label: 'Model Predicted (T+1)', data: pred, borderColor: '#0284c7', borderDash: [4, 3], pointRadius: 0, borderWidth: 1.5, fill: false, spanGaps: false },
    ]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => c.dataset.label + ': ' + rp(c.parsed.y) } } },
      scales: {
        x: { ticks: { maxTicksLimit: 6, font: { family: 'JetBrains Mono', size: 9 }, color: '#64748b' }, grid: { color: '#f1f5f9' } },
        y: { ticks: { font: { family: 'JetBrains Mono', size: 9 }, color: '#64748b', callback: (v) => num(v, 0) }, grid: { color: '#e2e8f0' } },
      },
    },
  });
}
/* ---------------- reports ---------------- */
function populateReportStocks() {
  $('#r-stock').innerHTML = state.stocks.map((s) => `<option value="${s.code}">${s.code} — ${esc(s.name)}</option>`).join('');
  $('#r-stock').value = state.selected;
}
function wireReports() {
  $('#r-type').addEventListener('change', () => {
    const custom = $('#r-type').value === 'custom';
    $('#r-start-wrap').classList.toggle('hidden', !custom);
    $('#r-end-wrap').classList.toggle('hidden', !custom);
  });
  $('#r-generate').addEventListener('click', doGenerateReport);
  $('#r-pdf').addEventListener('click', doGeneratePdf);
  $('#r-email').addEventListener('click', () => $('#r-email-row').classList.toggle('hidden'));
  $('#r-email-send').addEventListener('click', doSendEmail);
}
function reportParams() {
  const t = $('#r-type').value;
  return {
    stock_code: $('#r-stock').value, report_type: t,
    start: t === 'custom' ? $('#r-start').value : undefined,
    end: t === 'custom' ? $('#r-end').value : undefined,
    as_of: state.asOf || undefined,
  };
}
async function doGenerateReport() {
  const st = $('#r-status'); st.innerHTML = '<span class="spinner"></span> analysing…';
  try {
    const j = await api('/api/reports/generate', { method: 'POST', body: JSON.stringify(reportParams()) });
    state.currentReportId = j.report_id;
    $('#r-pdf').disabled = false; $('#r-email').disabled = false;
    renderReportPreview(j.report, j.report_id);
    st.innerHTML = `<span class="pos">report ${j.report_id} generated</span>`;
  } catch (e) { st.innerHTML = `<span class="neg">${esc(e.message)}</span>`; }
}
async function doGeneratePdf() {
  if (!state.currentReportId) return;
  const st = $('#r-status'); st.innerHTML = '<span class="spinner"></span> rendering PDF…';
  try {
    const j = await api(`/api/reports/${state.currentReportId}/pdf`, { method: 'POST', body: JSON.stringify({}) });
    st.innerHTML = `PDF ready: <a class="text-sky-700 underline" target="_blank" href="${j.pdf_url}">${esc(j.pdf_filename)}</a>`;
  } catch (e) { st.innerHTML = `<span class="neg">${esc(e.message)}</span>`; }
}
async function doSendEmail() {
  const to = $('#r-email-input').value.trim();
  const st = $('#r-status');
  if (!to) { st.innerHTML = '<span class="neg">enter a recipient email</span>'; return; }
  st.innerHTML = '<span class="spinner"></span> sending…';
  try {
    const j = await api(`/api/reports/${state.currentReportId}/email`, { method: 'POST', body: JSON.stringify({ recipient: to }) });
    st.innerHTML = j.success ? `<span class="pos">emailed to ${esc(to)}</span>` : `<span class="neg">${esc(j.error)}</span>`;
  } catch (e) { st.innerHTML = `<span class="neg">${esc(e.message)}</span>`; }
}
function kv(k, v) { return `<div class="k">${esc(k)}</div><div class="v">${v}</div>`; }
function renderReportPreview(r, id) {
  const s = r.statistics || {}, p = r.prediction || {}, pva = (r.prediction_vs_actual || {}).metrics || {};
  const per = r.period || {};
  const sec = (h, body) => `<div class="rp-section"><div class="rp-h">${h}</div>${body}</div>`;
  let html = '';
  html += sec('Report — ' + esc(r.stock) + ' (' + esc(r.report_type) + ')',
    `<div class="rp-kv">` +
    kv('Period', `${per.start} → ${per.end}`) +
    kv('Label', esc(per.label || '')) +
    kv('Generated', esc(r.generated_at || '')) +
    kv('Replay', r.replay ? 'YES (as of ' + esc(r.as_of) + ')' : 'no') +
    kv('Latest dataset date', esc(r.latest_dataset_date || '')) +
    `</div>`);
  if (s.available) {
    html += sec('Period statistics', `<div class="rp-kv">` +
      kv('Trading days', s.trading_days) +
      kv('Start / End price', `${rp(s.start_price)} / ${rp(s.end_price)}`) +
      kv('Change', `<span class="${cls(s.change_pct)}">${rp(s.change)} (${pct(s.change_pct)})</span>`) +
      kv('High / Low', `${rp(s.high)} / ${rp(s.low)}`) +
      kv('Mean / Median', `${rp(s.mean)} / ${rp(s.median)}`) +
      kv('Volatility (σ)', num(s.volatility_std, 2)) +
      kv('Avg volume', num(s.avg_volume, 0)) +
      `</div>`);
  } else {
    html += sec('Period statistics', `<div class="text-body-sm text-slate-500">${esc(s.reason || 'No data.')}</div>`);
  }
  if (p.available) {
    html += sec('Model prediction (T+1)', `<div class="rp-kv">` +
      kv('Cutoff (as-of)', esc(p.as_of || '')) +
      kv('Close at cutoff', rp(p.as_of_close)) +
      kv('Predicted next close', `<b>${rp(p.predicted_close)}</b> <span class="${cls(p.predicted_change_pct)}">(${pct(p.predicted_change_pct)})</span>`) +
      kv('FCM cluster', `C${p.cluster} (U=${p.membership_max})`) +
      (p.actual_close != null ? kv('Actual (eval)', `${rp(p.actual_close)} on ${esc(p.predicted_for_date)} → err ${rp(p.error)} (${p.ape}%)`) : kv('Actual (eval)', 'not available yet')) +
      `</div>`);
  }
  html += sec('Accuracy in period', `<div class="rp-kv">` +
    kv('Pairs', pva.n != null ? pva.n : (r.prediction_vs_actual || {}).count) +
    kv('MAE', num(pva.mae, 2)) + kv('RMSE', num(pva.rmse, 2)) + kv('MAPE', pva.mape != null ? pva.mape.toFixed(2) + '%' : '—') + `</div>`);
  if ((r.notable_movements || []).length) {
    html += sec('Notable movements', `<div class="rp-kv">` + r.notable_movements.map((n) =>
      kv(n.date, `${rp(n.close)} <span class="${cls(n.change_pct)}">${pct(n.change_pct)}</span>`)).join('') + `</div>`);
  }
  html += sec('Summary', `<div class="text-body-sm text-slate-700 leading-relaxed">${esc(r.summary_text || '')}</div>`);
  $('#r-preview').innerHTML = html;
}

/* ---------------- history ---------------- */
function wireHistory() { $('#h-refresh').addEventListener('click', loadHistory); }
async function loadHistory() {
  const box = $('#h-table');
  box.innerHTML = '<div class="p-4 text-body-sm text-slate-500"><span class="spinner"></span> loading…</div>';
  try {
    const j = await api('/api/reports/history' + qs({ limit: 100 }));
    if (!j.reports.length) { box.innerHTML = '<div class="p-6 text-body-sm text-slate-500">No reports yet. Generate one from the Reports view or ask the agent.</div>'; return; }
    box.innerHTML = `<table class="h-table"><thead><tr>
      <th>Stock</th><th>Type</th><th>Period</th><th>Created</th><th>Status</th><th>PDF</th><th>Email</th><th></th></tr></thead><tbody>` +
      j.reports.map((r) => `<tr>
        <td class="num font-bold">${r.stock}</td>
        <td>${esc(r.report_type)}</td>
        <td class="num">${r.start} → ${r.end}${r.as_of ? ' <span class="mut">(replay)</span>' : ''}</td>
        <td class="num mut">${esc(r.created_at || '')}</td>
        <td><span class="px-1.5 py-0.5 rounded text-[10px] font-semibold ${r.status === 'pdf_ready' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-slate-100 text-slate-600 border border-slate-200'}">${esc(r.status)}</span></td>
        <td>${r.pdf_filename ? `<a class="text-sky-700 underline" target="_blank" href="/api/reports/${r.id}/pdf">open</a>` : '<span class="mut">—</span>'}</td>
        <td>${r.emailed ? `<span class="pos">sent</span> <span class="mut">${esc(r.email_to || '')}</span>` : '<span class="mut">—</span>'}</td>
        <td><button data-open="${r.id}" class="text-slate-500 hover:text-slate-900 text-body-sm underline">view</button></td>
      </tr>`).join('') + `</tbody></table>`;
    $$('#h-table [data-open]').forEach((b) => b.addEventListener('click', async () => {
      try {
        const d = await api('/api/reports/' + b.dataset.open);
        state.currentReportId = d.record.id;
        setView('reports');
        $('#r-pdf').disabled = false; $('#r-email').disabled = false;
        renderReportPreview(d.report, d.record.id);
        $('#r-status').innerHTML = `loaded report <span class="num">${d.record.id}</span>`;
      } catch (e) { alert(e.message); }
    }));
  } catch (e) { box.innerHTML = `<div class="p-4 text-body-sm text-rose-600">${esc(e.message)}</div>`; }
}

/* ---------------- chat / agent ---------------- */
const QUICK = ['Analyze PGEO based on the latest data', 'What was the PGEO prediction vs actual?', 'Generate a weekly report for PGEO', 'How accurate is the PTBA model?'];
function renderChatQuick() {
  $('#chat-quick').innerHTML = QUICK.map((q) => `<button class="px-2.5 py-1 bg-white hover:bg-slate-100 border border-slate-200 rounded font-body-sm text-[11px] text-slate-700 hover:text-slate-900 transition-colors shadow-sm">${esc(q)}</button>`).join('');
  $$('#chat-quick button').forEach((b, i) => b.addEventListener('click', () => { $('#chat-input').value = QUICK[i]; sendChat(); }));
}
function wireChat() {
  $('#chat-send').addEventListener('click', sendChat);
  $('#chat-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') sendChat(); });
  $('#chat-clear').addEventListener('click', () => { state.chatHistory = []; $('#chat-log').innerHTML = ''; $('#chat-activity').innerHTML = ''; });
}
function pushMsg(role, html) {
  const div = document.createElement('div');
  div.className = role === 'user' ? 'msg-user' : 'msg-agent';
  div.innerHTML = html;
  $('#chat-log').appendChild(div);
  $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
  return div;
}
async function sendChat() {
  const input = $('#chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  pushMsg('user', esc(text));
  const act = $('#chat-activity');
  act.innerHTML = '<span class="spinner"></span> thinking…';
  const placeholder = pushMsg('agent', '<span class="spinner"></span> working…');
  try {
    const j = await api('/api/chat', { method: 'POST', body: JSON.stringify({ message: text, history: state.chatHistory.slice(-8), as_of: state.asOf }) });
    // activity feed
    act.innerHTML = (j.activities || []).length
      ? (j.activities || []).map((a) => `<span class="text-emerald-700 flex items-center gap-1"><span class="material-symbols-outlined text-xs">check</span>${esc(a)}</span>`).join('')
      : '<span class="mut">idle</span>';
    if (!j.configured) {
      placeholder.className = 'msg-error';
      placeholder.innerHTML = esc(j.reply);
    } else {
      const chips = (j.tool_trace || []).map((t) => `<span class="toolchip">${esc(t.tool)}</span>`).join('');
      placeholder.innerHTML = (chips ? `<div class="mb-1.5">${chips}</div>` : '') + '<div class="md-body">' + mdRender(j.reply) + '</div>';
    }
    state.chatHistory.push({ role: 'user', content: text });
    state.chatHistory.push({ role: 'assistant', content: j.reply });
    if (state.chatHistory.length > 16) state.chatHistory = state.chatHistory.slice(-16);
  } catch (e) {
    placeholder.className = 'msg-error';
    placeholder.innerHTML = 'Agent error: ' + esc(e.message);
    act.innerHTML = '<span class="neg">error</span>';
  }
}
