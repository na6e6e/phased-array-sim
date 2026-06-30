"use strict";

/* =========================================================================
 * フェーズドアレイ ビームフォーミング 視覚化
 *  - 2次元平面に波紋(干渉縞)を描画
 *  - 素子数 / 素子間隔 / 振幅テーパーを調整可能
 *  - 位相は「ビーム角から自動計算」または「素子ごとに手動」設定
 *  - 送信(Tx)と受信(Rx)の両モードに対応
 *
 * 座標系: 物理単位は波長 λ。アレイは画面下端中央 (y=0) に水平配置。
 *   素子 n の位置 off_n = (n - (N-1)/2) * d  [波長]
 *   ビーム角 θ は broadside(真上 +y)からの角度。+θ は右に傾く。
 *   ステアリング位相  φ_n = 2π · off_n · sin(θ0)
 * ========================================================================= */

// ---- 状態 -----------------------------------------------------------------
const state = {
  mode: "tx",            // 'tx' | 'rx'
  N: 8,
  d: 0.5,                // 素子間隔 (波長)
  taper: "uniform",
  ppw: 22,               // pixels per wavelength (ズーム)
  phaseMode: "beam",     // 'beam' | 'manual'
  beamAngle: 0,          // θ0 [deg]
  manualPhasesDeg: [],   // 手動位相 [deg]
  arrivalAngle: 20,      // θ_in [deg] (Rx)
  display: "wave",       // 'wave' | 'intensity'
  speed: 1.0,
  showDb: false,
  running: true,
};

// ---- DOM ------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const fieldCanvas = $("field");
const fieldCtx = fieldCanvas.getContext("2d");
const patternCanvas = $("pattern");
const patternCtx = patternCanvas.getContext("2d");
const phasorCanvas = $("phasor");
const phasorCtx = phasorCanvas.getContext("2d");

const cW = fieldCanvas.width;
const cH = fieldCanvas.height;

// 場の計算用ダウンサンプル係数(描画は拡大)
const DS = 2;
const bufW = Math.floor(cW / DS);
const bufH = Math.floor(cH / DS);

const offCanvas = document.createElement("canvas");
offCanvas.width = bufW;
offCanvas.height = bufH;
const offCtx = offCanvas.getContext("2d");
const imgData = offCtx.createImageData(bufW, bufH);

// 場の事前計算バッファ(時間に依存しない複素成分)
const bufC = new Float32Array(bufW * bufH);
const bufS = new Float32Array(bufW * bufH);
let maxEnv = 1;            // 振幅正規化用
let needFieldRecompute = true;

// 導出量(パラメータ変更時に再計算)
let offsets = [];          // off_n [波長]
let weights = [];          // 振幅テーパー a_n
let phasesRad = [];        // 位相 φ_n [rad]
let Gmax = 1;              // パターン正規化用

let wt = 0;                // 時間位相 ωt
let lastTime = 0;

// =========================================================================
// 導出量の更新
// =========================================================================
function taperWeights(N, kind) {
  const w = new Array(N);
  for (let n = 0; n < N; n++) {
    if (N === 1) { w[n] = 1; continue; }
    const x = n / (N - 1);              // 0..1
    switch (kind) {
      case "hamming":  w[n] = 0.54 - 0.46 * Math.cos(2 * Math.PI * x); break;
      case "hann":     w[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * x); break;
      case "blackman": w[n] = 0.42 - 0.5 * Math.cos(2 * Math.PI * x) + 0.08 * Math.cos(4 * Math.PI * x); break;
      default:         w[n] = 1;
    }
  }
  return w;
}

function refreshDerived() {
  const N = state.N;
  const mid = (N - 1) / 2;
  offsets = [];
  for (let n = 0; n < N; n++) offsets.push((n - mid) * state.d);

  weights = taperWeights(N, state.taper);

  // 位相
  phasesRad = new Array(N);
  if (state.phaseMode === "beam") {
    const s = Math.sin(state.beamAngle * Math.PI / 180);
    for (let n = 0; n < N; n++) phasesRad[n] = 2 * Math.PI * offsets[n] * s;
  } else {
    // 手動: 配列長を N に合わせる
    if (state.manualPhasesDeg.length !== N) {
      const old = state.manualPhasesDeg;
      state.manualPhasesDeg = new Array(N).fill(0).map((_, i) => old[i] ?? 0);
    }
    for (let n = 0; n < N; n++) phasesRad[n] = state.manualPhasesDeg[n] * Math.PI / 180;
  }

  Gmax = weights.reduce((a, b) => a + b, 0) || 1;
  needFieldRecompute = true;
}

// =========================================================================
// 波動場の事前計算 (時間非依存の C, S 成分)
//   瞬時場 E(t) = C·cos(ωt) + S·sin(ωt)
// =========================================================================
function recomputeField() {
  const TWO_PI = 2 * Math.PI;
  let mx = 1e-9;

  if (state.mode === "tx") {
    const N = state.N;
    for (let j = 0; j < bufH; j++) {
      const py = (cH - j * DS) / state.ppw;          // 物理 y (波長), 下端=0
      for (let i = 0; i < bufW; i++) {
        const px = (i * DS - cW / 2) / state.ppw;     // 物理 x (波長)
        let c = 0, s = 0;
        for (let n = 0; n < N; n++) {
          const dx = px - offsets[n];
          const r = Math.sqrt(dx * dx + py * py);     // 距離 (波長)
          const att = weights[n] / Math.sqrt(1 + r);  // 幾何減衰
          const ph = TWO_PI * r + phasesRad[n];
          c += att * Math.cos(ph);
          s += att * Math.sin(ph);
        }
        const idx = j * bufW + i;
        bufC[idx] = c;
        bufS[idx] = s;
        const env = c * c + s * s;
        if (env > mx) mx = env;
      }
    }
    maxEnv = Math.sqrt(mx);
  } else {
    // Rx: 到来平面波。配列に向かって(下向きに)進む。
    const sA = Math.sin(state.arrivalAngle * Math.PI / 180);
    const cA = Math.cos(state.arrivalAngle * Math.PI / 180);
    for (let j = 0; j < bufH; j++) {
      const py = (cH - j * DS) / state.ppw;
      for (let i = 0; i < bufW; i++) {
        const px = (i * DS - cW / 2) / state.ppw;
        const phi = TWO_PI * (px * sA + py * cA);
        const idx = j * bufW + i;
        bufC[idx] = Math.cos(phi);
        bufS[idx] = -Math.sin(phi);   // 符号反転で配列方向へ伝搬
      }
    }
    maxEnv = 1;
  }
  needFieldRecompute = false;
}

// =========================================================================
// カラーマップ
// =========================================================================
const DIVERGING = [
  [0.0, [38, 96, 190]],
  [0.5, [10, 12, 20]],
  [1.0, [222, 78, 52]],
];
const SEQ = [
  [0.0, [6, 10, 30]],
  [0.25, [26, 72, 132]],
  [0.5, [18, 150, 162]],
  [0.72, [236, 206, 72]],
  [1.0, [255, 250, 232]],
];

function sampleMap(stops, t) {
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  for (let k = 1; k < stops.length; k++) {
    if (t <= stops[k][0]) {
      const [p0, c0] = stops[k - 1];
      const [p1, c1] = stops[k];
      const f = (t - p0) / (p1 - p0 || 1);
      return [
        c0[0] + (c1[0] - c0[0]) * f,
        c0[1] + (c1[1] - c0[1]) * f,
        c0[2] + (c1[2] - c0[2]) * f,
      ];
    }
  }
  return stops[stops.length - 1][1];
}

// =========================================================================
// 波動場の描画
// =========================================================================
function renderField() {
  const data = imgData.data;
  const cw = Math.cos(wt);
  const sw = Math.sin(wt);
  const scale = (maxEnv * 0.92) || 1;

  if (state.display === "intensity" && state.mode === "tx") {
    // 強度(時間平均 ∝ envelope^2)
    for (let p = 0; p < bufC.length; p++) {
      const env = Math.sqrt(bufC[p] * bufC[p] + bufS[p] * bufS[p]);
      const u = Math.pow(Math.min(env / scale, 1), 0.6);
      const rgb = sampleMap(SEQ, u);
      const o = p * 4;
      data[o] = rgb[0]; data[o + 1] = rgb[1]; data[o + 2] = rgb[2]; data[o + 3] = 255;
    }
  } else {
    // 瞬時の波紋
    for (let p = 0; p < bufC.length; p++) {
      const e = bufC[p] * cw + bufS[p] * sw;
      const t = (Math.max(-1, Math.min(1, e / scale)) + 1) / 2;
      const rgb = sampleMap(DIVERGING, t);
      const o = p * 4;
      data[o] = rgb[0]; data[o + 1] = rgb[1]; data[o + 2] = rgb[2]; data[o + 3] = 255;
    }
  }

  offCtx.putImageData(imgData, 0, 0);
  fieldCtx.imageSmoothingEnabled = true;
  fieldCtx.drawImage(offCanvas, 0, 0, cW, cH);

  drawFieldOverlay();
}

function drawFieldOverlay() {
  const ctx = fieldCtx;
  const cx = cW / 2;
  const ay = cH - 3;               // アレイの画面 y (下端)

  // broadside 基準線(破線)
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.setLineDash([5, 6]);
  ctx.beginPath();
  ctx.moveTo(cx, ay);
  ctx.lineTo(cx, 20);
  ctx.stroke();
  ctx.setLineDash([]);

  // 方向矢印
  if (state.mode === "tx") {
    drawArrow(ctx, cx, ay, state.beamAngle, 150, "#ff7a59", "ビーム θ₀");
  } else {
    // 到来方向(上空から配列へ)
    const th = state.arrivalAngle * Math.PI / 180;
    const len = 150;
    const sx = cx + Math.sin(th) * len;
    const sy = ay - Math.cos(th) * len;
    drawArrowSeg(ctx, sx, sy, cx, ay, "#4ea1ff");
    label(ctx, sx, sy - 6, "到来 θ_in", "#4ea1ff");
    // 受信ステアリング方向
    drawArrow(ctx, cx, ay, state.beamAngle, 110, "rgba(255,122,89,0.85)", "ステア θ₀");
  }

  // 素子マーカー
  for (let n = 0; n < state.N; n++) {
    const ex = cx + offsets[n] * state.ppw;
    if (ex < 4 || ex > cW - 4) continue;
    let glow = "#5cf2c0";
    if (state.mode === "rx") {
      // 到来波の瞬時値で色付け
      const sA = Math.sin(state.arrivalAngle * Math.PI / 180);
      const e = Math.cos(2 * Math.PI * offsets[n] * sA) * Math.cos(wt)
        - Math.sin(2 * Math.PI * offsets[n] * sA) * Math.sin(wt);
      const t = (e + 1) / 2;
      const rgb = sampleMap(DIVERGING, t);
      glow = `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`;
    }
    ctx.beginPath();
    ctx.arc(ex, ay, 5.5, 0, 2 * Math.PI);
    ctx.fillStyle = glow;
    ctx.shadowColor = glow;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
  ctx.restore();
}

function drawArrow(ctx, x0, y0, angleDeg, len, color, text) {
  const th = angleDeg * Math.PI / 180;
  const x1 = x0 + Math.sin(th) * len;
  const y1 = y0 - Math.cos(th) * len;
  drawArrowSeg(ctx, x0, y0, x1, y1, color);
  if (text) label(ctx, x1, y1 - 4, text, color);
}

function drawArrowSeg(ctx, x0, y0, x1, y1, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  const ang = Math.atan2(y1 - y0, x1 - x0);
  const h = 9;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - h * Math.cos(ang - 0.4), y1 - h * Math.sin(ang - 0.4));
  ctx.lineTo(x1 - h * Math.cos(ang + 0.4), y1 - h * Math.sin(ang + 0.4));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function label(ctx, x, y, text, color) {
  ctx.save();
  ctx.font = "11px Segoe UI";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.fillText(text, x, y);
  ctx.restore();
}

// =========================================================================
// ビームパターン(アレイファクタ)
//   G(θ) = Σ a_n · exp( j (φ_n − 2π·off_n·sinθ) )
// =========================================================================
function patternMag(thetaDeg) {
  const s = Math.sin(thetaDeg * Math.PI / 180);
  let re = 0, im = 0;
  for (let n = 0; n < state.N; n++) {
    const ph = phasesRad[n] - 2 * Math.PI * offsets[n] * s;
    re += weights[n] * Math.cos(ph);
    im += weights[n] * Math.sin(ph);
  }
  return Math.hypot(re, im);
}

function drawBeamPattern() {
  const ctx = patternCtx;
  const W = patternCanvas.width, H = patternCanvas.height;
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H - 18, R = Math.min(W / 2 - 16, H - 30);

  // グリッド(半円)
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.font = "10px Segoe UI";
  ctx.lineWidth = 1;
  const rings = state.showDb ? [0.25, 0.5, 0.75, 1] : [0.25, 0.5, 0.75, 1];
  for (const rr of rings) {
    ctx.beginPath();
    ctx.arc(cx, cy, R * rr, Math.PI, 2 * Math.PI);
    ctx.stroke();
  }
  // 角度目盛
  for (let a = -90; a <= 90; a += 30) {
    const th = a * Math.PI / 180;
    const x = cx + Math.sin(th) * R;
    const y = cy - Math.cos(th) * R;
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.fillText(a + "°", x + Math.sin(th) * 10 - 8, y - Math.cos(th) * 10 + 3);
  }

  // 曲線
  const radiusOf = (mag) => {
    const norm = mag / Gmax;
    if (state.showDb) {
      const db = 20 * Math.log10(Math.max(norm, 1e-4));
      return Math.max(0, (db + 40) / 40) * R;
    }
    return norm * R;
  };

  ctx.beginPath();
  for (let a = -90; a <= 90; a += 0.5) {
    const r = radiusOf(patternMag(a));
    const th = a * Math.PI / 180;
    const x = cx + Math.sin(th) * r;
    const y = cy - Math.cos(th) * r;
    if (a === -90) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#4ea1ff";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.lineTo(cx, cy);
  ctx.closePath();
  ctx.fillStyle = "rgba(78,161,255,0.16)";
  ctx.fill();

  // ステアリング方向マーカー
  markAngle(ctx, cx, cy, R, state.beamAngle, "#ff7a59");
  if (state.mode === "rx") markAngle(ctx, cx, cy, R, state.arrivalAngle, "#36d399");
}

function markAngle(ctx, cx, cy, R, deg, color) {
  const th = deg * Math.PI / 180;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.sin(th) * R, cy - Math.cos(th) * R);
  ctx.stroke();
  ctx.restore();
}

// =========================================================================
// フェーザ合成図 (Rx): 各素子の重み付き出力を順に足し合わせる
//   c_n = a_n · exp( j (2π·off_n·sinθ_in − φ_n) )   ← ステアリングで位相補正
// =========================================================================
function drawPhasor() {
  const ctx = phasorCtx;
  const W = phasorCanvas.width, H = phasorCanvas.height;
  ctx.clearRect(0, 0, W, H);

  const sA = Math.sin(state.arrivalAngle * Math.PI / 180);
  // 各素子の複素出力(見やすさのため ωt でゆっくり回転)
  const comps = [];
  let sumRe = 0, sumIm = 0, sumW = 0;
  for (let n = 0; n < state.N; n++) {
    const ph = 2 * Math.PI * offsets[n] * sA - phasesRad[n] + wt * 0.3;
    const re = weights[n] * Math.cos(ph);
    const im = weights[n] * Math.sin(ph);
    comps.push([re, im]);
    sumRe += re; sumIm += im; sumW += weights[n];
  }
  const outMag = Math.hypot(sumRe, sumIm);

  // スケール: 全長が画面に収まるように
  const cx = W * 0.5, cy = H * 0.5;
  const scale = (Math.min(W, H) * 0.42) / (sumW || 1);

  // 個々のフェーザ(tip-to-tail)
  let x = cx, y = cy;
  ctx.lineWidth = 2;
  for (let n = 0; n < comps.length; n++) {
    const nx = x + comps[n][0] * scale;
    const ny = y - comps[n][1] * scale;
    ctx.strokeStyle = "rgba(140,180,255,0.65)";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(nx, ny);
    ctx.stroke();
    x = nx; y = ny;
  }
  // 合成ベクトル
  drawArrowSeg(ctx, cx, cy, cx + sumRe * scale, cy - sumIm * scale, "#ff7a59");

  // 最大長(全素子同相)の参照円
  ctx.strokeStyle = "rgba(54,211,153,0.35)";
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.arc(cx, cy, sumW * scale, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.setLineDash([]);

  const ratio = outMag / (sumW || 1);
  const db = 20 * Math.log10(Math.max(ratio, 1e-4));
  $("phasor-readout").innerHTML =
    `受信出力 |Σ| = <span class="k">${outMag.toFixed(2)}</span> / 最大 ${sumW.toFixed(2)} ` +
    `= <span class="${ratio > 0.98 ? "ok" : "hot"}">${(ratio * 100).toFixed(1)}%</span> ` +
    `(<span class="k">${db.toFixed(1)} dB</span>)`;
}

// =========================================================================
// 読み出し情報
// =========================================================================
function updateReadouts() {
  // グレーティングローブの注意
  let warn = "";
  if (state.d > 0.5) {
    const maxScan = Math.asin(Math.min(1, 1 / state.d - 1)) * 180 / Math.PI;
    if (state.d >= 1) warn = ` <span class="hot">⚠ d≥λ: グレーティングローブ発生</span>`;
    else warn = ` <span class="hot">⚠ d&gt;0.5λ: |θ₀|&gt;${maxScan.toFixed(0)}° でグレーティングローブ</span>`;
  }
  $("field-readout").innerHTML =
    `素子数 <span class="k">${state.N}</span>, 間隔 <span class="k">${state.d.toFixed(2)}λ</span>, ` +
    `アレイ全長 <span class="k">${((state.N - 1) * state.d).toFixed(2)}λ</span>${warn}`;

  // パターン: ピーク方向と半値幅(-3dB)
  let peakA = 0, peakV = -1;
  for (let a = -90; a <= 90; a += 0.25) {
    const v = patternMag(a);
    if (v > peakV) { peakV = v; peakA = a; }
  }
  const half = peakV / Math.SQRT2;
  let lo = peakA, hi = peakA;
  for (let a = peakA; a >= -90; a -= 0.25) { if (patternMag(a) < half) { lo = a; break; } }
  for (let a = peakA; a <= 90; a += 0.25) { if (patternMag(a) > half) hi = a; else break; }
  const bw = hi - lo;
  $("pattern-readout").innerHTML =
    `ピーク方向 <span class="hot">${peakA.toFixed(0)}°</span>, ` +
    `半値幅(HPBW) <span class="k">${bw > 0 ? bw.toFixed(1) + "°" : "—"}</span>`;
}

// =========================================================================
// メインループ
// =========================================================================
function loop(ts) {
  const dt = lastTime ? (ts - lastTime) / 1000 : 0.016;
  lastTime = ts;
  if (state.running) wt += 2 * Math.PI * 0.5 * state.speed * dt * 6;

  if (needFieldRecompute) recomputeField();
  renderField();
  drawBeamPattern();
  if (state.mode === "rx") drawPhasor();

  requestAnimationFrame(loop);
}

// =========================================================================
// 手動位相スライダーの生成
// =========================================================================
function buildManualPhases() {
  const box = $("manual-phases");
  box.innerHTML = "";
  if (state.manualPhasesDeg.length !== state.N) {
    state.manualPhasesDeg = new Array(state.N).fill(0);
  }
  for (let n = 0; n < state.N; n++) {
    const item = document.createElement("label");
    item.className = "mp-item";
    item.innerHTML =
      `<span>#${n + 1}: <b>${state.manualPhasesDeg[n]}°</b></span>` +
      `<input type="range" min="-180" max="180" step="5" value="${state.manualPhasesDeg[n]}" data-n="${n}">`;
    const input = item.querySelector("input");
    input.addEventListener("input", (e) => {
      const idx = +e.target.dataset.n;
      state.manualPhasesDeg[idx] = +e.target.value;
      item.querySelector("b").textContent = e.target.value + "°";
      refreshDerived();
    });
    box.appendChild(item);
  }
}

// =========================================================================
// UI 設定
// =========================================================================
function setMode(mode) {
  state.mode = mode;
  $("tab-tx").classList.toggle("active", mode === "tx");
  $("tab-rx").classList.toggle("active", mode === "rx");
  $("rx-panel").classList.toggle("hidden", mode !== "rx");
  $("phasor-card").classList.toggle("hidden", mode !== "rx");
  $("field-title").textContent = mode === "tx" ? "波動場 (送信)" : "到来波と素子受信 (受信)";
  // Rx では強度表示の代わりに波紋固定
  $("disp-mode").style.opacity = mode === "tx" ? "1" : "0.4";
  $("disp-mode").style.pointerEvents = mode === "tx" ? "auto" : "none";
  needFieldRecompute = true;
}

function setPhaseMode(pm) {
  state.phaseMode = pm;
  document.querySelectorAll("#phase-mode .seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.pm === pm));
  $("beam-ctrl").classList.toggle("hidden", pm !== "beam");
  $("manual-phases").classList.toggle("hidden", pm !== "manual");
  if (pm === "manual") buildManualPhases();
  refreshDerived();
}

function setDisplay(disp) {
  state.display = disp;
  document.querySelectorAll("#disp-mode .seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.disp === disp));
}

function bindEvents() {
  $("tab-tx").addEventListener("click", () => setMode("tx"));
  $("tab-rx").addEventListener("click", () => setMode("rx"));

  $("in-N").addEventListener("input", (e) => {
    state.N = +e.target.value;
    $("val-N").textContent = state.N;
    if (state.phaseMode === "manual") buildManualPhases();
    refreshDerived();
  });
  $("in-d").addEventListener("input", (e) => {
    state.d = +e.target.value;
    $("val-d").textContent = state.d.toFixed(2);
    refreshDerived();
  });
  $("in-taper").addEventListener("change", (e) => {
    state.taper = e.target.value;
    refreshDerived();
  });
  $("in-zoom").addEventListener("input", (e) => {
    state.ppw = +e.target.value;
    $("val-zoom").textContent = state.ppw;
    needFieldRecompute = true;
  });
  $("in-beam").addEventListener("input", (e) => {
    state.beamAngle = +e.target.value;
    $("val-beam").textContent = state.beamAngle + "°";
    refreshDerived();
  });
  $("in-arr").addEventListener("input", (e) => {
    state.arrivalAngle = +e.target.value;
    $("val-arr").textContent = state.arrivalAngle + "°";
    needFieldRecompute = true;
  });
  $("in-speed").addEventListener("input", (e) => {
    state.speed = +e.target.value;
    $("val-speed").textContent = state.speed.toFixed(1) + "×";
  });
  $("in-db").addEventListener("change", (e) => { state.showDb = e.target.checked; });

  document.querySelectorAll("#phase-mode .seg-btn").forEach((b) =>
    b.addEventListener("click", () => setPhaseMode(b.dataset.pm)));
  document.querySelectorAll("#disp-mode .seg-btn").forEach((b) =>
    b.addEventListener("click", () => setDisplay(b.dataset.disp)));

  $("btn-play").addEventListener("click", () => {
    state.running = !state.running;
    $("btn-play").textContent = state.running ? "⏸ 一時停止" : "▶ 再生";
  });
  $("btn-reset").addEventListener("click", resetAll);

  // 読み出しはパラメータ変更時に更新したいので入力イベントにフック
  document.addEventListener("input", () => updateReadouts());
}

function resetAll() {
  Object.assign(state, {
    mode: "tx", N: 8, d: 0.5, taper: "uniform", ppw: 22,
    phaseMode: "beam", beamAngle: 0, manualPhasesDeg: [],
    arrivalAngle: 20, display: "wave", speed: 1.0, showDb: false, running: true,
  });
  $("in-N").value = 8; $("val-N").textContent = "8";
  $("in-d").value = 0.5; $("val-d").textContent = "0.50";
  $("in-taper").value = "uniform";
  $("in-zoom").value = 22; $("val-zoom").textContent = "22";
  $("in-beam").value = 0; $("val-beam").textContent = "0°";
  $("in-arr").value = 20; $("val-arr").textContent = "20°";
  $("in-speed").value = 1; $("val-speed").textContent = "1.0×";
  $("in-db").checked = false;
  $("btn-play").textContent = "⏸ 一時停止";
  setMode("tx");
  setPhaseMode("beam");
  setDisplay("wave");
  refreshDerived();
  updateReadouts();
}

// 凡例バー
function buildLegend() {
  const bar = document.createElement("div");
  bar.className = "bar";
  const grad = [];
  for (let i = 0; i <= 10; i++) {
    const rgb = sampleMap(DIVERGING, i / 10);
    grad.push(`rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0}) ${i * 10}%`);
  }
  bar.style.background = `linear-gradient(90deg, ${grad.join(",")})`;
  const lg = $("field-legend");
  lg.innerHTML = "−";
  lg.appendChild(bar);
  const plus = document.createElement("span");
  plus.textContent = "＋ (波の振幅)";
  lg.appendChild(plus);
}

// =========================================================================
// 起動
// =========================================================================
function init() {
  bindEvents();
  buildLegend();
  setMode("tx");
  setPhaseMode("beam");
  refreshDerived();
  updateReadouts();
  requestAnimationFrame(loop);
}

init();
