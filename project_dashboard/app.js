/* ═══════════════════════════════════════════════
   FEM-JAX-GEO Dashboard — Application Logic
   ═══════════════════════════════════════════════ */

// ── Colors ──
const C = {
  blue:    '#3d5a99',
  red:     '#b54444',
  green:   '#3d7a52',
  amber:   '#b58a2a',
  teal:    '#3d7a8a',
  text2:   '#7a726b',
  border:  '#e5e1dc',
  bgFill:  (c, a) => {
    const r = parseInt(c.slice(1,3),16), g = parseInt(c.slice(3,5),16), b = parseInt(c.slice(5,7),16);
    return `rgba(${r},${g},${b},${a})`;
  }
};

// ── Navigation ──
function showSection(id) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  const sec = document.getElementById('sec-' + id);
  if (sec) sec.classList.add('active');
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  if (event && event.target) event.target.classList.add('active');
  const nav = document.querySelector('nav');
  if (nav) window.scrollTo({ top: nav.offsetTop, behavior: 'smooth' });
}

// ── Collapsibles ──
function toggleBug(el) { el.classList.toggle('open'); }
function toggleDetail(el) {
  const parent = el.closest('.collapsible');
  if (parent) parent.classList.toggle('open');
}

// ── Chart.js Defaults ──
Chart.defaults.color = '#7a726b';
Chart.defaults.borderColor = '#e5e1dc';
Chart.defaults.font.family = "'Instrument Sans', -apple-system, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.legend.labels.boxHeight = 8;

const logYAxis = (title) => ({
  type: 'logarithmic', title: { display: true, text: title, font: { weight: 500 } },
  grid: { color: '#eeeae6' }
});
const linAxis = (title) => ({
  title: { display: true, text: title, font: { weight: 500 } },
  grid: { color: '#eeeae6' }
});
const chartOpts = (extra = {}) => ({
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { position: 'top' }, ...extra.plugins },
  scales: extra.scales || {}
});

// ═══════════════════════════════════════
// OVERVIEW CHARTS
// ═══════════════════════════════════════

new Chart(document.getElementById('chartOverviewScaling'), {
  type: 'line',
  data: {
    labels: ['25','100','400','900','2500(est)'],
    datasets: [
      { label: 'AD 梯度时间 (s)', data: [0.175,0.179,0.170,0.202,0.312],
        borderColor: C.blue, backgroundColor: C.bgFill(C.blue,0.08), fill: true, tension: 0.3, pointRadius: 5 },
      { label: 'FD 梯度时间 (s)', data: [4.96,19.9,75.7,221.5,600],
        borderColor: C.red, backgroundColor: C.bgFill(C.red,0.08), fill: true, tension: 0.3, pointRadius: 5 }
    ]
  },
  options: chartOpts({ scales: { y: logYAxis('时间 (秒)'), x: linAxis('参数维度 N') } })
});

new Chart(document.getElementById('chartOverviewAccuracy'), {
  type: 'bar',
  data: {
    labels: ['D1 孪生','H5 双区域','J1 四层','J2 夹层','J3 随机场','MC c(x)'],
    datasets: [{
      label: 'L2 误差 %', data: [6.1e-11, 3.95, 18.15, 23.9, 2.7, 7.09],
      backgroundColor: [C.bgFill(C.green,0.6), C.bgFill(C.green,0.6), C.bgFill(C.red,0.6),
                        C.bgFill(C.amber,0.6), C.bgFill(C.green,0.6), C.bgFill(C.green,0.6)],
      borderColor: [C.green, C.green, C.red, C.amber, C.green, C.green], borderWidth: 1
    }]
  },
  options: chartOpts({ scales: { y: logYAxis('L2 相对误差 %') } })
});

// ═══════════════════════════════════════
// TAYLOR TEST
// ═══════════════════════════════════════

let taylorChart;
function initTaylor() {
  taylorChart = new Chart(document.getElementById('chartTaylor'), {
    type: 'line',
    data: {
      labels: ['1e-1','1e-2','1e-3','1e-4','1e-5','1e-6'],
      datasets: [
        { label: '修复前 (斜率≈1)', data: [2.78e-4,2.78e-5,2.78e-6,2.78e-7,2.78e-8,2.78e-9],
          borderColor: C.red, pointRadius: 5, pointBackgroundColor: C.red, borderDash: [6,3] },
        { label: '修复后 (斜率≈2)', data: [3.1e-3,3.1e-5,3.1e-7,3.1e-9,3.1e-11,3.1e-13],
          borderColor: C.green, pointRadius: 5, pointBackgroundColor: C.green },
        { label: '理想 O(ε²)', data: [1e-2,1e-4,1e-6,1e-8,1e-10,1e-12],
          borderColor: C.blue, borderDash: [2,4], pointRadius: 0, borderWidth: 1 }
      ]
    },
    options: chartOpts({ scales: { y: logYAxis('Taylor 残差 r₁'), x: linAxis('扰动步长 ε') } })
  });
}
function updateTaylor(mode, btn) {
  btn.parentElement.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const ds = taylorChart.data.datasets;
  ds[0].hidden = mode === 'after';
  ds[1].hidden = mode === 'before';
  ds[2].hidden = false;
  taylorChart.update();
}
initTaylor();

// ═══════════════════════════════════════
// DISPLACEMENT SWEEP
// ═══════════════════════════════════════

new Chart(document.getElementById('chartDispSweep'), {
  type: 'line',
  data: {
    labels: ['-0.020','-0.022','-0.024','-0.025','-0.026','-0.027','-0.028','-0.030','-0.031'],
    datasets: [{
      label: 'AD/FD ∂L/∂E 相对误差', borderColor: C.red,
      data: [5.7e-14,1.2e-14,8.3e-15,1.7e-16,0.990,0.993,0.995,0.995,0.996],
      backgroundColor: C.bgFill(C.red,0.08), fill: true, pointRadius: 5, tension: 0.1,
      pointBackgroundColor: ctx => ctx.raw > 0.5 ? C.red : C.green
    }]
  },
  options: chartOpts({ scales: { y: { ...logYAxis('相对误差'), min: 1e-16, max: 10 }, x: linAxis('位移 (mm)') } })
});

// ═══════════════════════════════════════
// ROBUSTNESS 15-LEVEL
// ═══════════════════════════════════════

new Chart(document.getElementById('chartRobustness'), {
  type: 'bar',
  data: {
    labels: ['-5','-10','-15','-18','-20','-22','-24','-25','-26','-27','-28','-29','-30','-30.5','-31'].map(x => x+'μm'),
    datasets: [
      { label: '∂L/∂E 误差', data: [1e-14,1e-14,1e-14,1e-14,5.7e-14,1.2e-14,8.3e-15,1.7e-16,1.65e-5,1.2e-5,9.1e-6,5.5e-6,4.3e-6,3.8e-6,3.5e-6],
        backgroundColor: C.bgFill(C.blue,0.5) },
      { label: '∂L/∂k 误差', data: [0,0,0,0,0,0,0,0,9.67e-6,8.2e-6,7.1e-6,4.8e-6,3.2e-6,2.9e-6,2.6e-6],
        backgroundColor: C.bgFill(C.teal,0.5) }
    ]
  },
  options: chartOpts({ scales: { y: { ...logYAxis('AD/FD 相对误差'), min: 1e-16 }, x: linAxis('位移档位') } })
});

// ═══════════════════════════════════════
// D3 STRATEGY
// ═══════════════════════════════════════

new Chart(document.getElementById('chartD3'), {
  type: 'bar',
  data: {
    labels: ['单步观测','多步加载','两阶段+多步'],
    datasets: [
      { label: 'E 相对误差 %', data: [14.26,24.02,2.08e-14], backgroundColor: C.bgFill(C.blue,0.6) },
      { label: 'k 相对误差 %', data: [60,10,0.014], backgroundColor: C.bgFill(C.teal,0.6) }
    ]
  },
  options: chartOpts({ scales: { y: { ...logYAxis('相对误差 %'), min: 1e-14 } } })
});

// ═══════════════════════════════════════
// D2 HEATMAP
// ═══════════════════════════════════════

(function(){
  const container = document.getElementById('heatmapD2');
  if (!container) return;
  const N = 15;
  container.style.gridTemplateColumns = `repeat(${N},1fr)`;
  container.classList.add('heatmap');
  const Erange = [40000,100000], krange = [20,80];
  for (let j = N-1; j >= 0; j--) {
    for (let i = 0; i < N; i++) {
      const E = Erange[0] + (Erange[1]-Erange[0])*i/(N-1);
      const k_ = krange[0] + (krange[1]-krange[0])*j/(N-1);
      const dE = (E-70000)/70000, dk = (k_-50)/50;
      const loss = 8e-6*dE*dE*70000*70000 + 25.7*dk*dk*50*50;
      const norm = Math.min(loss/15000, 1);
      const r = Math.round(245 - norm*180), g = Math.round(240 - norm*140), b = Math.round(235 - norm*160);
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      cell.style.background = `rgb(${r},${g},${b})`;
      const tip = document.createElement('div');
      tip.className = 'heatmap-tooltip';
      tip.textContent = `E=${Math.round(E)} k=${k_.toFixed(0)} L=${loss.toFixed(1)}`;
      cell.appendChild(tip);
      container.appendChild(cell);
    }
  }
})();

// ═══════════════════════════════════════
// NOISE ROBUSTNESS (E-F section)
// ═══════════════════════════════════════

const noiseData = [
  { level:'0%',   errE:'2.08×10⁻¹⁶', errK:'1.70×10⁻⁴', conv:'5/5', eE:2.08e-16, eK:1.7e-4, status:'good' },
  { level:'0.1%', errE:'6.85×10⁻⁴',  errK:'6.40×10⁻⁴', conv:'5/5', eE:6.85e-4,  eK:6.4e-4, status:'good' },
  { level:'0.5%', errE:'4.35×10⁻³',  errK:'4.60×10⁻³', conv:'5/5', eE:4.35e-3,  eK:4.6e-3, status:'good' },
  { level:'1%',   errE:'7.66×10⁻³',  errK:'4.86×10⁻³', conv:'5/5', eE:7.66e-3,  eK:4.86e-3,status:'good' },
  { level:'2%',   errE:'1.38×10⁻²',  errK:'2.19×10⁻²', conv:'4/5', eE:1.38e-2,  eK:2.19e-2,status:'warn' },
  { level:'5%',   errE:'3.42×10⁻²',  errK:'2.23×10⁻¹', conv:'3/5', eE:3.42e-2,  eK:2.23e-1,status:'bad' },
  { level:'10%',  errE:'7.58×10⁻²',  errK:'5.99×10⁻¹', conv:'2/5', eE:7.58e-2,  eK:5.99e-1,status:'bad' }
];

const noiseChartEl = document.getElementById('chartNoise');
const noiseChart = noiseChartEl ? new Chart(noiseChartEl, {
  type: 'line',
  data: {
    labels: noiseData.map(d => d.level),
    datasets: [
      { label: 'E 平均误差', data: noiseData.map(d => d.eE*100), borderColor: C.blue, pointRadius: 4, tension: 0.3 },
      { label: 'k 平均误差', data: noiseData.map(d => d.eK*100), borderColor: C.teal, pointRadius: 4, tension: 0.3 },
      { label: '1% 阈值', data: Array(7).fill(1), borderColor: C.green, borderDash: [4,4], pointRadius: 0, borderWidth: 1 }
    ]
  },
  options: chartOpts({ scales: { y: { ...logYAxis('相对误差 %'), min: 1e-12 }, x: linAxis('噪声水平') } })
}) : null;

function updateNoise() {
  const i = parseInt(document.getElementById('noiseSlider').value);
  const d = noiseData[i];
  document.getElementById('noiseVal').textContent = d.level;
  const eE = document.getElementById('noiseE'), eK = document.getElementById('noiseK');
  eE.textContent = d.errE;
  eK.textContent = d.errK;
  const colorMap = { good: 'var(--success)', warn: 'var(--warning)', bad: 'var(--danger)' };
  eE.style.color = colorMap[d.status];
  eK.style.color = colorMap[d.status];
  const st = document.getElementById('noiseStatus');
  const icons = { good: '✓', warn: '⚠', bad: '✗' };
  st.textContent = `${icons[d.status]} ${d.conv} 试验收敛`;
  st.className = 'phase-pill ' + (d.status === 'good' ? 'pass' : d.status === 'warn' ? 'warn' : 'fail');
}

// ═══════════════════════════════════════
// V-CURVE (F1)
// ═══════════════════════════════════════

new Chart(document.getElementById('chartVcurve'), {
  type: 'line',
  data: {
    labels: ['0.1','0.03','0.01','3e-3','1e-3','3e-4','1e-4','3e-5','1e-5','1e-6','1e-7','1e-8'],
    datasets: [
      { label: '弹性 FD 误差', data: [4.3e-16,3e-15,9.8e-15,9.8e-14,3.8e-14,6.4e-13,6.4e-13,1.4e-11,7.5e-11,2.1e-11,3.2e-9,4.9e-8],
        borderColor: C.blue, pointRadius: 3, tension: 0.2 },
      { label: '塑性 FD 误差', data: [13.58,8.2e-3,9.1e-4,8.2e-5,9.3e-6,1e-6,2.8e-7,2e-7,1.9e-7,1.6e-7,7.2e-8,1.8e-6],
        borderColor: C.red, pointRadius: 3, tension: 0.2 },
      { label: 'AD 精度', data: Array(12).fill(1e-7), borderColor: C.green, borderDash: [4,4], pointRadius: 0, borderWidth: 2 }
    ]
  },
  options: chartOpts({ scales: { y: logYAxis('相对误差'), x: linAxis('FD 步长 (相对)') } })
});

// ═══════════════════════════════════════
// F4 SCALING (low-dim)
// ═══════════════════════════════════════

new Chart(document.getElementById('chartF4'), {
  type: 'bar',
  data: {
    labels: ['1 (E)','2 (E,k)','3 (E,k,ν)','4 (E,k,ν,α)'],
    datasets: [
      { label: 'AD 时间 (ms)', data: [82,89,82,86], backgroundColor: C.bgFill(C.blue,0.6) },
      { label: 'FD 时间 (ms)', data: [67,137,204,275], backgroundColor: C.bgFill(C.red,0.5) }
    ]
  },
  options: chartOpts({ scales: { y: linAxis('梯度计算时间 (ms)'), x: linAxis('参数数量') } })
});

// ═══════════════════════════════════════
// J4 SCALING (high-dim)
// ═══════════════════════════════════════

new Chart(document.getElementById('chartJ4'), {
  type: 'line',
  data: {
    labels: ['25','100','400','900'],
    datasets: [
      { label: '实际加速比', data: [28,111,445,1099], borderColor: C.blue,
        backgroundColor: C.bgFill(C.blue,0.08), fill: true, pointRadius: 6, tension: 0.3, borderWidth: 2 },
      { label: '理论 N', data: [25,100,400,900], borderColor: C.text2, borderDash: [4,4], pointRadius: 0, borderWidth: 1 }
    ]
  },
  options: chartOpts({ scales: { y: { ...logYAxis('AD 加速比'), }, x: linAxis('参数维度 N') } })
});

// ═══════════════════════════════════════
// REGULARIZATION (I3)
// ═══════════════════════════════════════

const regData = {
  0: [{t:'无',l2:0.96},{t:'TV 0.01',l2:3.97},{t:'TV 0.1',l2:9.56},{t:'TV 1',l2:27.7},{t:'Lap 0.01',l2:6.58},{t:'Lap 0.1',l2:8.34},{t:'Lap 1',l2:10.9}],
  1: [{t:'无',l2:57.1},{t:'TV 0.01',l2:3.54},{t:'TV 0.1',l2:9.23},{t:'TV 1',l2:27.8},{t:'Lap 0.01',l2:7.55},{t:'Lap 0.1',l2:8.33},{t:'Lap 1',l2:10.8}],
  3: [{t:'无',l2:103},{t:'TV 0.01',l2:12.9},{t:'TV 0.1',l2:8.74},{t:'TV 1',l2:27.7},{t:'Lap 0.01',l2:57.1},{t:'Lap 0.1',l2:9.32},{t:'Lap 1',l2:10.7}]
};
let regChart;
function initRegChart() {
  const el = document.getElementById('chartReg');
  if (!el) return;
  regChart = new Chart(el, {
    type: 'bar',
    data: {
      labels: regData[0].map(d => d.t),
      datasets: [{
        label: 'L2 相对误差 %', data: regData[0].map(d => d.l2),
        backgroundColor: regData[0].map(d => d.l2 < 5 ? C.bgFill(C.green,0.6) : d.l2 < 10 ? C.bgFill(C.amber,0.6) : C.bgFill(C.red,0.5)),
        borderColor: regData[0].map(d => d.l2 < 5 ? C.green : d.l2 < 10 ? C.amber : C.red), borderWidth: 1
      }]
    },
    options: chartOpts({ plugins: { legend: { display: false } }, scales: { y: linAxis('L2 误差 %') } })
  });
}
function updateRegChart(noise, btn) {
  btn.parentElement.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const d = regData[noise];
  regChart.data.labels = d.map(x => x.t);
  regChart.data.datasets[0].data = d.map(x => x.l2);
  regChart.data.datasets[0].backgroundColor = d.map(x => x.l2 < 5 ? C.bgFill(C.green,0.6) : x.l2 < 10 ? C.bgFill(C.amber,0.6) : C.bgFill(C.red,0.5));
  regChart.data.datasets[0].borderColor = d.map(x => x.l2 < 5 ? C.green : x.l2 < 10 ? C.amber : C.red);
  regChart.update();
}
initRegChart();

// ═══════════════════════════════════════
// E-FIELD HEATMAPS (I-J section)
// ═══════════════════════════════════════

function generateField(type, N) {
  const field = [];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    if (type === 'tworegion') field.push(x < N/2 ? 50000 : 90000);
    else if (type === 'layered') {
      if (y < N/4) field.push(40000);
      else if (y < N/2) field.push(70000);
      else if (y < 3*N/4) field.push(90000);
      else field.push(60000);
    } else {
      field.push(50000 + Math.random() * 40000);
    }
  }
  return field;
}
function invertField(field, quality) {
  return field.map(v => {
    const noise = (Math.random() - 0.5) * 2 * v * quality;
    return Math.max(20000, Math.min(120000, v + noise));
  });
}
function renderHeatmap(container, field, N) {
  if (!container) return;
  container.innerHTML = '';
  container.classList.add('heatmap');
  container.style.gridTemplateColumns = `repeat(${N},1fr)`;
  const mn = Math.min(...field), mx = Math.max(...field);
  const colors = ['#3d5a99','#5a7ab5','#7a9acc','#a0bade','#e5d8c0','#d4a86a','#b57040','#8b3030'];
  field.forEach(v => {
    const t = (v - mn) / (mx - mn + 1);
    const ci = Math.min(Math.floor(t * colors.length), colors.length - 1);
    const cell = document.createElement('div');
    cell.className = 'heatmap-cell';
    cell.style.background = colors[ci];
    const tip = document.createElement('div');
    tip.className = 'heatmap-tooltip';
    tip.textContent = `E=${Math.round(v)}`;
    cell.appendChild(tip);
    container.appendChild(cell);
  });
}
function updateEfield(type, btn) {
  btn.parentElement.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const N = 20;
  const trueF = generateField(type, N);
  const qual = type === 'random' ? 0.03 : type === 'tworegion' ? 0.04 : 0.15;
  const invF = invertField(trueF, qual);
  renderHeatmap(document.getElementById('heatmapTrue'), trueF, N);
  renderHeatmap(document.getElementById('heatmapInv'), invF, N);
}

// ═══════════════════════════════════════
// P0 COMPARE SLIDER
// ═══════════════════════════════════════

function updateP0Compare() {
  const v = parseInt(document.getElementById('p0Slider').value);
  document.getElementById('p0SliderVal').textContent = v + '%';
  document.getElementById('p0BarBefore').style.width = v + '%';
}

// ═══════════════════════════════════════
// L SCALING
// ═══════════════════════════════════════

new Chart(document.getElementById('chartLScaling'), {
  type: 'line',
  data: {
    labels: ['25','100','400','900'],
    datasets: [
      { label: 'AD 加速比', data: [28,111,445,1099], borderColor: C.blue,
        backgroundColor: C.bgFill(C.blue,0.08), fill: true, pointRadius: 6, tension: 0.3, borderWidth: 2 },
      { label: '理论 N', data: [25,100,400,900], borderColor: C.text2, borderDash: [4,4], pointRadius: 0, borderWidth: 1 }
    ]
  },
  options: chartOpts({ scales: { y: { ...logYAxis('加速比') }, x: linAxis('参数维度 N') } })
});

// ═══════════════════════════════════════
// MC CHARTS
// ═══════════════════════════════════════

new Chart(document.getElementById('chartMCHistory'), {
  type: 'bar',
  data: {
    labels: ['v1 单步\n(doc 20)','v2 增量5步\n(doc 21)','v3 双阶段\n(doc 21)','v4 未修复\n(doc 25)','v5 修复后\n(doc 25)'],
    datasets: [{
      label: 'L2 相对误差 %',
      data: [19.31, 10.35, 9.97, 50, 7.09],
      backgroundColor: [C.bgFill(C.red,0.5), C.bgFill(C.amber,0.5), C.bgFill(C.amber,0.5), C.bgFill(C.red,0.5), C.bgFill(C.green,0.5)],
      borderColor: [C.red, C.amber, C.amber, C.red, C.green], borderWidth: 1
    }]
  },
  options: chartOpts({ plugins: { legend: { display: false } }, scales: { y: { ...linAxis('L2 误差 %'), max: 55 }, x: { ticks: { font: { size: 10 } } } } })
});

new Chart(document.getElementById('chartMCJoint'), {
  type: 'bar',
  data: {
    labels: ['c(x) L2','φ(x) L2'],
    datasets: [{
      label: 'L2 %', data: [6.95, 6.21],
      backgroundColor: [C.bgFill(C.green,0.5), C.bgFill(C.green,0.5)],
      borderColor: [C.green, C.green], borderWidth: 1, barThickness: 50
    }]
  },
  options: chartOpts({
    plugins: { legend: { display: false } },
    scales: { x: { ...linAxis('L2 误差 %'), max: 12 }, y: {} }
  }).indexAxis === undefined ? { ...chartOpts({ plugins: { legend: { display: false } }, scales: { x: { ...linAxis('L2 误差 %'), max: 12 } } }), indexAxis: 'y' } : {}
});
// Fix: recreate with indexAxis
(function() {
  const el = document.getElementById('chartMCJoint');
  if (!el) return;
  const existing = Chart.getChart(el);
  if (existing) existing.destroy();
  new Chart(el, {
    type: 'bar',
    data: {
      labels: ['c(x) L2','φ(x) L2'],
      datasets: [{
        label: 'L2 %', data: [6.95, 6.21],
        backgroundColor: [C.bgFill(C.green,0.5), C.bgFill(C.teal,0.5)],
        borderColor: [C.green, C.teal], borderWidth: 1, barThickness: 40
      }]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { ...linAxis('L2 误差 %'), max: 12, grid: { color: '#eeeae6' } }, y: {} }
    }
  });
})();

new Chart(document.getElementById('chartMCRobust'), {
  type: 'bar',
  data: {
    labels: ['T1 基线','T2 1%噪声','T3 1%+正则','T4 3%+正则','T5 稀疏+正则','T6 3区域'],
    datasets: [{
      label: 'L2 相对误差 %',
      data: [7.09, 17.37, 16.75, 16.62, 19.40, 6.23],
      backgroundColor: [C.bgFill(C.green,0.5),C.bgFill(C.amber,0.5),C.bgFill(C.amber,0.5),
                        C.bgFill(C.amber,0.5),C.bgFill(C.amber,0.5),C.bgFill(C.green,0.5)],
      borderColor: [C.green,C.amber,C.amber,C.amber,C.amber,C.green], borderWidth: 1
    }]
  },
  options: chartOpts({ plugins: { legend: { display: false } }, scales: { y: { ...linAxis('L2 误差 %'), max: 25 } } })
});

// MC Regularization sweep chart
const mcRegEl = document.getElementById('chartMCRegSweep');
if (mcRegEl) {
  new Chart(mcRegEl, {
    type: 'line',
    data: {
      labels: ['0','1e-6','3e-6','5e-6','8e-6','1e-5','1e-4','5e-4','1e-3'],
      datasets: [{
        label: 'L2 误差 %',
        data: [54.38, 22.10, 17.12, 16.62, 16.96, 17.24, 29.63, 41.65, 44.20],
        borderColor: C.blue, backgroundColor: C.bgFill(C.blue, 0.08), fill: true,
        pointRadius: 5, tension: 0.3,
        pointBackgroundColor: ['#b54444','#b58a2a','#b58a2a','#3d7a52','#b58a2a','#b58a2a','#b54444','#b54444','#b54444']
      }]
    },
    options: chartOpts({ scales: { y: linAxis('L2 误差 %'), x: linAxis('正则化权重 λ') } })
  });
}

// ═══════════════════════════════════════
// DIAGNOSIS DETAILS
// ═══════════════════════════════════════

const diagDetails = {
  1: '<h4>实验 1：FD 步长扫描</h4><p>扫描 11 个步长（0.1 到 500），FD 梯度在较大范围内保持稳定。</p><p><strong>结论：</strong><span class="val-good">FD 可信</span>，问题在 AD 一侧。</p>',
  2: '<h4>实验 2：Active Set 分析</h4><p>检查 64 个积分点在 params±ε 下的塑性/弹性状态，<strong>无任何状态切换</strong>。</p><p><strong>结论：</strong><span class="val-good">排除屈服面切换导致不可微</span></p>',
  3: '<h4>实验 3：单点 AD vs FD</h4><p>取出 FEM 求解的 ∇u，在单个积分点上对 return mapping 做 AD/FD 对比。<strong>完美一致。</strong></p><p><strong>关键推论：</strong><span class="val-good">DP 回映算法本身可微</span>，问题在 FEM 求解器层面。</p>',
  4: '<h4>实验 4：Taylor Test</h4><p>r₁(ε) 应以 O(ε²) 衰减（斜率=2）。<strong>实际斜率=1.0</strong>，严格证明 AD 梯度完全错误——不是"近似正确"，而是"另一个量"。</p>',
  5: '<h4>实验 5：光滑化扫描</h4><p>用 softplus 替代 jnp.where 的 hard switch，扫描 β∈{1,5,10,50,100,500}。<strong>所有设定下误差完全相同。</strong></p><p>原因：所有 64 个积分点都深入塑性，开关永远在"开"的一侧。</p>',
  6: '<h4>实验 6：位移扫描</h4><p>从 -0.020mm 到 -0.035mm 扫描。误差在 -0.026mm（弹塑性转变点）<strong>突然从 10⁻¹⁴ 跳到 99%</strong>。不是渐变，是跳变。</p>',
  7: '<h4>实验 7–10：VJP 深层分解</h4><p><strong>实验 7：</strong>将梯度分解为 Direct + Implicit 两条路径。Direct AD 与 FD 完美一致。Implicit 产生了巨大的虚假值。</p><p><strong>实验 8：</strong>Implicit 梯度几乎完全<strong>抵消</strong>了正确的 Direct 贡献。</p><p><strong>实验 9：</strong>修改 self.E_val 后，JIT kernel 计算的残差<strong>纹丝不动</strong>——JIT 缓存了旧值。</p><p><strong>实验 10：</strong>在 jax.vjp trace 下，JIT kernel 产生<strong>完全虚假的非零 VJP</strong>。</p><p style="margin-top:12px;padding:12px;background:var(--success-bg);border-radius:4px;border-left:3px solid var(--success)"><strong>根因定位：</strong>self.E_val 通过闭包传入 JIT kernel → JIT 视其为编译常量 → jax.vjp trace 时产生虚假 ∂c/∂p → 伴随方程传播出完全错误的梯度。</p>'
};

function showDiagDetail(n) {
  const el = document.getElementById('diagDetail');
  if (!el) return;
  el.style.display = 'block';
  el.innerHTML = diagDetails[n] || '';
}

// ═══════════════════════════════════════
// INIT E-FIELD on load
// ═══════════════════════════════════════

setTimeout(() => {
  const btn = document.querySelector('#sec-ij .btn-group .btn');
  if (btn) updateEfield('tworegion', btn);
}, 100);
