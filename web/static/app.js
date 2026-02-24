/** @typedef {Object} MetricsPoint */
/** @typedef {Object<string, any>} ChartInstance */

/** @type {Object<string, ChartInstance>} */
let charts = {};

/** @type {WebSocket | null} */
let ws = null;

/** @type {boolean} */
let isRunning = false;

function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
            x: { title: { display: true, text: 'Step' } },
            y: { beginAtZero: false },
        },
        plugins: { legend: { display: true, position: 'top' } },
    };

    charts.distSps = new Chart(document.getElementById('distSpsChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Distributed SPS', data: [], borderColor: '#667eea', backgroundColor: 'rgba(102, 126, 234, 0.1)', fill: true, tension: 0.4 }] },
        options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, beginAtZero: true } } },
    });

    charts.distReturn = new Chart(document.getElementById('distReturnChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Distributed Return', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)', fill: true, tension: 0.4 }] },
        options: commonOptions,
    });

    charts.distLoss = new Chart(document.getElementById('distLossChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Distributed Loss', data: [], borderColor: '#dc3545', backgroundColor: 'rgba(220, 53, 69, 0.1)', fill: true, tension: 0.4 }] },
        options: commonOptions,
    });

    charts.distBuffer = new Chart(document.getElementById('distBufferChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Distributed Buffer', data: [], borderColor: '#28a745', backgroundColor: 'rgba(40, 167, 69, 0.1)', fill: true, tension: 0.4 }] },
        options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, beginAtZero: true } } },
    });

    charts.singleSps = new Chart(document.getElementById('singleSpsChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Single SPS', data: [], borderColor: '#667eea', backgroundColor: 'rgba(102, 126, 234, 0.1)', fill: true, tension: 0.4 }] },
        options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, beginAtZero: true } } },
    });

    charts.singleReturn = new Chart(document.getElementById('singleReturnChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Single Return', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)', fill: true, tension: 0.4 }] },
        options: commonOptions,
    });

    charts.singleLoss = new Chart(document.getElementById('singleLossChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Single Loss', data: [], borderColor: '#dc3545', backgroundColor: 'rgba(220, 53, 69, 0.1)', fill: true, tension: 0.4 }] },
        options: commonOptions,
    });

    charts.singleBuffer = new Chart(document.getElementById('singleBufferChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Single Buffer', data: [], borderColor: '#28a745', backgroundColor: 'rgba(40, 167, 69, 0.1)', fill: true, tension: 0.4 }] },
        options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, beginAtZero: true } } },
    });

    charts.compSps = new Chart(document.getElementById('compSpsChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Distributed', data: [], borderColor: '#667eea', backgroundColor: 'rgba(102, 126, 234, 0.1)', fill: true, tension: 0.4 },
                { label: 'Single', data: [], borderColor: '#20c997', backgroundColor: 'rgba(32, 201, 151, 0.1)', fill: true, tension: 0.4 }
            ]
        },
        options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, beginAtZero: true } } },
    });

    charts.compReturn = new Chart(document.getElementById('compReturnChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Distributed', data: [], borderColor: '#667eea', backgroundColor: 'rgba(102, 126, 234, 0.1)', fill: true, tension: 0.4 },
                { label: 'Single', data: [], borderColor: '#20c997', backgroundColor: 'rgba(32, 201, 151, 0.1)', fill: true, tension: 0.4 }
            ]
        },
        options: commonOptions,
    });

    charts.compLoss = new Chart(document.getElementById('compLossChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Distributed', data: [], borderColor: '#667eea', backgroundColor: 'rgba(102, 126, 234, 0.1)', fill: true, tension: 0.4 },
                { label: 'Single', data: [], borderColor: '#20c997', backgroundColor: 'rgba(32, 201, 151, 0.1)', fill: true, tension: 0.4 }
            ]
        },
        options: commonOptions,
    });

    charts.compBuffer = new Chart(document.getElementById('compBufferChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Distributed', data: [], borderColor: '#667eea', backgroundColor: 'rgba(102, 126, 234, 0.1)', fill: true, tension: 0.4 },
                { label: 'Single', data: [], borderColor: '#20c997', backgroundColor: 'rgba(32, 201, 151, 0.1)', fill: true, tension: 0.4 }
            ]
        },
        options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, beginAtZero: true } } },
    });
}

/**
 * @param {ChartInstance} chart
 * @param {MetricsPoint[]} metrics
 * @param {string} key
 * @param {string} label
 */
function updateSingleChart(chart, metrics, key, label) {
    const labels = metrics.map(m => m.step);
    chart.data.labels = labels;
    chart.data.datasets[0].data = metrics.map(m => m[key]);
    chart.data.datasets[0].label = label;
    chart.update('none');
}

/**
 * @param {ChartInstance} chart
 * @param {MetricsPoint[]} distMetrics
 * @param {MetricsPoint[]} singleMetrics
 * @param {string} key
 */
function updateComparisonChart(chart, distMetrics, singleMetrics, key) {
    const distLabels = distMetrics.map(m => m.step);
    const singleLabels = singleMetrics.map(m => m.step);
    const allLabels = Array.from(new Set([...distLabels, ...singleLabels])).sort((a, b) => a - b);
    chart.data.labels = allLabels;
    chart.data.datasets[0].data = allLabels.map(step => {
        const m = distMetrics.find(x => x.step === step);
        return m ? m[key] : null;
    });
    chart.data.datasets[1].data = allLabels.map(step => {
        const m = singleMetrics.find(x => x.step === step);
        return m ? m[key] : null;
    });
    chart.update('none');
}

/**
 * @param {MetricsPoint[]} distMetrics
 * @param {MetricsPoint[]} singleMetrics
 */
function updateAllCharts(distMetrics, singleMetrics) {
    if (distMetrics.length > 0) {
        updateSingleChart(charts.distSps, distMetrics, 'sps', 'Distributed SPS');
        updateSingleChart(charts.distReturn, distMetrics, 'avg_return', 'Distributed Return');
        updateSingleChart(charts.distLoss, distMetrics, 'loss', 'Distributed Loss');
        updateSingleChart(charts.distBuffer, distMetrics, 'buffer_size', 'Distributed Buffer');
    }
    if (singleMetrics.length > 0) {
        updateSingleChart(charts.singleSps, singleMetrics, 'sps', 'Single SPS');
        updateSingleChart(charts.singleReturn, singleMetrics, 'avg_return', 'Single Return');
        updateSingleChart(charts.singleLoss, singleMetrics, 'loss', 'Single Loss');
        updateSingleChart(charts.singleBuffer, singleMetrics, 'buffer_size', 'Single Buffer');
    }
    updateComparisonChart(charts.compSps, distMetrics, singleMetrics, 'sps');
    updateComparisonChart(charts.compReturn, distMetrics, singleMetrics, 'avg_return');
    updateComparisonChart(charts.compLoss, distMetrics, singleMetrics, 'loss');
    updateComparisonChart(charts.compBuffer, distMetrics, singleMetrics, 'buffer_size');
}

/**
 * @param {string} text
 * @param {boolean} [running=false]
 */
function setStatus(text, running = false) {
    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.textContent = text;
        statusEl.className = 'status' + (running ? ' running' : '');
    }
}

async function startDistributedTraining() {
    try {
        const resp = await fetch('/api/training/distributed/start', { method: 'POST' });
        const data = await resp.json();
        if (data.status === 'started' || data.status === 'already running') {
            isRunning = true;
            const startDistBtn = document.getElementById('startDistBtn');
            const startSingleBtn = document.getElementById('startSingleBtn');
            const stopBtn = document.getElementById('stopBtn');
            if (startDistBtn) startDistBtn.disabled = true;
            if (startSingleBtn) startSingleBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            setStatus('状态: 分布式训练中...', true);
            connectWebSocket();
        }
    } catch (e) {
        console.error(e);
    }
}

async function startSingleTraining() {
    try {
        const resp = await fetch('/api/training/single/start', { method: 'POST' });
        const data = await resp.json();
        if (data.status === 'started' || data.status === 'already running') {
            isRunning = true;
            const startDistBtn = document.getElementById('startDistBtn');
            const startSingleBtn = document.getElementById('startSingleBtn');
            const stopBtn = document.getElementById('stopBtn');
            if (startDistBtn) startDistBtn.disabled = true;
            if (startSingleBtn) startSingleBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            setStatus('状态: 单机训练中...', true);
            connectWebSocket();
        }
    } catch (e) {
        console.error(e);
    }
}

async function stopTraining() {
    try {
        await fetch('/api/training/stop', { method: 'POST' });
        isRunning = false;
        const startDistBtn = document.getElementById('startDistBtn');
        const startSingleBtn = document.getElementById('startSingleBtn');
        const stopBtn = document.getElementById('stopBtn');
        if (startDistBtn) startDistBtn.disabled = false;
        if (startSingleBtn) startSingleBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        setStatus('状态: 已停止');
        if (ws) {
            ws.close();
        }
    } catch (e) {
        console.error(e);
    }
}

async function refreshMetrics() {
    try {
        const [distResp, singleResp] = await Promise.all([
            fetch('/api/metrics/distributed'),
            fetch('/api/metrics/single')
        ]);
        const distMetrics = await distResp.json();
        const singleMetrics = await singleResp.json();
        updateAllCharts(distMetrics, singleMetrics);
    } catch (e) {
        console.error(e);
    }
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/training`);
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'metrics') {
            updateAllCharts(msg.distributed || [], msg.single || []);
        }
    };
    ws.onclose = () => {
        if (isRunning) {
            setTimeout(connectWebSocket, 1000);
        }
    };
}

/**
 * @param {string} tabName
 */
function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const activeTab = document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`);
    const activeContent = document.getElementById(`tab-${tabName}`);
    if (activeTab) activeTab.classList.add('active');
    if (activeContent) activeContent.classList.add('active');
}

initCharts();
refreshMetrics();
