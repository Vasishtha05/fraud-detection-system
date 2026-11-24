
const API_BASE = window.location.protocol.startsWith("http")
    ? window.location.origin
    : "http://127.0.0.1:8000";

const form = document.getElementById("predictionForm");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBox = document.getElementById("predictionResult");
const historyBody = document.getElementById("historyBody");
const apiStatus = document.getElementById("apiStatus");

async function apiRequest(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {})
        }
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            data.detail || `Request failed with status ${response.status}`
        );
    }

    return data;
}

function formatAmount(value) {
    return Number(value ?? 0).toLocaleString("en-IN", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function formatProbability(value) {
    return `${(Number(value ?? 0) * 100).toFixed(2)}%`;
}

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[character]);
}

async function checkAPI() {
    try {
        await apiRequest("/");

        apiStatus.textContent = "Connected";
        apiStatus.style.color = "#69e0aa";

        const dot = document.querySelector(".status-dot");
        if (dot) dot.style.background = "#35d99a";
    } catch (error) {
        apiStatus.textContent = "Disconnected — check Docker API";
        apiStatus.style.color = "#ffb3bd";

        const dot = document.querySelector(".status-dot");
        if (dot) dot.style.background = "#dc4055";
    }
}

async function loadHistory() {
    historyBody.innerHTML =
        '<tr><td colspan="6">Loading prediction history...</td></tr>';

    try {
        const data = await apiRequest("/predictions?limit=50");

        // Matches your API response: { count, predictions: [...] }
        const records = Array.isArray(data)
            ? data
            : (data.predictions || data.records || []);

        document.getElementById("totalPredictions").textContent =
            data.count ?? records.length;

        const fraudCount = records.filter(record =>
            Number(record.prediction) === 1
        ).length;

        document.getElementById("fraudCount").textContent = fraudCount;
        document.getElementById("legitimateCount").textContent =
            records.length - fraudCount;

        if (records.length === 0) {
            historyBody.innerHTML =
                '<tr><td colspan="6">No prediction records yet.</td></tr>';
            return;
        }

        historyBody.innerHTML = records.map(record => {
            const isFraud = Number(record.prediction) === 1;
            const label = isFraud ? "FRAUD" : "LEGITIMATE";

            return `
                <tr>
                    <td>${escapeHTML(record.id ?? record.record_id ?? "—")}</td>
                    <td>${escapeHTML(record.step ?? "—")}</td>
                    <td>${escapeHTML(
                        record.transaction_type ?? record.type ?? "—"
                    )}</td>
                    <td>₹${formatAmount(record.amount)}</td>
                    <td>${formatProbability(record.fraud_probability)}</td>
                    <td>
                        <span class="badge ${isFraud ? "fraud" : "legitimate"}">
                            ${label}
                        </span>
                    </td>
                </tr>
            `;
        }).join("");

    } catch (error) {
        historyBody.innerHTML = `
            <tr>
                <td colspan="6">
                    Could not load history: ${escapeHTML(error.message)}
                </td>
            </tr>
        `;

        document.getElementById("totalPredictions").textContent = "—";
        document.getElementById("fraudCount").textContent = "—";
        document.getElementById("legitimateCount").textContent = "—";
    }
}

form.addEventListener("submit", async event => {
    event.preventDefault();

    const transaction = {
        step: Number(document.getElementById("step").value),
        type: document.getElementById("type").value,
        amount: Number(document.getElementById("amount").value),
        isFlaggedFraud: Number(document.getElementById("flagged").value)
    };

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";
    resultBox.className = "result-box";
    resultBox.textContent = "Sending transaction to the ML model...";

    try {
        const result = await apiRequest("/predict", {
            method: "POST",
            body: JSON.stringify(transaction)
        });

        const predictionData = result.prediction ?? result;
        const prediction = Number(predictionData.prediction);

        const probability = Number(
            predictionData.fraud_probability ??
            (predictionData.fraud_probability_percent != null
                ? Number(predictionData.fraud_probability_percent) / 100
                : NaN)
        );

        if (!Number.isFinite(prediction) ||
            !Number.isFinite(probability)) {
            throw new Error("Invalid prediction response received from API");
        }

        const label = predictionData.result ||
            (prediction === 1 ? "FRAUD" : "LEGITIMATE");

        const threshold = Number(
            predictionData.threshold ??
            predictionData.decision_threshold ??
            result.threshold ??
            result.decision_threshold ??
            0.9
        );

        const recordId =
            predictionData.record_id ??
            predictionData.id ??
            result.record_id ??
            result.id ??
            "Saved";

        resultBox.className =
            `result-box ${prediction === 1 ? "fraud" : "legitimate"}`;

        resultBox.innerHTML = `
            <h3>${prediction === 1 ? "⚠️" : "✓"} ${escapeHTML(label)}</h3>
            <p><strong>Fraud probability:</strong> ${formatProbability(probability)}</p>
            <p><strong>Decision threshold:</strong> ${formatProbability(threshold)}</p>
            <p><strong>Record ID:</strong> ${escapeHTML(recordId)}</p>
            <p>Model predictions are estimates, not guarantees.</p>
        `;

        await loadHistory();
        await checkAPI();

    } catch (error) {
        resultBox.className = "result-box error";
        resultBox.textContent = `Prediction failed: ${error.message}`;
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze Transaction →";
    }
});

document.getElementById("refreshBtn").addEventListener("click", async () => {
    await checkAPI();
    await loadHistory();
});

async function initializeDashboard() {
    await checkAPI();
    await loadHistory();
}

initializeDashboard();
