/**
 * Dashboard — HQC Research Console Home
 * Shows corpus stats, recent runs, system status, and quick-launch pipeline.
 * Blueprint §4 compliant — all data fetched from API contracts.
 */

import { useState, useEffect, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ingestApi, hybridApi, evalApi, healthApi } from "../services/api";

// ─── Sub-components ────────────────────────────────────────────────────────────

function StatCard({ label, value, unit, accent, icon, sublabel, loading }) {
  return (
    <div className="stat-card" style={{ "--accent": accent }}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-body">
        <div className="stat-value">
          {loading ? <span className="stat-skel" /> : (
            <>{value}<span className="stat-unit">{unit}</span></>
          )}
        </div>
        <div className="stat-label">{label}</div>
        {sublabel && <div className="stat-sublabel">{sublabel}</div>}
      </div>
      <div className="stat-bar" />
    </div>
  );
}

function CorpusRow({ corpus, onSelect, selected }) {
  const docCount = corpus.doc_count ?? "—";
  const vocabSize = corpus.vocab_size ?? "—";
  return (
    <tr
      className={`corpus-row ${selected ? "selected" : ""}`}
      onClick={() => onSelect(corpus.corpus_id)}
    >
      <td>
        <span className="corpus-id">{corpus.corpus_id}</span>
      </td>
      <td>{docCount.toLocaleString?.() ?? docCount}</td>
      <td>{vocabSize.toLocaleString?.() ?? vocabSize}</td>
      <td>
        <span className={`badge badge-${corpus.status ?? "ready"}`}>
          {corpus.status ?? "READY"}
        </span>
      </td>
      <td className="corpus-ts">
        {corpus.created_at
          ? new Date(corpus.created_at).toLocaleDateString()
          : "—"}
      </td>
    </tr>
  );
}

function RecentRunRow({ run }) {
  const statusColors = {
    done:    "#10ffb0",
    running: "#f59e0b",
    failed:  "#ff4466",
    pending: "#6b7280",
  };
  const color = statusColors[run.status] ?? "#888";
  return (
    <div className="run-row">
      <div className="run-id-col">
        <span className="run-id-mono">{run.run_id ?? run.job_id ?? "—"}</span>
        <span className="run-method">{run.method ?? "hybrid"}</span>
      </div>
      <div className="run-metrics">
        {run.silhouette_hybrid != null && (
          <span className="metric-pill">
            sil <b>{run.silhouette_hybrid?.toFixed(3)}</b>
          </span>
        )}
        {run.topic_coherence_cv != null && (
          <span className="metric-pill">
            C_v <b>{run.topic_coherence_cv?.toFixed(3)}</b>
          </span>
        )}
      </div>
      <div className="run-status" style={{ color }}>
        <span className="run-dot" style={{ background: color }} />
        {run.status?.toUpperCase()}
      </div>
    </div>
  );
}

// ─── Supported datasets from Blueprint §7 ─────────────────────────────────────
const DATASETS = [
  { id: "20ng",        label: "20 Newsgroups", docs: 18846, classes: 20 },
  { id: "reuters",     label: "Reuters-21578", docs: 10788, classes: 90 },
  { id: "bbc",         label: "BBC News",      docs: 2225,  classes: 5  },
  { id: "agnews_mini", label: "AG News (mini)", docs: 5000, classes: 4  },
];

const SUBSETS = [
  { label: "Toy (50 docs)", size: 50,  desc: "Circuit debugging" },
  { label: "Small (200 docs)", size: 200, desc: "QAOA feasibility" },
  { label: "Medium (500 docs)", size: 500, desc: "Hybrid comparison" },
];

// ─── Main Dashboard Component ──────────────────────────────────────────────────
export default function Dashboard() {
  const navigate = useNavigate();

  const [systemStatus, setSystemStatus] = useState(null);
  const [recentRuns, setRecentRuns] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingRuns, setLoadingRuns] = useState(true);

  // Quick-launch state
  const [selectedDataset, setSelectedDataset] = useState("20ng");
  const [selectedSubset, setSelectedSubset] = useState(200);
  const [launching, setLaunching] = useState(false);
  const [launchMsg, setLaunchMsg] = useState(null); // { type: 'ok'|'err', text }

  // Upload state
  const [uploadProgress, setUploadProgress] = useState(null);

  // ── Fetch system data ────────────────────────────────────────────────────────
  const fetchData = useCallback(async () => {
    try {
      const [health, docs, runs] = await Promise.allSettled([
        healthApi.check(),
        ingestApi.listDocuments(0, 500),
        evalApi.getAllResults(0, 8),
      ]);

      if (health.status === "fulfilled") setSystemStatus(health.value);
      if (docs.status === "fulfilled") setDocuments(docs.value?.documents ?? docs.value ?? []);
      if (runs.status === "fulfilled") setRecentRuns(runs.value?.results ?? runs.value ?? []);
    } catch (e) {
      console.error("Dashboard fetch error:", e);
    } finally {
      setLoading(false);
      setLoadingRuns(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 15_000);
    return () => clearInterval(id);
  }, [fetchData]);

  // ── Quick Launch ─────────────────────────────────────────────────────────────
  const handleQuickLaunch = async () => {
    setLaunching(true);
    setLaunchMsg(null);
    try {
      // Step 1: load dataset
      const loaded = await ingestApi.loadDataset(selectedDataset, selectedSubset, null);
      const corpusId = loaded.corpus_id ?? loaded.dataset_name ?? selectedDataset;
      // Step 2: trigger hybrid pipeline
      const run = await hybridApi.run({
        corpus_id: corpusId,
        method: "hybrid",
        num_topics: 5,
        num_clusters: 5,
        qaoa_layers: 2,
        enable_noise: true,
        enable_mitigation: true,
        dataset_name: selectedDataset,
      });
      setLaunchMsg({ type: "ok", text: `Pipeline launched → job ${run.run_id ?? run.job_id}` });
      setTimeout(() => navigate("/pipeline"), 1500);
    } catch (err) {
      setLaunchMsg({ type: "err", text: err.message ?? "Launch failed" });
    } finally {
      setLaunching(false);
    }
  };

  // ── File upload ──────────────────────────────────────────────────────────────
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadProgress(0);
    try {
      await ingestApi.upload(file, "default", (pct) => setUploadProgress(pct));
      setUploadProgress(null);
      fetchData();
    } catch (err) {
      setUploadProgress(null);
      alert("Upload failed: " + err.message);
    }
  };

  // ── Stats derived from data ──────────────────────────────────────────────────
  const totalDocs = documents.length;
  const doneRuns  = recentRuns.filter((r) => r.status === "done").length;
  const avgSil    = recentRuns
    .filter((r) => r.silhouette_hybrid != null)
    .reduce((s, r, _, a) => s + r.silhouette_hybrid / a.length, 0);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;800&display=swap');

        /* ── Root ── */
        .dashboard {
          min-height: calc(100vh - 60px);
          background: #06080e;
          padding: 80px 1.5rem 3rem;
          font-family: 'Share Tech Mono', monospace;
          color: #c8dcd8;
          position: relative;
          overflow: hidden;
        }

        /* Grid noise texture */
        .dashboard::before {
          content: '';
          position: fixed;
          inset: 0;
          background-image:
            linear-gradient(rgba(0,230,170,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,230,170,0.015) 1px, transparent 1px);
          background-size: 40px 40px;
          pointer-events: none;
          z-index: 0;
        }

        .dash-content {
          position: relative;
          z-index: 1;
          max-width: 1400px;
          margin: 0 auto;
        }

        /* ── Page Header ── */
        .dash-header {
          margin-bottom: 2.5rem;
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .dash-title {
          font-family: 'Exo 2', sans-serif;
          font-weight: 800;
          font-size: clamp(1.4rem, 3vw, 2.2rem);
          color: #e8f4f0;
          letter-spacing: -0.01em;
          line-height: 1.15;
        }

        .dash-title span {
          color: #00e6aa;
          text-shadow: 0 0 20px rgba(0,230,170,0.4);
        }

        .dash-subtitle {
          font-size: 0.72rem;
          color: rgba(0, 230, 170, 0.45);
          letter-spacing: 0.18em;
          text-transform: uppercase;
          margin-top: 0.3rem;
        }

        .header-actions {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .upload-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          border: 1px solid rgba(0,180,255,0.3);
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.75rem;
          color: rgba(0,180,255,0.8);
          letter-spacing: 0.08em;
          transition: all 0.18s;
          font-family: 'Share Tech Mono', monospace;
        }

        .upload-label:hover {
          background: rgba(0,180,255,0.06);
          border-color: rgba(0,180,255,0.5);
          color: #00b4ff;
        }

        .upload-label input { display: none; }

        /* ── Progress bar ── */
        .upload-bar-wrap {
          height: 3px;
          background: rgba(0,180,255,0.1);
          border-radius: 2px;
          margin-bottom: 2rem;
          overflow: hidden;
        }
        .upload-bar {
          height: 100%;
          background: linear-gradient(90deg, #00b4ff, #00e6aa);
          border-radius: 2px;
          transition: width 0.2s;
          box-shadow: 0 0 8px #00b4ff;
        }

        /* ── Stats Grid ── */
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1px;
          margin-bottom: 2rem;
          border: 1px solid rgba(0,230,170,0.1);
          border-radius: 6px;
          overflow: hidden;
          background: rgba(0,230,170,0.08);
        }

        .stat-card {
          background: #080b12;
          padding: 1.5rem;
          position: relative;
          display: flex;
          gap: 1rem;
          align-items: flex-start;
          overflow: hidden;
          transition: background 0.2s;
        }

        .stat-card:hover { background: #0a0e18; }

        .stat-card .stat-bar {
          position: absolute;
          bottom: 0; left: 0; right: 0;
          height: 2px;
          background: var(--accent, #00e6aa);
          opacity: 0.4;
          transform: scaleX(0);
          transform-origin: left;
          transition: transform 0.4s ease;
        }

        .stat-card:hover .stat-bar { transform: scaleX(1); }

        .stat-icon {
          font-size: 1.5rem;
          opacity: 0.7;
          flex-shrink: 0;
          margin-top: 0.2rem;
        }

        .stat-value {
          font-family: 'Exo 2', sans-serif;
          font-weight: 700;
          font-size: 1.8rem;
          color: var(--accent, #00e6aa);
          line-height: 1;
          text-shadow: 0 0 12px var(--accent, rgba(0,230,170,0.3));
        }

        .stat-unit {
          font-size: 0.85rem;
          font-weight: 400;
          margin-left: 0.3rem;
          opacity: 0.6;
        }

        .stat-label {
          font-size: 0.68rem;
          letter-spacing: 0.15em;
          color: rgba(200,220,215,0.45);
          text-transform: uppercase;
          margin-top: 0.3rem;
        }

        .stat-sublabel {
          font-size: 0.62rem;
          color: rgba(200,220,215,0.25);
          margin-top: 0.15rem;
        }

        .stat-skel {
          display: inline-block;
          width: 60px;
          height: 1.8rem;
          background: linear-gradient(90deg, #111 25%, #1a1a1a 50%, #111 75%);
          background-size: 200% 100%;
          animation: shimmer 1.5s infinite;
          border-radius: 3px;
        }

        @keyframes shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }

        /* ── Main Grid ── */
        .main-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1.5rem;
          margin-bottom: 1.5rem;
        }

        @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }

        /* ── Panel ── */
        .panel {
          background: #080b12;
          border: 1px solid rgba(0,230,170,0.1);
          border-radius: 6px;
          overflow: hidden;
        }

        .panel-header {
          padding: 1rem 1.5rem;
          border-bottom: 1px solid rgba(0,230,170,0.08);
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .panel-title {
          font-size: 0.72rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: rgba(0,230,170,0.6);
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .panel-title-icon {
          font-size: 1rem;
          opacity: 0.8;
        }

        .panel-action {
          font-size: 0.68rem;
          color: rgba(0,180,255,0.5);
          text-decoration: none;
          letter-spacing: 0.1em;
          transition: color 0.18s;
        }

        .panel-action:hover { color: #00b4ff; }

        .panel-body { padding: 1.25rem 1.5rem; }

        /* ── Corpus Table ── */
        .corpus-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.75rem;
        }

        .corpus-table th {
          text-align: left;
          padding: 0 0 0.75rem 0;
          font-size: 0.62rem;
          letter-spacing: 0.15em;
          text-transform: uppercase;
          color: rgba(200,220,215,0.3);
          border-bottom: 1px solid rgba(0,230,170,0.06);
          font-weight: 400;
        }

        .corpus-row {
          border-bottom: 1px solid rgba(0,230,170,0.04);
          cursor: pointer;
          transition: background 0.15s;
        }

        .corpus-row:hover { background: rgba(0,230,170,0.04); }
        .corpus-row.selected { background: rgba(0,230,170,0.07); }

        .corpus-row td {
          padding: 0.65rem 0;
          vertical-align: middle;
        }

        .corpus-id {
          color: #00e6aa;
          font-weight: 500;
        }

        .corpus-ts { color: rgba(200,220,215,0.3); font-size: 0.7rem; }

        .badge {
          display: inline-block;
          padding: 0.2rem 0.55rem;
          border-radius: 3px;
          font-size: 0.62rem;
          letter-spacing: 0.12em;
          font-weight: 600;
        }

        .badge-ready, .badge-done {
          background: rgba(16,255,176,0.1);
          color: #10ffb0;
          border: 1px solid rgba(16,255,176,0.2);
        }

        .badge-running, .badge-pending {
          background: rgba(245,158,11,0.1);
          color: #f59e0b;
          border: 1px solid rgba(245,158,11,0.2);
        }

        .badge-failed {
          background: rgba(255,68,102,0.1);
          color: #ff4466;
          border: 1px solid rgba(255,68,102,0.2);
        }

        /* ── Quick Launch ── */
        .launch-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.75rem;
          margin-bottom: 1.25rem;
        }

        @media (max-width: 600px) { .launch-grid { grid-template-columns: 1fr; } }

        .select-group label {
          display: block;
          font-size: 0.62rem;
          letter-spacing: 0.15em;
          text-transform: uppercase;
          color: rgba(0,230,170,0.45);
          margin-bottom: 0.4rem;
        }

        .hqc-select {
          width: 100%;
          background: #0d1117;
          border: 1px solid rgba(0,230,170,0.15);
          border-radius: 4px;
          color: #c8dcd8;
          padding: 0.6rem 0.75rem;
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.78rem;
          outline: none;
          cursor: pointer;
          transition: border-color 0.18s;
          appearance: none;
          -webkit-appearance: none;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' fill='none'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2300e6aa44' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
          background-repeat: no-repeat;
          background-position: right 0.75rem center;
          padding-right: 2.2rem;
        }

        .hqc-select:focus { border-color: rgba(0,230,170,0.4); }

        /* ── Launch button ── */
        .launch-btn {
          width: 100%;
          padding: 0.85rem;
          background: linear-gradient(135deg, rgba(0,230,170,0.12), rgba(0,180,255,0.08));
          border: 1px solid rgba(0,230,170,0.3);
          border-radius: 4px;
          color: #00e6aa;
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.82rem;
          letter-spacing: 0.15em;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.75rem;
          text-transform: uppercase;
        }

        .launch-btn:hover:not(:disabled) {
          background: linear-gradient(135deg, rgba(0,230,170,0.2), rgba(0,180,255,0.12));
          border-color: rgba(0,230,170,0.5);
          box-shadow: 0 0 20px rgba(0,230,170,0.1);
        }

        .launch-btn:disabled {
          opacity: 0.45;
          cursor: not-allowed;
        }

        .launch-spinner {
          animation: spin 0.8s linear infinite;
          display: inline-block;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .launch-msg {
          margin-top: 0.75rem;
          padding: 0.6rem 0.75rem;
          border-radius: 4px;
          font-size: 0.72rem;
          letter-spacing: 0.06em;
        }

        .launch-msg.ok {
          background: rgba(16,255,176,0.08);
          border: 1px solid rgba(16,255,176,0.2);
          color: #10ffb0;
        }

        .launch-msg.err {
          background: rgba(255,68,102,0.08);
          border: 1px solid rgba(255,68,102,0.2);
          color: #ff4466;
        }

        /* ── Dataset Cards ── */
        .dataset-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }

        .dataset-chip {
          padding: 0.35rem 0.75rem;
          border-radius: 3px;
          border: 1px solid rgba(0,230,170,0.15);
          background: transparent;
          cursor: pointer;
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.7rem;
          color: rgba(200,220,215,0.5);
          letter-spacing: 0.06em;
          transition: all 0.15s;
        }

        .dataset-chip.selected {
          background: rgba(0,230,170,0.1);
          border-color: rgba(0,230,170,0.35);
          color: #00e6aa;
        }

        .dataset-chip:hover { border-color: rgba(0,230,170,0.3); color: #c8dcd8; }

        /* ── Recent Runs ── */
        .run-row {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 0.75rem 0;
          border-bottom: 1px solid rgba(0,230,170,0.05);
        }

        .run-row:last-child { border-bottom: none; }

        .run-id-col {
          display: flex;
          flex-direction: column;
          gap: 0.2rem;
          min-width: 0;
          flex: 1;
        }

        .run-id-mono {
          font-size: 0.72rem;
          color: #00b4ff;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .run-method {
          font-size: 0.62rem;
          color: rgba(200,220,215,0.3);
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }

        .run-metrics {
          display: flex;
          gap: 0.4rem;
          flex-wrap: wrap;
        }

        .metric-pill {
          padding: 0.2rem 0.5rem;
          background: rgba(0,180,255,0.06);
          border: 1px solid rgba(0,180,255,0.12);
          border-radius: 3px;
          font-size: 0.65rem;
          color: rgba(0,180,255,0.7);
          white-space: nowrap;
        }

        .metric-pill b { color: #00b4ff; font-weight: 500; }

        .run-status {
          display: flex;
          align-items: center;
          gap: 0.4rem;
          font-size: 0.65rem;
          letter-spacing: 0.1em;
          flex-shrink: 0;
        }

        .run-dot {
          width: 5px; height: 5px;
          border-radius: 50%;
          flex-shrink: 0;
        }

        /* ── Empty state ── */
        .empty-state {
          text-align: center;
          padding: 2rem;
          color: rgba(200,220,215,0.2);
          font-size: 0.75rem;
          letter-spacing: 0.1em;
        }

        .empty-icon { font-size: 2rem; margin-bottom: 0.5rem; opacity: 0.3; }

        /* ── Pipeline Stage Viz ── */
        .stage-flow {
          display: flex;
          gap: 0;
          overflow-x: auto;
          padding-bottom: 0.5rem;
        }

        .stage-item {
          display: flex;
          align-items: center;
          gap: 0;
          flex-shrink: 0;
        }

        .stage-box {
          background: #0d1117;
          border: 1px solid rgba(0,230,170,0.1);
          border-radius: 4px;
          padding: 0.5rem 0.75rem;
          text-align: center;
          min-width: 90px;
        }

        .stage-num {
          font-size: 0.6rem;
          color: rgba(0,230,170,0.35);
          letter-spacing: 0.12em;
        }

        .stage-name {
          font-size: 0.68rem;
          color: rgba(200,220,215,0.6);
          margin-top: 0.2rem;
          white-space: nowrap;
        }

        .stage-arrow {
          color: rgba(0,230,170,0.2);
          padding: 0 0.3rem;
          font-size: 0.8rem;
        }

        .nav-cta {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1.25rem;
          background: rgba(0,230,170,0.08);
          border: 1px solid rgba(0,230,170,0.25);
          border-radius: 4px;
          color: #00e6aa;
          font-size: 0.75rem;
          letter-spacing: 0.12em;
          text-decoration: none;
          text-transform: uppercase;
          font-family: 'Share Tech Mono', monospace;
          transition: all 0.18s;
        }

        .nav-cta:hover {
          background: rgba(0,230,170,0.14);
          border-color: rgba(0,230,170,0.4);
        }
      `}</style>

      <div className="dashboard">
        <div className="dash-content">

          {/* ── Page Header ── */}
          <div className="dash-header">
            <div>
              <div className="dash-title">
                Research <span>Console</span>
              </div>
              <div className="dash-subtitle">
                HQC Topic Modeling · Blueprint v1.0 · Master Dashboard
              </div>
            </div>
            <div className="header-actions">
              <label className="upload-label">
                ⬆ Upload Corpus
                <input
                  type="file"
                  accept=".txt,.csv,.jsonl,.json"
                  onChange={handleFileUpload}
                />
              </label>
              <Link to="/pipeline" className="nav-cta">◈ New Pipeline →</Link>
            </div>
          </div>

          {/* ── Upload Progress ── */}
          {uploadProgress !== null && (
            <div className="upload-bar-wrap">
              <div className="upload-bar" style={{ width: `${uploadProgress}%` }} />
            </div>
          )}

          {/* ── Stats Grid ── */}
          <div className="stats-grid">
            <StatCard
              label="Documents Loaded"
              value={totalDocs.toLocaleString()}
              icon="◫"
              accent="#00e6aa"
              sublabel="Across all corpora"
              loading={loading}
            />
            <StatCard
              label="Pipeline Runs"
              value={recentRuns.length}
              icon="◈"
              accent="#00b4ff"
              sublabel={`${doneRuns} completed`}
              loading={loadingRuns}
            />
            <StatCard
              label="Avg Silhouette"
              value={doneRuns > 0 ? avgSil.toFixed(3) : "—"}
              icon="◉"
              accent="#f59e0b"
              sublabel="Hybrid clustering quality"
              loading={loadingRuns}
            />
            <StatCard
              label="API Status"
              value={systemStatus ? "Online" : "—"}
              icon="⬡"
              accent={systemStatus ? "#10ffb0" : "#ff4466"}
              sublabel="FastAPI · Port 8000"
              loading={loading}
            />
          </div>

          {/* ── Pipeline Stage Flow ── */}
          <div className="panel" style={{ marginBottom: "1.5rem" }}>
            <div className="panel-header">
              <span className="panel-title">
                <span className="panel-title-icon">▶</span>
                End-to-End Pipeline (§5.1 — 8 Stages)
              </span>
              <Link to="/pipeline" className="panel-action">Launch →</Link>
            </div>
            <div className="panel-body">
              <div className="stage-flow">
                {[
                  ["01", "Ingest"],
                  ["02", "Preprocess"],
                  ["03", "Tokenize"],
                  ["04", "Classical"],
                  ["05", "Hamiltonian"],
                  ["06", "QAOA/VQE"],
                  ["07", "Hybrid Cluster"],
                  ["08", "Evaluate"],
                ].map(([num, name], i, arr) => (
                  <div className="stage-item" key={num}>
                    <div className="stage-box">
                      <div className="stage-num">STAGE {num}</div>
                      <div className="stage-name">{name}</div>
                    </div>
                    {i < arr.length - 1 && (
                      <span className="stage-arrow">→</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── Main Grid ── */}
          <div className="main-grid">

            {/* ── Quick Launch Panel ── */}
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">
                  <span className="panel-title-icon">⚡</span>
                  Quick Launch (§7 Datasets)
                </span>
              </div>
              <div className="panel-body">
                <div style={{ marginBottom: "0.75rem" }}>
                  <div style={{ fontSize: "0.62rem", letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(0,230,170,0.45)", marginBottom: "0.5rem" }}>
                    Dataset
                  </div>
                  <div className="dataset-chips">
                    {DATASETS.map((d) => (
                      <button
                        key={d.id}
                        className={`dataset-chip ${selectedDataset === d.id ? "selected" : ""}`}
                        onClick={() => setSelectedDataset(d.id)}
                      >
                        {d.label}
                        <span style={{ opacity: 0.5, marginLeft: "0.4rem" }}>
                          ({d.docs.toLocaleString()})
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="launch-grid">
                  <div className="select-group">
                    <label>Quantum Subset</label>
                    <select
                      className="hqc-select"
                      value={selectedSubset}
                      onChange={(e) => setSelectedSubset(Number(e.target.value))}
                    >
                      {SUBSETS.map((s) => (
                        <option key={s.size} value={s.size}>
                          {s.label} — {s.desc}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="select-group">
                    <label>Pipeline Mode</label>
                    <select className="hqc-select" defaultValue="hybrid">
                      <option value="hybrid">Hybrid (QAOA + Classical)</option>
                      <option value="classical">Classical Baseline Only</option>
                      <option value="quantum">Quantum Only (QAOA)</option>
                    </select>
                  </div>
                </div>

                <button
                  className="launch-btn"
                  onClick={handleQuickLaunch}
                  disabled={launching}
                >
                  {launching ? (
                    <><span className="launch-spinner">◈</span> Initialising Pipeline…</>
                  ) : (
                    <>▶ Launch Hybrid Pipeline</>
                  )}
                </button>

                {launchMsg && (
                  <div className={`launch-msg ${launchMsg.type}`}>
                    {launchMsg.type === "ok" ? "✓" : "✗"} {launchMsg.text}
                  </div>
                )}
              </div>
            </div>

            {/* ── Recent Runs Panel ── */}
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">
                  <span className="panel-title-icon">◉</span>
                  Recent Pipeline Runs
                </span>
                <Link to="/results" className="panel-action">View all →</Link>
              </div>
              <div className="panel-body">
                {loadingRuns ? (
                  <div className="empty-state">
                    <div className="launch-spinner" style={{ fontSize: "1.5rem", display: "block", marginBottom: "0.5rem" }}>◈</div>
                    Loading runs…
                  </div>
                ) : recentRuns.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon">◉</div>
                    No runs yet — launch your first pipeline above.
                  </div>
                ) : (
                  recentRuns.map((run, i) => (
                    <RecentRunRow key={run.run_id ?? run.job_id ?? i} run={run} />
                  ))
                )}
              </div>
            </div>

          </div>

          {/* ── Corpus Table ── */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">
                <span className="panel-title-icon">◫</span>
                Loaded Corpus (§7.1 — Blueprint Datasets)
              </span>
              <Link to="/pipeline" className="panel-action">Manage →</Link>
            </div>
            <div className="panel-body">
              {loading ? (
                <div className="empty-state">Fetching documents…</div>
              ) : documents.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">◫</div>
                  No documents loaded. Upload a corpus or use Quick Launch above.
                </div>
              ) : (
                <table className="corpus-table">
                  <thead>
                    <tr>
                      <th>Corpus ID</th>
                      <th>Doc Count</th>
                      <th>Vocab Size</th>
                      <th>Status</th>
                      <th>Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Aggregate by source for display */}
                    <CorpusRow
                      key="all"
                      corpus={{
                        corpus_id: "all_documents",
                        doc_count: totalDocs,
                        status: "ready",
                        created_at: documents[0]?.created_at,
                      }}
                      onSelect={() => {}}
                      selected={false}
                    />
                    {DATASETS.filter((d) =>
                      documents.some((doc) =>
                        doc.source === d.id || doc.category
                      )
                    ).map((d) => (
                      <CorpusRow
                        key={d.id}
                        corpus={{ corpus_id: d.id, doc_count: d.docs, classes: d.classes, status: "ready" }}
                        onSelect={() => {}}
                        selected={false}
                      />
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
