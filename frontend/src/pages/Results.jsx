/**
 * Results Page — Pipeline Run History, Metrics, Topic Viewer, Export
 * §4.4 Hybrid endpoints · §4.5 Evaluation endpoints
 * Integrates MetricsPanel and TopicViewer sub-components.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { Link } from "react-router-dom";
import { evalApi } from "../services/api";
import MetricsPanel from "../components/Results/MetricsPanel";
import TopicViewer  from "../components/Results/TopicViewer";

// ─── Constants ────────────────────────────────────────────────────────────────
const STATUS_COLORS = {
  done:    "#10ffb0",
  failed:  "#ff4466",
  running: "#f59e0b",
  pending: "#6b7280",
};

const METHOD_LABELS = {
  hybrid:    "Hybrid",
  classical: "Classical",
  quantum:   "Quantum",
};

const PAGE_SIZE = 20;

// ─── StatusDot ────────────────────────────────────────────────────────────────
function StatusDot({ status }) {
  const color = STATUS_COLORS[status] ?? "#6b7280";
  return (
    <span style={{
      display: "inline-block",
      width: 6, height: 6, borderRadius: "50%",
      background: color, flexShrink: 0,
      boxShadow: status === "running" ? `0 0 0 3px ${color}33` : "none",
      animation: status === "running" ? "rPulse 1.5s ease-in-out infinite" : "none",
    }} />
  );
}

// ─── Inline metric chip ───────────────────────────────────────────────────────
function MetricChip({ label, value, color = "#00e6aa" }) {
  if (value == null) return null;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: "0.3rem",
      padding: "0.18rem 0.5rem",
      background: `${color}0d`, border: `1px solid ${color}28`,
      borderRadius: 3, fontSize: "0.64rem", letterSpacing: "0.06em",
      color: `${color}bb`, whiteSpace: "nowrap", fontFamily: "var(--font-mono)",
    }}>
      {label} <strong style={{ color }}>{typeof value === "number" ? value.toFixed(3) : value}</strong>
    </span>
  );
}

// ─── Run list item ────────────────────────────────────────────────────────────
function RunItem({ run, isSelected, onClick }) {
  const id     = run.run_id ?? run.job_id ?? "—";
  const status = run.status ?? "pending";
  const color  = STATUS_COLORS[status] ?? "#6b7280";
  const ts     = run.created_at
    ? new Date(run.created_at).toLocaleString(undefined, {
        month: "short", day: "numeric",
        hour: "2-digit", minute: "2-digit",
      })
    : null;

  return (
    <div
      onClick={onClick}
      style={{
        padding: "0.85rem 1.5rem",
        borderBottom: "1px solid rgba(0,230,170,0.04)",
        borderLeft: isSelected ? "2px solid #00e6aa" : "2px solid transparent",
        cursor: "pointer",
        background: isSelected ? "rgba(0,230,170,0.05)" : "transparent",
        transition: "all 0.13s",
        display: "flex", flexDirection: "column", gap: "0.4rem",
      }}
      onMouseEnter={(e) => { if (!isSelected) e.currentTarget.style.background = "rgba(0,230,170,0.025)"; }}
      onMouseLeave={(e) => { if (!isSelected) e.currentTarget.style.background = "transparent"; }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.55rem" }}>
        <StatusDot status={status} />
        <span style={{
          fontSize: "0.72rem", color: "#00b4ff",
          fontFamily: "var(--font-mono)", flex: 1,
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        }}>
          {id}
        </span>
        <span style={{ fontSize: "0.6rem", letterSpacing: "0.1em", textTransform: "uppercase", color, flexShrink: 0 }}>
          {status}
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", flexWrap: "wrap" }}>
        <span style={{ fontSize: "0.6rem", color: "rgba(200,220,215,0.22)", letterSpacing: "0.12em", textTransform: "uppercase", marginRight: "0.15rem" }}>
          {METHOD_LABELS[run.method] ?? run.method ?? "hybrid"}
        </span>
        <MetricChip label="sil"  value={run.silhouette_hybrid}  color="#00e6aa" />
        <MetricChip label="Cᵥ"   value={run.topic_coherence_cv} color="#00b4ff" />
        <MetricChip label="cost" value={run.qaoa_final_cost}     color="#f59e0b" />
      </div>
      {ts && (
        <span style={{ fontSize: "0.59rem", color: "rgba(200,220,215,0.18)", letterSpacing: "0.06em" }}>
          {ts}
        </span>
      )}
    </div>
  );
}

// ─── Detail header ────────────────────────────────────────────────────────────
function DetailHeader({ run, onExportJson, onExportCsv, exporting }) {
  const id     = run.run_id ?? run.job_id ?? "—";
  const status = run.status ?? "pending";
  const color  = STATUS_COLORS[status] ?? "#6b7280";

  const metaFields = [
    ["Method",   METHOD_LABELS[run.method] ?? run.method ?? "hybrid"],
    ["Topics",   run.num_topics  ?? "—"],
    ["Clusters", run.num_clusters ?? "—"],
    ["QAOA p",   run.qaoa_layers  ?? "—"],
    ["Noise",    run.enable_noise      ? "ON" : "OFF"],
    ["ZNE",      run.enable_mitigation ? "ON" : "OFF"],
  ];

  return (
    <div style={{ padding: "1.25rem 1.5rem", borderBottom: "1px solid rgba(0,230,170,0.07)", flexShrink: 0 }}>
      {/* Job ID row */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.75rem", marginBottom: "0.85rem" }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: "0.58rem", letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(0,230,170,0.38)", marginBottom: "0.2rem" }}>
            Job ID
          </div>
          <div style={{ fontSize: "0.78rem", color: "#00b4ff", fontFamily: "var(--font-mono)", wordBreak: "break-all" }}>
            {id}
          </div>
        </div>
        <span style={{
          display: "inline-flex", alignItems: "center", gap: "0.4rem",
          padding: "0.28rem 0.7rem",
          border: `1px solid ${color}2e`, background: `${color}0b`,
          borderRadius: 3, fontSize: "0.63rem", letterSpacing: "0.12em",
          textTransform: "uppercase", color, flexShrink: 0,
        }}>
          <StatusDot status={status} />
          {status}
        </span>
      </div>

      {/* Meta grid */}
      <div style={{ display: "flex", gap: "1.25rem", flexWrap: "wrap", marginBottom: "0.85rem" }}>
        {metaFields.map(([k, v]) => (
          <div key={k}>
            <div style={{ fontSize: "0.57rem", letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(0,230,170,0.32)", marginBottom: "0.1rem" }}>{k}</div>
            <div style={{ fontSize: "0.72rem", color: "#c8dcd8", fontFamily: "var(--font-mono)" }}>{String(v)}</div>
          </div>
        ))}
      </div>

      {/* Export row */}
      <div style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap" }}>
        {[
          { label: "↓ JSON", color: "#00b4ff", border: "rgba(0,180,255,0.18)", onClick: onExportJson },
          { label: "↓ CSV",  color: "#00e6aa", border: "rgba(0,230,170,0.15)", onClick: onExportCsv },
        ].map(({ label, color: c, border, onClick }) => (
          <button
            key={label}
            onClick={onClick}
            disabled={exporting}
            style={{
              display: "inline-flex", alignItems: "center", gap: "0.35rem",
              padding: "0.35rem 0.8rem",
              background: "transparent", border: `1px solid ${border}`,
              borderRadius: 3, cursor: "pointer",
              color: `${c}88`, fontSize: "0.67rem",
              letterSpacing: "0.1em", fontFamily: "var(--font-mono)",
              textTransform: "uppercase", transition: "all 0.14s",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = c; e.currentTarget.style.borderColor = `${c}44`; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = `${c}88`; e.currentTarget.style.borderColor = border; }}
          >
            {label}
          </button>
        ))}
        <Link
          to="/pipeline"
          style={{
            display: "inline-flex", alignItems: "center", gap: "0.35rem",
            padding: "0.35rem 0.8rem",
            background: "transparent", border: "1px solid rgba(245,158,11,0.15)",
            borderRadius: 3, color: "rgba(245,158,11,0.55)",
            fontSize: "0.67rem", letterSpacing: "0.1em",
            fontFamily: "var(--font-mono)", textTransform: "uppercase",
            transition: "all 0.14s",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.color = "#f59e0b"; e.currentTarget.style.borderColor = "rgba(245,158,11,0.35)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = "rgba(245,158,11,0.55)"; e.currentTarget.style.borderColor = "rgba(245,158,11,0.15)"; }}
        >
          ◈ Rerun
        </Link>
      </div>
    </div>
  );
}

// ─── Best-method banner ────────────────────────────────────────────────────────
function BestMethodBanner({ result }) {
  if (!result) return null;
  const methods = ["classical", "quantum", "hybrid"];
  const scores  = methods.map((m) => ({ m, v: result[`silhouette_${m}`] ?? -Infinity }));
  const best    = scores.reduce((a, b) => (b.v > a.v ? b : a));
  if (best.v === -Infinity) return null;

  return (
    <div style={{
      margin: "0.85rem 1.5rem 0",
      padding: "0.65rem 1rem",
      background: "rgba(0,230,170,0.035)",
      border: "1px solid rgba(0,230,170,0.09)",
      borderRadius: 4, display: "flex", alignItems: "center", gap: "0.7rem",
      fontSize: "0.68rem", color: "rgba(200,220,215,0.45)", flexShrink: 0,
    }}>
      <span style={{ color: "rgba(0,230,170,0.5)" }}>⟨ψ⟩</span>
      <span>
        Best clustering:{" "}
        <strong style={{ color: "#00e6aa" }}>{best.m.toUpperCase()}</strong>
        {" "}(silhouette ={" "}
        <strong style={{ color: "#00e6aa" }}>{best.v.toFixed(4)}</strong>)
      </span>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function Results() {
  const [runs,          setRuns]          = useState([]);
  const [loading,       setLoading]       = useState(true);
  const [selected,      setSelected]      = useState(null);
  const [detailResult,  setDetailResult]  = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [exporting,     setExporting]     = useState(false);
  const [exportMsg,     setExportMsg]     = useState(null);
  const [searchQuery,   setSearchQuery]   = useState("");
  const [statusFilter,  setStatusFilter]  = useState("all");
  const [activePanel,   setActivePanel]   = useState("metrics"); // metrics | topics
  const [page,          setPage]          = useState(0);
  const detailRef = useRef(null);

  // ── Fetch run list ────────────────────────────────────────────────────────────
  const fetchRuns = useCallback(async (p = 0) => {
    setLoading(true);
    try {
      const data = await evalApi.getAllResults(p * PAGE_SIZE, PAGE_SIZE);
      setRuns(data?.results ?? data ?? []);
    } catch (err) {
      console.error("fetchRuns:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchRuns(page); }, [fetchRuns, page]);

  // Auto-refresh while runs are in progress
  useEffect(() => {
    const hasActive = runs.some((r) => r.status === "running" || r.status === "pending");
    if (!hasActive) return;
    const id = setInterval(() => fetchRuns(page), 12_000);
    return () => clearInterval(id);
  }, [runs, fetchRuns, page]);

  // ── Select run + fetch detail ─────────────────────────────────────────────────
  const handleSelect = useCallback(async (run) => {
    setSelected(run);
    setDetailResult(null);
    setLoadingDetail(true);
    setActivePanel("metrics");
    detailRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    try {
      const id   = run.run_id ?? run.job_id;
      const data = await evalApi.getResultByJobId(id);
      setDetailResult(data ?? run);
    } catch {
      setDetailResult(run);
    } finally {
      setLoadingDetail(false);
    }
  }, []);

  // ── Export ────────────────────────────────────────────────────────────────────
  const handleExport = useCallback(async (format) => {
    if (!selected) return;
    setExporting(true);
    setExportMsg(null);
    try {
      const id   = selected.run_id ?? selected.job_id;
      const blob = await evalApi.exportResult(id, format);
      const url  = URL.createObjectURL(new Blob([blob]));
      const a    = document.createElement("a");
      a.href = url; a.download = `hqc_${id}.${format}`;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
      setExportMsg({ type: "ok", text: `${format.toUpperCase()} exported.` });
    } catch (err) {
      setExportMsg({ type: "err", text: err.message ?? "Export failed." });
    } finally {
      setExporting(false);
      setTimeout(() => setExportMsg(null), 3500);
    }
  }, [selected]);

  // ── Filter ────────────────────────────────────────────────────────────────────
  const filteredRuns = runs.filter((r) => {
    const q     = searchQuery.toLowerCase();
    const id    = (r.run_id ?? r.job_id ?? "").toLowerCase();
    const meth  = (r.method ?? "").toLowerCase();
    const match = !q || id.includes(q) || meth.includes(q);
    const byStatus = statusFilter === "all" || r.status === statusFilter;
    return match && byStatus;
  });

  const topics = detailResult?.topics ?? [];

  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <>
      <style>{`
        @keyframes rPulse {
          0%,100% { opacity:1; }
          50%      { opacity:0.35; }
        }

        .rp-page {
          min-height: calc(100vh - 60px);
          background: #06080e;
          padding-top: 60px;
          font-family: 'Share Tech Mono', monospace;
          color: #c8dcd8;
          position: relative;
        }

        .rp-page::before {
          content:'';
          position: fixed; inset: 0;
          background-image:
            linear-gradient(rgba(0,230,170,0.01) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,230,170,0.01) 1px, transparent 1px);
          background-size: 40px 40px;
          pointer-events: none; z-index: 0;
        }

        .rp-inner {
          position: relative; z-index: 1;
          max-width: 1400px; margin: 0 auto;
          padding: 2rem 1.5rem 3rem;
        }

        .rp-layout {
          display: grid;
          grid-template-columns: 310px 1fr;
          gap: 1.5rem;
          min-height: calc(100vh - 200px);
        }

        @media (max-width: 960px) { .rp-layout { grid-template-columns: 1fr; } }

        /* Left panel */
        .rp-list {
          background: #080b12;
          border: 1px solid rgba(0,230,170,0.1);
          border-radius: 6px;
          overflow: hidden;
          display: flex; flex-direction: column;
          max-height: calc(100vh - 200px);
        }

        .rp-list-head {
          padding: 0.9rem 1.5rem;
          border-bottom: 1px solid rgba(0,230,170,0.07);
          flex-shrink: 0;
          display: flex; align-items: center; justify-content: space-between;
        }

        .rp-filters {
          padding: 0.65rem 1rem;
          border-bottom: 1px solid rgba(0,230,170,0.05);
          display: flex; gap: 0.4rem; flex-wrap: wrap;
          flex-shrink: 0;
        }

        .rp-search {
          flex: 1; min-width: 110px;
          background: #0d1117;
          border: 1px solid rgba(0,230,170,0.1);
          border-radius: 4px;
          color: #c8dcd8;
          padding: 0.38rem 0.65rem;
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.71rem; outline: none;
          transition: border-color 0.15s;
        }
        .rp-search:focus { border-color: rgba(0,230,170,0.28); }
        .rp-search::placeholder { color: rgba(200,220,215,0.18); }

        .rp-chip {
          padding: 0.28rem 0.6rem;
          border-radius: 3px; border: 1px solid rgba(0,230,170,0.1);
          background: transparent;
          color: rgba(200,220,215,0.3);
          font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase;
          cursor: pointer; font-family: 'Share Tech Mono', monospace;
          transition: all 0.13s; white-space: nowrap;
        }
        .rp-chip:hover { border-color: rgba(0,230,170,0.22); color: rgba(200,220,215,0.55); }
        .rp-chip.on {
          background: rgba(0,230,170,0.07);
          border-color: rgba(0,230,170,0.26);
          color: #00e6aa;
        }

        .rp-scroll { overflow-y: auto; flex: 1; }

        .rp-pagination {
          display: flex; align-items: center; justify-content: space-between;
          padding: 0.65rem 1.5rem;
          border-top: 1px solid rgba(0,230,170,0.06);
          flex-shrink: 0;
        }

        .rp-pgbtn {
          padding: 0.28rem 0.7rem;
          background: transparent; border: 1px solid rgba(0,230,170,0.12);
          border-radius: 3px; color: rgba(0,230,170,0.4);
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.66rem; cursor: pointer; transition: all 0.13s;
        }
        .rp-pgbtn:hover:not(:disabled) { border-color: rgba(0,230,170,0.3); color: #00e6aa; }
        .rp-pgbtn:disabled { opacity: 0.3; cursor: not-allowed; }

        /* Right panel */
        .rp-detail {
          background: #080b12;
          border: 1px solid rgba(0,230,170,0.1);
          border-radius: 6px;
          overflow: hidden;
          display: flex; flex-direction: column;
          max-height: calc(100vh - 200px);
        }

        .rp-detail-scroll { overflow-y: auto; flex: 1; }

        .rp-tabs {
          display: flex;
          border-bottom: 1px solid rgba(0,230,170,0.07);
          flex-shrink: 0;
        }

        .rp-tab {
          padding: 0.7rem 1.15rem;
          font-size: 0.66rem; letter-spacing: 0.14em;
          text-transform: uppercase;
          color: rgba(200,220,215,0.3);
          cursor: pointer;
          border-bottom: 2px solid transparent;
          transition: all 0.14s;
          background: none; border-top: none;
          border-left: none; border-right: none;
          font-family: 'Share Tech Mono', monospace;
          white-space: nowrap;
        }
        .rp-tab:hover { color: rgba(200,220,215,0.6); }
        .rp-tab.on { color: #00e6aa; border-bottom: 2px solid #00e6aa; }

        .rp-tab-body { padding: 1.5rem; }

        /* Empty states */
        .rp-empty {
          text-align: center; padding: 3rem 1.5rem;
          color: rgba(200,220,215,0.12);
          font-size: 0.73rem; letter-spacing: 0.1em;
        }

        .rp-empty-detail {
          display: flex; flex-direction: column;
          align-items: center; justify-content: center;
          height: 100%; min-height: 300px;
          color: rgba(200,220,215,0.12);
          font-size: 0.73rem; letter-spacing: 0.1em; gap: 0.75rem;
        }

        /* Toast */
        .rp-toast {
          position: fixed; bottom: 2rem; right: 2rem;
          padding: 0.6rem 1rem; border-radius: 4px;
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.7rem; letter-spacing: 0.06em;
          z-index: 600; animation: appFadeIn 0.2s ease;
        }
      `}</style>

      <div className="rp-page">
        <div className="rp-inner">

          {/* ── Page header ── */}
          <div style={{ marginBottom: "1.5rem", display: "flex", alignItems: "flex-end", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem" }}>
            <div>
              <div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: "clamp(1.4rem,3vw,2rem)", color: "#e8f4f0" }}>
                Pipeline <span style={{ color: "#00e6aa", textShadow: "0 0 20px rgba(0,230,170,0.35)" }}>Results</span>
              </div>
              <div style={{ fontSize: "0.68rem", color: "rgba(0,230,170,0.4)", letterSpacing: "0.18em", textTransform: "uppercase", marginTop: "0.3rem" }}>
                §4.5 Evaluation · §5.1 Stage 08 · Benchmark Reports
              </div>
            </div>
            <div style={{ display: "flex", gap: "0.65rem" }}>
              <button
                onClick={() => fetchRuns(page)}
                style={{
                  display: "inline-flex", alignItems: "center", gap: "0.4rem",
                  padding: "0.43rem 1rem", background: "transparent",
                  border: "1px solid rgba(0,230,170,0.14)", borderRadius: 4,
                  color: "rgba(0,230,170,0.48)", cursor: "pointer",
                  fontFamily: "var(--font-mono)", fontSize: "0.7rem",
                  letterSpacing: "0.1em", textTransform: "uppercase", transition: "all 0.14s",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = "rgba(0,230,170,0.32)"; e.currentTarget.style.color = "#00e6aa"; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = "rgba(0,230,170,0.14)"; e.currentTarget.style.color = "rgba(0,230,170,0.48)"; }}
              >
                ↺ Refresh
              </button>
              <Link
                to="/pipeline"
                style={{
                  display: "inline-flex", alignItems: "center", gap: "0.4rem",
                  padding: "0.43rem 1rem",
                  background: "rgba(0,230,170,0.07)", border: "1px solid rgba(0,230,170,0.22)",
                  borderRadius: 4, color: "#00e6aa",
                  fontFamily: "var(--font-mono)", fontSize: "0.7rem",
                  letterSpacing: "0.1em", textTransform: "uppercase",
                }}
              >
                ◈ New Run →
              </Link>
            </div>
          </div>

          {/* ── Layout ── */}
          <div className="rp-layout">

            {/* ══ Left: Run list ══ */}
            <div className="rp-list">
              <div className="rp-list-head">
                <span style={{ fontSize: "0.68rem", letterSpacing: "0.18em", textTransform: "uppercase", color: "rgba(0,230,170,0.55)" }}>
                  ◉ Run History
                </span>
                <span style={{ fontSize: "0.65rem", color: "rgba(200,220,215,0.2)" }}>
                  {loading ? "…" : `${filteredRuns.length}/${runs.length}`}
                </span>
              </div>

              <div className="rp-filters">
                <input
                  className="rp-search"
                  placeholder="Search job ID…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                {["all","done","running","failed"].map((s) => (
                  <button
                    key={s}
                    className={`rp-chip ${statusFilter === s ? "on" : ""}`}
                    onClick={() => setStatusFilter(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>

              <div className="rp-scroll">
                {loading ? (
                  <div className="rp-empty">
                    <div style={{ fontSize: "1.5rem", opacity: 0.25, marginBottom: "0.5rem", animation: "appSpin 1s linear infinite", display: "inline-block" }}>◈</div>
                    <div>Loading runs…</div>
                  </div>
                ) : filteredRuns.length === 0 ? (
                  <div className="rp-empty">
                    <div style={{ fontSize: "2rem", opacity: 0.1, marginBottom: "0.5rem" }}>◉</div>
                    <div>{searchQuery || statusFilter !== "all" ? "No runs match filters." : "No runs yet."}</div>
                  </div>
                ) : filteredRuns.map((run, i) => (
                  <RunItem
                    key={run.run_id ?? run.job_id ?? i}
                    run={run}
                    isSelected={
                      selected &&
                      (selected.run_id ?? selected.job_id) === (run.run_id ?? run.job_id)
                    }
                    onClick={() => handleSelect(run)}
                  />
                ))}
              </div>

              {runs.length >= PAGE_SIZE && (
                <div className="rp-pagination">
                  <button className="rp-pgbtn" onClick={() => setPage((p) => Math.max(0, p-1))} disabled={page === 0}>← Prev</button>
                  <span style={{ fontSize: "0.63rem", color: "rgba(200,220,215,0.2)", letterSpacing: "0.1em" }}>pg {page+1}</span>
                  <button className="rp-pgbtn" onClick={() => setPage((p) => p+1)} disabled={runs.length < PAGE_SIZE}>Next →</button>
                </div>
              )}
            </div>

            {/* ══ Right: Detail ══ */}
            <div className="rp-detail">
              {!selected ? (
                <div className="rp-empty-detail">
                  <div style={{ fontSize: "3rem", opacity: 0.07 }}>◉</div>
                  <div>Select a run to view results</div>
                </div>
              ) : (
                <>
                  <DetailHeader
                    run={selected}
                    onExportJson={() => handleExport("json")}
                    onExportCsv={() => handleExport("csv")}
                    exporting={exporting}
                  />

                  <BestMethodBanner result={detailResult} />

                  <div className="rp-tabs">
                    {[
                      { key: "metrics", label: "◉ Metrics" },
                      { key: "topics",  label: "◫ Topics"  },
                    ].map(({ key, label }) => (
                      <button
                        key={key}
                        className={`rp-tab ${activePanel === key ? "on" : ""}`}
                        onClick={() => setActivePanel(key)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  <div className="rp-detail-scroll" ref={detailRef}>
                    {loadingDetail ? (
                      <div style={{ textAlign: "center", padding: "3rem", color: "rgba(200,220,215,0.15)", fontFamily: "var(--font-mono)", fontSize: "0.73rem" }}>
                        <div style={{ animation: "appSpin 1s linear infinite", display: "inline-block", fontSize: "1.5rem", marginBottom: "0.5rem" }}>◈</div>
                        <div>Fetching run detail…</div>
                      </div>
                    ) : (
                      <div className="rp-tab-body">
                        {activePanel === "metrics" && <MetricsPanel result={detailResult} />}
                        {activePanel === "topics"  && (
                          <TopicViewer
                            topics={topics}
                            modelType={detailResult?.topic_model ?? "LDA"}
                            numTopics={detailResult?.num_topics}
                            loading={false}
                          />
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>

          </div>
        </div>
      </div>

      {/* ── Export toast ── */}
      {exportMsg && (
        <div
          className="rp-toast"
          style={{
            background: exportMsg.type === "ok" ? "rgba(16,255,176,0.07)" : "rgba(255,68,102,0.07)",
            border: `1px solid ${exportMsg.type === "ok" ? "rgba(16,255,176,0.2)" : "rgba(255,68,102,0.2)"}`,
            color: exportMsg.type === "ok" ? "#10ffb0" : "#ff4466",
          }}
        >
          {exportMsg.type === "ok" ? "✓" : "✗"} {exportMsg.text}
        </div>
      )}
    </>
  );
}