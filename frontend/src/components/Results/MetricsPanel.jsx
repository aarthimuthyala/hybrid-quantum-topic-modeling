/**
 * MetricsPanel — Evaluation Metrics Comparison Component
 * Renders clustering, topic, and quantum metrics across classical / quantum / hybrid methods.
 * Uses Recharts for convergence chart and custom CSS bars for metric gauges.
 * Integrates with §4.5 Evaluation endpoints.
 */

import { useState, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";

// ─── Colour map per method ────────────────────────────────────────────────────
const METHOD_COLORS = {
  classical: "#f59e0b",
  quantum:   "#00b4ff",
  hybrid:    "#00e6aa",
};

// ─── Metric metadata: label, description, range, higher-is-better ────────────
const METRIC_META = {
  silhouette_score:       { label: "Silhouette",        unit: "",    min: -1,   max: 1,    hib: true,  desc: "Cluster cohesion vs separation. Range [-1, 1]." },
  davies_bouldin_index:   { label: "Davies-Bouldin",    unit: "",    min: 0,    max: 5,    hib: false, desc: "Avg similarity of each cluster to its most similar cluster. Lower is better." },
  calinski_harabasz_score:{ label: "Calinski-Harabasz", unit: "",    min: 0,    max: null, hib: true,  desc: "Ratio of between-cluster to within-cluster variance." },
  nmi:                    { label: "NMI",               unit: "",    min: 0,    max: 1,    hib: true,  desc: "Normalized Mutual Information vs ground-truth labels." },
  ari:                    { label: "ARI",               unit: "",    min: -1,   max: 1,    hib: true,  desc: "Adjusted Rand Index. Corrects for chance." },
  coherence_cv:           { label: "Coherence Cᵥ",     unit: "",    min: 0,    max: 1,    hib: true,  desc: "Topic coherence — sliding window PMI. Higher is more coherent." },
  coherence_umass:        { label: "UMass Coherence",   unit: "",    min: null, max: 0,    hib: true,  desc: "UMass coherence score. Closer to 0 is better." },
  perplexity:             { label: "Perplexity",        unit: "",    min: 0,    max: null, hib: false, desc: "Topic model perplexity on held-out data. Lower is better." },
};

// ─── Gauge Bar ────────────────────────────────────────────────────────────────
function GaugeBar({ value, min = 0, max = 1, color = "#00e6aa", hib = true }) {
  if (value == null) return <span style={{ color: "rgba(200,220,215,0.2)", fontSize: "0.7rem" }}>—</span>;
  const pct = Math.min(100, Math.max(0, ((value - min) / ((max || 1) - min)) * 100));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flex: 1 }}>
      <div style={{
        flex: 1, height: 5,
        background: "rgba(255,255,255,0.05)",
        borderRadius: 3,
        overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`,
          height: "100%",
          background: color,
          borderRadius: 3,
          transition: "width 0.6s ease",
          boxShadow: `0 0 6px ${color}66`,
        }} />
      </div>
      <span style={{
        fontSize: "0.72rem",
        color,
        fontFamily: "var(--font-mono)",
        minWidth: 52,
        textAlign: "right",
        letterSpacing: "0.04em",
      }}>
        {typeof value === "number" ? value.toFixed(4) : value}
      </span>
    </div>
  );
}

// ─── Custom Recharts Tooltip ──────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0d1117",
      border: "1px solid rgba(0,230,170,0.15)",
      borderRadius: 4,
      padding: "0.6rem 0.85rem",
      fontFamily: "var(--font-mono)",
      fontSize: "0.7rem",
    }}>
      <div style={{ color: "rgba(200,220,215,0.45)", marginBottom: "0.3rem", letterSpacing: "0.1em" }}>
        ITER {label}
      </div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color, display: "flex", gap: "0.75rem", justifyContent: "space-between" }}>
          <span style={{ textTransform: "uppercase", letterSpacing: "0.08em" }}>{p.name}</span>
          <span style={{ fontWeight: 600 }}>{typeof p.value === "number" ? p.value.toFixed(5) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Metric Row ───────────────────────────────────────────────────────────────
function MetricRow({ metricKey, classicalVal, quantumVal, hybridVal }) {
  const meta = METRIC_META[metricKey] ?? { label: metricKey, min: 0, max: 1, hib: true, desc: "" };
  const vals = { classical: classicalVal, quantum: quantumVal, hybrid: hybridVal };

  // Determine best
  const defined = Object.entries(vals).filter(([, v]) => v != null);
  const best = defined.length
    ? (meta.hib
        ? defined.reduce((a, b) => (b[1] > a[1] ? b : a))[0]
        : defined.reduce((a, b) => (b[1] < a[1] ? b : a))[0])
    : null;

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "130px 1fr 1fr 1fr",
      gap: "0.75rem",
      padding: "0.6rem 0",
      borderBottom: "1px solid rgba(0,230,170,0.04)",
      alignItems: "center",
    }}>
      <div data-tooltip={meta.desc} style={{ cursor: "default" }}>
        <div style={{ fontSize: "0.7rem", color: "rgba(200,220,215,0.7)", letterSpacing: "0.04em" }}>
          {meta.label}
        </div>
        <div style={{ fontSize: "0.58rem", color: "rgba(200,220,215,0.25)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
          {meta.hib ? "↑ higher" : "↓ lower"}
        </div>
      </div>

      {["classical", "quantum", "hybrid"].map((m) => (
        <div key={m} style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          {best === m && (
            <span style={{ fontSize: "0.55rem", color: METHOD_COLORS[m], letterSpacing: "0.1em", flexShrink: 0 }}>★</span>
          )}
          <GaugeBar
            value={vals[m]}
            min={meta.min ?? 0}
            max={meta.max ?? 1}
            color={METHOD_COLORS[m]}
            hib={meta.hib}
          />
        </div>
      ))}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function MetricsPanel({ result }) {
  const [activeTab, setActiveTab] = useState("clustering"); // clustering | topics | quantum | convergence

  // Build convergence chart data from history array
  const convergenceData = useMemo(() => {
    const hist = result?.convergence_history ?? [];
    if (!hist.length) return [];
    return hist.map((cost, i) => ({ iteration: i + 1, cost: typeof cost === "number" ? cost : 0 }));
  }, [result]);

  if (!result) {
    return (
      <div style={{
        textAlign: "center",
        padding: "3rem",
        color: "rgba(200,220,215,0.2)",
        fontFamily: "var(--font-mono)",
        fontSize: "0.75rem",
        letterSpacing: "0.1em",
      }}>
        <div style={{ fontSize: "2rem", marginBottom: "0.75rem", opacity: 0.2 }}>◉</div>
        Select a completed run to view metrics.
      </div>
    );
  }

  const tabs = [
    { key: "clustering",  label: "Clustering" },
    { key: "topics",      label: "Topics" },
    { key: "quantum",     label: "Quantum" },
    { key: "convergence", label: "Convergence" },
  ];

  // Extract per-method metrics from result schema
  const cl = {
    silhouette_score:        result.silhouette_classical,
    davies_bouldin_index:    result.davies_bouldin_classical,
    nmi:                     result.nmi_classical,
    ari:                     result.ari_classical,
  };
  const qu = {
    silhouette_score:        result.silhouette_quantum,
    davies_bouldin_index:    result.davies_bouldin_quantum,
    nmi:                     result.nmi_quantum,
    ari:                     result.ari_quantum,
  };
  const hy = {
    silhouette_score:        result.silhouette_hybrid,
    davies_bouldin_index:    result.davies_bouldin_hybrid,
    nmi:                     result.nmi_hybrid,
    ari:                     result.ari_hybrid,
  };

  return (
    <div style={{ fontFamily: "var(--font-mono)" }}>

      {/* ── Tab Bar ── */}
      <div style={{
        display: "flex",
        borderBottom: "1px solid rgba(0,230,170,0.08)",
        marginBottom: "1.25rem",
      }}>
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            style={{
              padding: "0.6rem 1.1rem",
              fontSize: "0.68rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: activeTab === key ? "#00e6aa" : "rgba(200,220,215,0.35)",
              borderBottom: activeTab === key ? "2px solid #00e6aa" : "2px solid transparent",
              cursor: "pointer",
              background: "none",
              border: "none",
              borderBottom: activeTab === key ? "2px solid #00e6aa" : "2px solid transparent",
              fontFamily: "var(--font-mono)",
              transition: "color 0.15s",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Method legend ── */}
      {activeTab !== "convergence" && (
        <div style={{ display: "flex", gap: "1.5rem", marginBottom: "1rem" }}>
          {Object.entries(METHOD_COLORS).map(([m, c]) => (
            <div key={m} style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(200,220,215,0.4)" }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: c, boxShadow: `0 0 6px ${c}` }} />
              {m}
            </div>
          ))}
          <div style={{ marginLeft: "auto", fontSize: "0.6rem", color: "rgba(200,220,215,0.2)", letterSpacing: "0.1em" }}>
            ★ best per metric
          </div>
        </div>
      )}

      {/* ── Column headers ── */}
      {activeTab !== "convergence" && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "130px 1fr 1fr 1fr",
          gap: "0.75rem",
          paddingBottom: "0.5rem",
          borderBottom: "1px solid rgba(0,230,170,0.08)",
          marginBottom: "0.25rem",
        }}>
          <div style={{ fontSize: "0.6rem", color: "rgba(200,220,215,0.2)", letterSpacing: "0.15em", textTransform: "uppercase" }}>Metric</div>
          {["Classical", "Quantum", "Hybrid"].map((m, i) => (
            <div key={m} style={{ fontSize: "0.6rem", letterSpacing: "0.15em", textTransform: "uppercase", color: Object.values(METHOD_COLORS)[i], opacity: 0.6 }}>
              {m}
            </div>
          ))}
        </div>
      )}

      {/* ── Clustering tab ── */}
      {activeTab === "clustering" && (
        <div>
          {[
            ["silhouette_score",        cl.silhouette_score,        qu.silhouette_score,        hy.silhouette_score],
            ["davies_bouldin_index",    cl.davies_bouldin_index,    qu.davies_bouldin_index,    hy.davies_bouldin_index],
            ["nmi",                     cl.nmi,                     qu.nmi,                     hy.nmi],
            ["ari",                     cl.ari,                     qu.ari,                     hy.ari],
          ].map(([key, c, q, h]) => (
            <MetricRow key={key} metricKey={key} classicalVal={c} quantumVal={q} hybridVal={h} />
          ))}
        </div>
      )}

      {/* ── Topics tab ── */}
      {activeTab === "topics" && (
        <div>
          {[
            ["coherence_cv",    result.topic_coherence_cv,    null, result.topic_coherence_cv],
            ["coherence_umass", result.topic_coherence_umass, null, result.topic_coherence_umass],
            ["perplexity",      result.perplexity,            null, null],
          ].map(([key, c, q, h]) => (
            <MetricRow key={key} metricKey={key} classicalVal={c} quantumVal={q} hybridVal={h} />
          ))}

          {/* Reconstruction error if available */}
          {result.reconstruction_error != null && (
            <div style={{ marginTop: "1rem", padding: "0.75rem 1rem", background: "rgba(0,230,170,0.03)", border: "1px solid rgba(0,230,170,0.08)", borderRadius: 4 }}>
              <div style={{ fontSize: "0.62rem", letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(0,230,170,0.4)", marginBottom: "0.3rem" }}>
                NMF Reconstruction Error
              </div>
              <div style={{ fontSize: "1rem", color: "#00e6aa", fontFamily: "var(--font-display)", fontWeight: 700 }}>
                {result.reconstruction_error?.toFixed(4) ?? "—"}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Quantum tab ── */}
      {activeTab === "quantum" && (
        <div>
          {/* QAOA cost */}
          {[
            { k: "QAOA Final Cost",       v: result.qaoa_final_cost,           c: "#00b4ff" },
            { k: "Approximation Ratio",   v: result.qaoa_approximation_ratio,  c: "#00e6aa" },
            { k: "Noise TVD",             v: result.noise_tvd,                 c: "#f59e0b" },
            { k: "Noise Fidelity",        v: result.noise_fidelity,            c: "#10ffb0" },
          ].map(({ k, v, c }) => (
            <div key={k} style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "0.65rem 0",
              borderBottom: "1px solid rgba(0,230,170,0.04)",
              gap: "1rem",
            }}>
              <span style={{ fontSize: "0.72rem", color: "rgba(200,220,215,0.5)", flex: 1 }}>{k}</span>
              <div style={{ flex: 2 }}>
                {v != null
                  ? <GaugeBar value={v} min={0} max={1} color={c} hib />
                  : <span style={{ color: "rgba(200,220,215,0.2)", fontSize: "0.7rem" }}>—</span>
                }
              </div>
            </div>
          ))}

          {/* ZNE scale table */}
          {result.zne_scale_expectations && (
            <div style={{ marginTop: "1.25rem" }}>
              <div style={{ fontSize: "0.62rem", letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(0,180,255,0.4)", marginBottom: "0.75rem" }}>
                ZNE Scale Expectations
              </div>
              {Object.entries(result.zne_scale_expectations).map(([scale, exp]) => (
                <div key={scale} style={{ display: "flex", justifyContent: "space-between", padding: "0.35rem 0", fontSize: "0.72rem" }}>
                  <span style={{ color: "rgba(0,180,255,0.5)" }}>λ = {scale}</span>
                  <span style={{ color: "#00b4ff" }}>{typeof exp === "number" ? exp.toFixed(6) : exp}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Convergence tab ── */}
      {activeTab === "convergence" && (
        <div>
          {convergenceData.length === 0 ? (
            <div style={{ textAlign: "center", padding: "2rem", color: "rgba(200,220,215,0.2)", fontSize: "0.75rem" }}>
              No convergence history for this run.
            </div>
          ) : (
            <>
              <div style={{ fontSize: "0.62rem", letterSpacing: "0.12em", textTransform: "uppercase", color: "rgba(0,230,170,0.4)", marginBottom: "1rem" }}>
                QAOA Cost Function · {convergenceData.length} Iterations
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={convergenceData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke="rgba(0,230,170,0.06)" />
                  <XAxis
                    dataKey="iteration"
                    tick={{ fontSize: 10, fill: "rgba(200,220,215,0.3)", fontFamily: "Share Tech Mono" }}
                    axisLine={{ stroke: "rgba(0,230,170,0.08)" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "rgba(200,220,215,0.3)", fontFamily: "Share Tech Mono" }}
                    axisLine={false}
                    tickLine={false}
                    width={55}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Line
                    type="monotone"
                    dataKey="cost"
                    name="QAOA Cost"
                    stroke="#00e6aa"
                    strokeWidth={1.5}
                    dot={false}
                    activeDot={{ r: 3, fill: "#00e6aa" }}
                  />
                  {result.qaoa_final_cost != null && (
                    <ReferenceLine
                      y={result.qaoa_final_cost}
                      stroke="rgba(0,230,170,0.2)"
                      strokeDasharray="4 4"
                      label={{ value: "final", position: "right", fontSize: 9, fill: "rgba(0,230,170,0.35)", fontFamily: "Share Tech Mono" }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>

              {/* Summary stats below chart */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem", marginTop: "1rem" }}>
                {[
                  { label: "Initial Cost",  v: convergenceData[0]?.cost },
                  { label: "Final Cost",    v: convergenceData[convergenceData.length - 1]?.cost },
                  { label: "Iterations",   v: convergenceData.length },
                ].map(({ label, v }) => (
                  <div key={label} style={{ padding: "0.65rem", background: "rgba(0,230,170,0.03)", border: "1px solid rgba(0,230,170,0.07)", borderRadius: 4 }}>
                    <div style={{ fontSize: "0.6rem", letterSpacing: "0.12em", textTransform: "uppercase", color: "rgba(0,230,170,0.4)", marginBottom: "0.3rem" }}>{label}</div>
                    <div style={{ fontSize: "1rem", color: "#00e6aa", fontFamily: "var(--font-display)", fontWeight: 700 }}>
                      {v != null ? (typeof v === "number" && v % 1 !== 0 ? v.toFixed(5) : v) : "—"}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}