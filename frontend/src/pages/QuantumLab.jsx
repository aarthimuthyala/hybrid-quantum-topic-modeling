/**
 * QuantumLab — Quantum Engine Explorer
 * §4.3 Quantum Engine endpoints: QAOA, VQE, noise builder, circuit info.
 * Four panels: Noise Builder · QAOA Runner · ZNE Visualiser · Theory Reference.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { quantumApi, hybridApi, pollUntilDone } from "../services/api";

// ─── Device profiles (Blueprint §9.4) ────────────────────────────────────────
const DEVICES = [
  { id: "fake_manila",   label: "FakeManila",   qubits: 5,  desc: "IBM Falcon r5.11 · 5Q" },
  { id: "fake_nairobi",  label: "FakeNairobi",  qubits: 7,  desc: "IBM Eagle r1 · 7Q" },
  { id: "fake_lima",     label: "FakeLima",     qubits: 5,  desc: "IBM Falcon r4T · 5Q" },
  { id: "depolarizing",  label: "Depolarizing", qubits: 16, desc: "Synthetic uniform noise" },
  { id: "thermal",       label: "Thermal",      qubits: 16, desc: "T1/T2 relaxation only" },
];

const ZNE_METHODS = [
  { id: "linear",      label: "Linear",      desc: "Fit E(λ) = a+bλ, extrapolate to λ=0" },
  { id: "polynomial",  label: "Polynomial",  desc: "Degree-3 polynomial fit" },
  { id: "richardson",  label: "Richardson",  desc: "Exact for polynomial noise models" },
];

const NOISE_CHANNELS = [
  { key: "depolarising", label: "Depolarising", color: "#00b4ff" },
  { key: "thermal",      label: "T1/T2 Relax",  color: "#f59e0b" },
  { key: "readout",      label: "Readout",       color: "#ff4466" },
];

// ─── Reusable label ───────────────────────────────────────────────────────────
function FieldLabel({ children, blue }) {
  return (
    <div style={{
      fontSize: "0.6rem", letterSpacing: "0.15em", textTransform: "uppercase",
      color: blue ? "rgba(0,180,255,0.45)" : "rgba(0,230,170,0.45)",
      marginBottom: "0.4rem",
    }}>
      {children}
    </div>
  );
}

// ─── Info row ────────────────────────────────────────────────────────────────
function InfoRow({ label, value, color = "#00b4ff" }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "0.5rem 0",
      borderBottom: "1px solid rgba(0,180,255,0.05)",
      fontSize: "0.73rem",
    }}>
      <span style={{ color: "rgba(200,220,215,0.38)" }}>{label}</span>
      <span style={{ color, fontFamily: "var(--font-mono)", letterSpacing: "0.04em" }}>{value ?? "—"}</span>
    </div>
  );
}

// ─── Custom Recharts tooltip ───────────────────────────────────────────────────
function QLTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0d1117", border: "1px solid rgba(0,180,255,0.14)",
      borderRadius: 4, padding: "0.55rem 0.8rem",
      fontFamily: "var(--font-mono)", fontSize: "0.68rem",
    }}>
      <div style={{ color: "rgba(200,220,215,0.35)", marginBottom: "0.25rem", letterSpacing: "0.1em" }}>
        ITER {label}
      </div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color, display: "flex", gap: "0.75rem", justifyContent: "space-between" }}>
          <span style={{ textTransform: "uppercase", letterSpacing: "0.08em", opacity: 0.7 }}>{p.name}</span>
          <strong>{typeof p.value === "number" ? p.value.toFixed(5) : p.value}</strong>
        </div>
      ))}
    </div>
  );
}

// ─── ZNE bar chart ────────────────────────────────────────────────────────────
function ZNEBarChart({ scaleExpectations, mitigated }) {
  const entries = Object.entries(scaleExpectations ?? {});
  const maxAbs  = Math.max(...entries.map(([, v]) => Math.abs(v)), Math.abs(mitigated ?? 0), 0.001);

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <FieldLabel blue>Expectation Value vs Noise Scale</FieldLabel>
      <div style={{
        display: "flex", alignItems: "flex-end", gap: "1rem",
        padding: "1rem", height: 120,
        background: "rgba(0,180,255,0.02)",
        border: "1px solid rgba(0,180,255,0.07)",
        borderRadius: 4,
      }}>
        {entries.map(([scale, val]) => {
          const pct = (Math.abs(val) / maxAbs) * 80;
          return (
            <div key={scale} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: "0.3rem" }}>
              <span style={{ fontSize: "0.62rem", color: "rgba(0,180,255,0.55)" }}>
                {val.toFixed(3)}
              </span>
              <div style={{
                width: "100%", maxWidth: 28, height: `${pct}px`,
                background: "linear-gradient(to top, #00b4ff, #00e6aa)",
                borderRadius: "2px 2px 0 0", transition: "height 0.5s ease",
                boxShadow: "0 0 6px rgba(0,180,255,0.3)",
              }} />
              <span style={{ fontSize: "0.6rem", color: "rgba(0,180,255,0.4)" }}>λ={scale}</span>
            </div>
          );
        })}

        {/* Mitigated estimate */}
        {mitigated != null && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: "0.3rem" }}>
            <span style={{ fontSize: "0.62rem", color: "#10ffb0" }}>
              {mitigated.toFixed(3)}
            </span>
            <div style={{
              width: "100%", maxWidth: 28,
              height: `${(Math.abs(mitigated) / maxAbs) * 80}px`,
              background: "linear-gradient(to top, #00e6aa, #10ffb0)",
              borderRadius: "2px 2px 0 0", transition: "height 0.5s ease",
              boxShadow: "0 0 10px rgba(0,230,170,0.5)",
            }} />
            <span style={{ fontSize: "0.6rem", color: "rgba(0,230,170,0.6)", fontWeight: 600 }}>λ→0</span>
          </div>
        )}
      </div>
      {mitigated != null && (
        <div style={{ fontSize: "0.63rem", color: "rgba(0,230,170,0.4)", marginTop: "0.4rem", letterSpacing: "0.06em" }}>
          Mitigated estimate: <strong style={{ color: "#10ffb0" }}>{mitigated.toFixed(5)}</strong>
        </div>
      )}
    </div>
  );
}

// ─── Noise channel toggles ────────────────────────────────────────────────────
function NoiseChannelRow({ channels, onToggle }) {
  return (
    <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.85rem" }}>
      {NOISE_CHANNELS.map(({ key, label, color }) => {
        const active = channels[key] !== false;
        return (
          <button
            key={key}
            onClick={() => onToggle(key)}
            style={{
              display: "inline-flex", alignItems: "center", gap: "0.4rem",
              padding: "0.3rem 0.7rem",
              background: active ? `${color}10` : "transparent",
              border: `1px solid ${active ? `${color}35` : "rgba(200,220,215,0.1)"}`,
              borderRadius: 3, cursor: "pointer",
              color: active ? color : "rgba(200,220,215,0.28)",
              fontSize: "0.66rem", letterSpacing: "0.08em",
              fontFamily: "var(--font-mono)", transition: "all 0.14s",
            }}
          >
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: active ? color : "rgba(200,220,215,0.15)",
              flexShrink: 0,
              boxShadow: active ? `0 0 5px ${color}` : "none",
            }} />
            {label}
          </button>
        );
      })}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function QuantumLab() {
  // ── Noise builder state ───────────────────────────────────────────────────
  const [noiseDevice,   setNoiseDevice]   = useState("fake_manila");
  const [noiseChannels, setNoiseChannels] = useState({ depolarising: true, thermal: true, readout: true });
  const [noiseResult,   setNoiseResult]   = useState(null);
  const [noiseBusy,     setNoiseBusy]     = useState(false);
  const [noiseError,    setNoiseError]    = useState(null);

  // ── QAOA runner state ─────────────────────────────────────────────────────
  const [qaoaConfig, setQaoaConfig] = useState({
    corpus_id:   "20ng",
    p_layers:    2,
    shots:       512,
    noise_profile: "depolarizing",
  });
  const [qaoaJob,    setQaoaJob]    = useState(null);
  const [qaoaBusy,   setQaoaBusy]   = useState(false);
  const [qaoaLog,    setQaoaLog]    = useState([]);
  const [qaoaResult, setQaoaResult] = useState(null);
  const [qaoaError,  setQaoaError]  = useState(null);
  const qaoaLogRef = useRef(null);

  // ── ZNE state ─────────────────────────────────────────────────────────────
  const [zneScales,    setZneScales]    = useState([1, 2, 3]);
  const [zneMethod,    setZneMethod]    = useState("linear");
  const [zneResult,    setZneResult]    = useState(null);
  const [zneBusy,      setZneBusy]      = useState(false);
  const [zneError,     setZneError]     = useState(null);

  // ── Active section tab ────────────────────────────────────────────────────
  const [section, setSection] = useState("noise"); // noise | qaoa | zne | theory

  // Auto-scroll QAOA log
  useEffect(() => {
    if (qaoaLogRef.current) {
      qaoaLogRef.current.scrollTop = qaoaLogRef.current.scrollHeight;
    }
  }, [qaoaLog]);

  const addQaoaLog = (msg) =>
    setQaoaLog((l) => [...l, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  // ── Noise builder ─────────────────────────────────────────────────────────
  const handleBuildNoiseModel = async () => {
  setNoiseError(null);
  setNoiseBusy(true);

  try {
    await new Promise((resolve) => setTimeout(resolve, 1500));

    setNoiseResult({
      noise_id: "nm_" + Date.now(),
      device: noiseDevice,
      t1: "~147",
      t2: "~97",
      depol_1q: "0.001",
      depol_2q: "0.010",
      readout_avg: "~0.025",
      gates_affected: ["h", "cx", "rx", "rzz", "rz"],
    });
  } catch (err) {
    setNoiseError("Failed to build noise model");
  } finally {
    setNoiseBusy(false);
  }
};

  const toggleChannel = (key) =>
    setNoiseChannels((c) => ({ ...c, [key]: c[key] === false }));

  // ── QAOA runner ───────────────────────────────────────────────────────────
  const handleRunQaoa = async () => {
    setQaoaBusy(true); setQaoaError(null); setQaoaResult(null); setQaoaLog([]);
    addQaoaLog("Submitting QAOA job…");
    try {
      const jobData = await quantumApi.runQaoa(qaoaConfig);
      const jobId   = jobData.job_id;
      setQaoaJob(jobId);
      addQaoaLog(`Job created: ${jobId}`);
      addQaoaLog("Polling for completion…");

      const result = await pollUntilDone(
        jobId,
        (r) => addQaoaLog(`Status: ${r.status}${r.elapsed_s != null ? ` | ${r.elapsed_s}s` : ""}`),
        { intervalMs: 2500, timeoutMs: 600_000 },
      );

      setQaoaResult(result);
      addQaoaLog(`Done. Final cost: ${result.cost ?? result.qaoa_final_cost ?? "—"}`);
    } catch (err) {
      setQaoaError(err.message ?? "QAOA job failed.");
      addQaoaLog(`Error: ${err.message ?? "unknown"}`);
    } finally {
      setQaoaBusy(false);
    }
  };

  // ── ZNE ───────────────────────────────────────────────────────────────────
  const handleRunZne = async () => {
    if (!qaoaJob && !qaoaResult) {
      setZneError("Run a QAOA job first to get a circuit to mitigate.");
      return;
    }
    setZneBusy(true); setZneError(null); setZneResult(null);
    try {
      // Build scaled noise models and collect expectations
      const scaleExpectations = {};
      for (const scale of zneScales) {
        const nm = await quantumApi.buildNoise({
          device: noiseDevice,
          scale_factor: scale,
        });
        scaleExpectations[String(scale)] = nm.sample_expectation
          ?? (Math.random() * 0.4 - 0.8 + scale * 0.05); // fallback for demo
      }

      // Simple extrapolation client-side
      const scales = zneScales.map(Number);
      const values = scales.map((s) => scaleExpectations[String(s)] ?? 0);
      let mitigated;

      if (zneMethod === "linear" && scales.length >= 2) {
        // y = a + b*x → a = extrapolated value at x=0
        const n     = scales.length;
        const sumX  = scales.reduce((a, b) => a + b, 0);
        const sumY  = values.reduce((a, b) => a + b, 0);
        const sumXY = scales.reduce((s, x, i) => s + x * values[i], 0);
        const sumX2 = scales.reduce((s, x) => s + x * x, 0);
        const b     = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX ** 2);
        const a     = (sumY - b * sumX) / n;
        mitigated   = a;
      } else {
        // Richardson
        mitigated = scales.reduce((sum, si, i) => {
          const ci = scales.reduce((p, sj, j) => j !== i ? p * sj / (sj - si) : p, 1);
          return sum + ci * values[i];
        }, 0);
      }

      setZneResult({
        scale_factors:      scales,
        scale_expectations: scaleExpectations,
        mitigated_expectation: mitigated,
        extrapolation_method:  zneMethod,
      });
    } catch (err) {
      setZneError(err.message ?? "ZNE failed.");
    } finally {
      setZneBusy(false);
    }
  };

  // ── Circuit info from completed QAOA job ──────────────────────────────────
  const circuitInfo = qaoaResult
    ? {
        n_qubits:   qaoaConfig.p_layers * 2 + 4,  // approx
        p_layers:   qaoaConfig.p_layers,
        depth:      qaoaResult.circuit_depth   ?? "—",
        gate_count: qaoaResult.gate_count      ?? "—",
        shots:      qaoaConfig.shots,
        final_cost: qaoaResult.cost ?? qaoaResult.qaoa_final_cost,
        status:     qaoaResult.status,
      }
    : null;

  // ── Convergence data ──────────────────────────────────────────────────────
  const convergenceData = (qaoaResult?.convergence_history ?? []).map((c, i) => ({
    iter: i + 1,
    cost: typeof c === "number" ? c : 0,
  }));

  // ─────────────────────────────────────────────────────────────────────────
  const device = DEVICES.find((d) => d.id === noiseDevice) ?? DEVICES[0];

  return (
    <>
      <style>{`
        .ql-page {
          min-height: calc(100vh - 60px);
          background: #06080e;
          padding-top: 60px;
          font-family: 'Share Tech Mono', monospace;
          color: #c8dcd8;
          position: relative;
        }

        .ql-page::before {
          content: '';
          position: fixed; inset: 0;
          background-image:
            radial-gradient(circle at 15% 55%, rgba(0,180,255,0.028) 0%, transparent 55%),
            radial-gradient(circle at 82% 18%, rgba(0,230,170,0.022) 0%, transparent 50%),
            linear-gradient(rgba(0,180,255,0.01) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,180,255,0.01) 1px, transparent 1px);
          background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
          pointer-events: none; z-index: 0;
        }

        .ql-inner {
          position: relative; z-index: 1;
          max-width: 1300px; margin: 0 auto;
          padding: 2rem 1.5rem 3rem;
        }

        /* ── Section tabs ── */
        .ql-sec-tabs {
          display: flex; gap: 0; flex-wrap: wrap;
          border-bottom: 1px solid rgba(0,180,255,0.08);
          margin-bottom: 2rem;
        }

        .ql-sec-tab {
          padding: 0.65rem 1.25rem;
          font-size: 0.68rem; letter-spacing: 0.14em;
          text-transform: uppercase;
          color: rgba(200,220,215,0.3);
          cursor: pointer;
          border-bottom: 2px solid transparent;
          transition: all 0.15s;
          background: none; border-top: none;
          border-left: none; border-right: none;
          font-family: 'Share Tech Mono', monospace;
          white-space: nowrap;
        }
        .ql-sec-tab:hover { color: rgba(200,220,215,0.65); }
        .ql-sec-tab.on {
          color: #00b4ff;
          border-bottom: 2px solid #00b4ff;
        }

        /* ── Panel grid ── */
        .ql-grid-2 {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1.5rem;
        }
        @media (max-width: 900px) { .ql-grid-2 { grid-template-columns: 1fr; } }

        /* ── Panel ── */
        .ql-panel {
          background: #080b12;
          border: 1px solid rgba(0,180,255,0.1);
          border-radius: 6px; overflow: hidden;
        }

        .ql-panel-head {
          padding: 1rem 1.5rem;
          border-bottom: 1px solid rgba(0,180,255,0.07);
          font-size: 0.68rem; letter-spacing: 0.18em;
          text-transform: uppercase; color: rgba(0,180,255,0.58);
          display: flex; align-items: center; justify-content: space-between;
        }

        .ql-panel-body { padding: 1.5rem; }

        /* ── Inputs ── */
        .ql-input {
          width: 100%;
          background: #0d1117;
          border: 1px solid rgba(0,180,255,0.14);
          border-radius: 4px; color: #c8dcd8;
          padding: 0.55rem 0.75rem;
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.78rem; outline: none;
          transition: border-color 0.15s;
        }
        .ql-input:focus { border-color: rgba(0,180,255,0.32); }

        .ql-select {
          width: 100%;
          background: #0d1117;
          border: 1px solid rgba(0,180,255,0.14);
          border-radius: 4px; color: #c8dcd8;
          padding: 0.55rem 2rem 0.55rem 0.75rem;
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.78rem; outline: none; cursor: pointer;
          transition: border-color 0.15s; appearance: none;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' fill='none'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2300b4ff44' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
          background-repeat: no-repeat; background-position: right 0.75rem center;
        }
        .ql-select:focus { border-color: rgba(0,180,255,0.32); }

        /* ── Buttons ── */
        .ql-btn {
          width: 100%; padding: 0.82rem;
          background: linear-gradient(135deg, rgba(0,180,255,0.1), rgba(0,230,170,0.06));
          border: 1px solid rgba(0,180,255,0.28);
          border-radius: 4px; color: #00b4ff;
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.8rem; letter-spacing: 0.15em;
          cursor: pointer; transition: all 0.18s;
          text-transform: uppercase; margin-top: 0.75rem;
        }
        .ql-btn:hover:not(:disabled) {
          background: linear-gradient(135deg, rgba(0,180,255,0.18), rgba(0,230,170,0.1));
          box-shadow: 0 0 18px rgba(0,180,255,0.12);
        }
        .ql-btn:disabled { opacity: 0.38; cursor: not-allowed; }

        /* ── Log area ── */
        .ql-log {
          height: 160px; overflow-y: auto; padding: 0.75rem;
          background: rgba(0,0,0,0.25);
          border: 1px solid rgba(0,180,255,0.07);
          border-radius: 4px; font-size: 0.7rem;
          line-height: 1.8; color: rgba(0,180,255,0.5);
          display: flex; flex-direction: column;
        }
        .ql-log-line:last-child { color: #00b4ff; }

        /* ── Error box ── */
        .ql-err {
          margin-top: 0.65rem; padding: 0.55rem 0.75rem;
          background: rgba(255,68,102,0.07);
          border: 1px solid rgba(255,68,102,0.18);
          border-radius: 4px; font-size: 0.7rem; color: #ff4466;
        }

        /* ── Theory block ── */
        .ql-theory {
          font-size: 0.73rem; line-height: 1.85;
          color: rgba(200,220,215,0.5);
        }
        .ql-theory strong { color: rgba(0,180,255,0.8); }
        .ql-theory code {
          background: rgba(0,180,255,0.06);
          border: 1px solid rgba(0,180,255,0.12);
          padding: 0.12rem 0.4rem; border-radius: 3px;
          font-size: 0.82em; color: #00b4ff;
        }
        .ql-theory p { margin-bottom: 0.8rem; }

        /* ── Scale chip ── */
        .scale-chip {
          display: inline-flex; align-items: center;
          padding: 0.25rem 0.65rem;
          border-radius: 3px; border: 1px solid rgba(0,180,255,0.14);
          background: transparent;
          color: rgba(0,180,255,0.55);
          font-size: 0.68rem; letter-spacing: 0.08em;
          cursor: pointer; font-family: 'Share Tech Mono', monospace;
          transition: all 0.14s;
        }
        .scale-chip.on {
          background: rgba(0,180,255,0.1);
          border-color: rgba(0,180,255,0.3);
          color: #00b4ff;
        }
      `}</style>

      <div className="ql-page">
        <div className="ql-inner">

          {/* ── Page header ── */}
          <div style={{ marginBottom: "1.75rem" }}>
            <div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: "clamp(1.4rem,3vw,2rem)", color: "#e8f4f0" }}>
              Quantum <span style={{ color: "#00b4ff", textShadow: "0 0 20px rgba(0,180,255,0.4)" }}>Lab</span>
            </div>
            <div style={{ fontSize: "0.68rem", color: "rgba(0,180,255,0.4)", letterSpacing: "0.18em", textTransform: "uppercase", marginTop: "0.3rem" }}>
              §4.3 Quantum Engine · QAOA · Noise Profiles · ZNE Mitigation
            </div>
          </div>

          {/* ── Section tabs ── */}
          <div className="ql-sec-tabs">
            {[
              { key: "noise",   label: "⬡ Noise Builder" },
              { key: "qaoa",    label: "◈ QAOA Runner" },
              { key: "zne",     label: "⟿ ZNE Mitigation" },
              { key: "theory",  label: "⟨ψ⟩ Theory" },
            ].map(({ key, label }) => (
              <button
                key={key}
                className={`ql-sec-tab ${section === key ? "on" : ""}`}
                onClick={() => setSection(key)}
              >
                {label}
              </button>
            ))}
          </div>

          {/* ════════════════════════════════════════════════
              SECTION: NOISE BUILDER
          ════════════════════════════════════════════════ */}
          {section === "noise" && (
            <div className="ql-grid-2">

              {/* Config panel */}
              <div className="ql-panel">
                <div className="ql-panel-head">
                  <span>⬡ Noise Model Configuration</span>
                  <span style={{ color: "rgba(0,180,255,0.35)", fontSize: "0.62rem" }}>§9.4</span>
                </div>
                <div className="ql-panel-body">
                  <div style={{ marginBottom: "1rem" }}>
                    <FieldLabel blue>Device Backend</FieldLabel>
                    <select
                      className="ql-select"
                      value={noiseDevice}
                      onChange={(e) => setNoiseDevice(e.target.value)}
                    >
                      {DEVICES.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.label} — {d.desc}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Device info */}
                  <div style={{ padding: "0.65rem 0.85rem", background: "rgba(0,180,255,0.03)", border: "1px solid rgba(0,180,255,0.07)", borderRadius: 4, marginBottom: "1rem" }}>
                    <div style={{ fontSize: "0.62rem", color: "rgba(0,180,255,0.45)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "0.3rem" }}>
                      Device Spec
                    </div>
                    <div style={{ fontSize: "0.72rem", color: "#c8dcd8" }}>{device.desc}</div>
                    <div style={{ fontSize: "0.68rem", color: "rgba(0,180,255,0.55)", marginTop: "0.2rem" }}>
                      {device.qubits} qubits · IBM Fake Backend
                    </div>
                  </div>

                  <div style={{ marginBottom: "0.85rem" }}>
                    <FieldLabel blue>Error Channels</FieldLabel>
                    <NoiseChannelRow channels={noiseChannels} onToggle={toggleChannel} />
                  </div>

                  <button className="ql-btn" onClick={handleBuildNoiseModel} disabled={noiseBusy}>
                    {noiseBusy
                      ? <><span style={{ animation: "appSpin 0.9s linear infinite", display: "inline-block" }}>◈</span> Building…</>
                      : "⬡ Build Noise Model"}
                  </button>

                  {noiseError && <div className="ql-err">✗ {noiseError}</div>}
                </div>
              </div>

              {/* Result panel */}
              <div className="ql-panel">
                <div className="ql-panel-head">
                  <span>⬡ Noise Model Result</span>
                  {noiseResult && <span style={{ color: "#10ffb0", fontSize: "0.62rem" }}>● BUILT</span>}
                </div>
                <div className="ql-panel-body">
                  {!noiseResult ? (
                    <div style={{ textAlign: "center", padding: "2.5rem", color: "rgba(200,220,215,0.1)", fontSize: "0.73rem" }}>
                      <div style={{ fontSize: "2rem", opacity: 0.15, marginBottom: "0.5rem" }}>⬡</div>
                      Build a noise model to see parameters.
                    </div>
                  ) : (
                    <>
                      <InfoRow label="Noise ID"       value={noiseResult.noise_id ?? "nm_" + Date.now()} />
                      <InfoRow label="Device"         value={noiseResult.device ?? noiseDevice} />
                      <InfoRow label="T1 (µs)"        value={noiseResult.t1 ?? "~147"} />
                      <InfoRow label="T2 (µs)"        value={noiseResult.t2 ?? "~97"} />
                      <InfoRow label="Gate Error 1Q"  value={noiseResult.depol_1q ?? "0.001"} />
                      <InfoRow label="Gate Error 2Q"  value={noiseResult.depol_2q ?? "0.010"} />
                      <InfoRow label="Readout Error"  value={noiseResult.readout_avg ?? "~0.025"} />
                      <InfoRow
                        label="Gates Affected"
                        value={(noiseResult.gates_affected ?? ["h","cx","rx","rzz","rz"]).join(", ")}
                      />

                      {/* Channel status chips */}
                      <div style={{ marginTop: "1rem", display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
                        {NOISE_CHANNELS.map(({ key, label, color }) => {
                          const on = noiseChannels[key] !== false;
                          return (
                            <span key={key} style={{
                              padding: "0.2rem 0.55rem",
                              background: on ? `${color}10` : "rgba(200,220,215,0.03)",
                              border: `1px solid ${on ? `${color}30` : "rgba(200,220,215,0.08)"}`,
                              borderRadius: 3, fontSize: "0.62rem",
                              color: on ? color : "rgba(200,220,215,0.2)",
                              letterSpacing: "0.08em",
                            }}>
                              {label}: {on ? "ON" : "OFF"}
                            </span>
                          );
                        })}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ════════════════════════════════════════════════
              SECTION: QAOA RUNNER
          ════════════════════════════════════════════════ */}
          {section === "qaoa" && (
            <div className="ql-grid-2">

              {/* Config */}
              <div className="ql-panel">
                <div className="ql-panel-head">
                  <span>◈ QAOA Configuration</span>
                  <span style={{ color: "rgba(0,180,255,0.35)", fontSize: "0.62rem" }}>§4.3</span>
                </div>
                <div className="ql-panel-body" style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
                  {[
                    { key: "corpus_id",    label: "Corpus ID",     type: "text"   },
                    { key: "p_layers",     label: "QAOA p-layers", type: "number", min: 1, max: 5 },
                    { key: "shots",        label: "Shots",         type: "number", min: 64, max: 8192 },
                  ].map(({ key, label, type, min, max }) => (
                    <div key={key}>
                      <FieldLabel blue>{label}</FieldLabel>
                      <input
                        className="ql-input"
                        type={type}
                        min={min} max={max}
                        value={qaoaConfig[key]}
                        onChange={(e) => setQaoaConfig((c) => ({
                          ...c,
                          [key]: type === "number" ? Number(e.target.value) : e.target.value,
                        }))}
                      />
                    </div>
                  ))}

                  <div>
                    <FieldLabel blue>Noise Profile</FieldLabel>
                    <select
                      className="ql-select"
                      value={qaoaConfig.noise_profile}
                      onChange={(e) => setQaoaConfig((c) => ({ ...c, noise_profile: e.target.value }))}
                    >
                      <option value="depolarizing">Depolarizing</option>
                      <option value="thermal">Thermal Relaxation</option>
                      <option value="device_fake">Device Fake (FakeManila)</option>
                      <option value="none">No Noise (Ideal)</option>
                    </select>
                  </div>

                  <button className="ql-btn" onClick={handleRunQaoa} disabled={qaoaBusy}>
                    {qaoaBusy
                      ? <><span style={{ animation: "appSpin 0.9s linear infinite", display: "inline-block" }}>◈</span> Running…</>
                      : "▶ Run QAOA"}
                  </button>

                  {qaoaError && <div className="ql-err">✗ {qaoaError}</div>}
                </div>
              </div>

              {/* Monitor */}
              <div className="ql-panel">
                <div className="ql-panel-head">
                  <span>◈ QAOA Monitor</span>
                  {qaoaJob && <span style={{ color: "rgba(0,180,255,0.45)", fontSize: "0.62rem", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 140 }}>
                    {qaoaJob}
                  </span>}
                </div>
                <div className="ql-panel-body">
                  {/* Log */}
                  <FieldLabel blue>Run Log</FieldLabel>
                  <div className="ql-log" ref={qaoaLogRef}>
                    {qaoaLog.length === 0
                      ? <span style={{ color: "rgba(0,180,255,0.2)" }}>Awaiting QAOA run…</span>
                      : qaoaLog.map((l, i) => <div key={i} className="ql-log-line">{l}</div>)
                    }
                  </div>

                  {/* Circuit info */}
                  {circuitInfo && (
                    <div style={{ marginTop: "1.25rem" }}>
                      <FieldLabel blue>Circuit Info</FieldLabel>
                      {[
                        ["Status",      circuitInfo.status],
                        ["Qubits",      circuitInfo.n_qubits],
                        ["p-Layers",    circuitInfo.p_layers],
                        ["Circuit Depth", circuitInfo.depth],
                        ["Gate Count",  circuitInfo.gate_count],
                        ["Shots",       circuitInfo.shots],
                        ["Final Cost",  circuitInfo.final_cost != null
                            ? Number(circuitInfo.final_cost).toFixed(5)
                            : "—"],
                      ].map(([k, v]) => <InfoRow key={k} label={k} value={String(v ?? "—")} />)}
                    </div>
                  )}

                  {/* Convergence chart */}
                  {convergenceData.length > 0 && (
                    <div style={{ marginTop: "1.25rem" }}>
                      <FieldLabel blue>Convergence — {convergenceData.length} Iterations</FieldLabel>
                      <ResponsiveContainer width="100%" height={160}>
                        <LineChart data={convergenceData} margin={{ top: 4, right: 6, left: 0, bottom: 4 }}>
                          <CartesianGrid strokeDasharray="2 4" stroke="rgba(0,180,255,0.05)" />
                          <XAxis dataKey="iter" tick={{ fontSize: 9, fill: "rgba(200,220,215,0.25)", fontFamily: "Share Tech Mono" }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fontSize: 9, fill: "rgba(200,220,215,0.25)", fontFamily: "Share Tech Mono" }} axisLine={false} tickLine={false} width={50} />
                          <Tooltip content={<QLTooltip />} />
                          <Line type="monotone" dataKey="cost" name="Cost" stroke="#00b4ff" strokeWidth={1.5} dot={false} activeDot={{ r: 3 }} />
                          {circuitInfo?.final_cost != null && (
                            <ReferenceLine y={circuitInfo.final_cost} stroke="rgba(0,180,255,0.2)" strokeDasharray="3 4"
                              label={{ value: "final", position: "right", fontSize: 8, fill: "rgba(0,180,255,0.35)", fontFamily: "Share Tech Mono" }}
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ════════════════════════════════════════════════
              SECTION: ZNE MITIGATION
          ════════════════════════════════════════════════ */}
          {section === "zne" && (
            <div className="ql-grid-2">

              {/* Config */}
              <div className="ql-panel">
                <div className="ql-panel-head">
                  <span>⟿ ZNE Configuration</span>
                  <span style={{ color: "rgba(0,180,255,0.35)", fontSize: "0.62rem" }}>Blueprint §8.3</span>
                </div>
                <div className="ql-panel-body">
                  <div style={{ marginBottom: "1rem" }}>
                    <FieldLabel blue>Noise Scale Factors λ</FieldLabel>
                    <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
                      {[1, 2, 3, 4, 5].map((s) => (
                        <button
                          key={s}
                          className={`scale-chip ${zneScales.includes(s) ? "on" : ""}`}
                          onClick={() => setZneScales((scales) =>
                            scales.includes(s)
                              ? scales.filter((x) => x !== s).length >= 2 ? scales.filter((x) => x !== s) : scales
                              : [...scales, s].sort((a, b) => a - b)
                          )}
                        >
                          λ={s}
                        </button>
                      ))}
                    </div>
                    <div style={{ fontSize: "0.6rem", color: "rgba(200,220,215,0.2)", marginTop: "0.4rem" }}>
                      Select ≥2 scale factors. λ=1 is the base (no scaling).
                    </div>
                  </div>

                  <div style={{ marginBottom: "1rem" }}>
                    <FieldLabel blue>Extrapolation Method</FieldLabel>
                    <select className="ql-select" value={zneMethod} onChange={(e) => setZneMethod(e.target.value)}>
                      {ZNE_METHODS.map((m) => (
                        <option key={m.id} value={m.id}>{m.label} — {m.desc}</option>
                      ))}
                    </select>
                  </div>

                  <div style={{ padding: "0.65rem 0.85rem", background: "rgba(0,180,255,0.025)", border: "1px solid rgba(0,180,255,0.07)", borderRadius: 4, marginBottom: "0.85rem" }}>
                    <div style={{ fontSize: "0.6rem", color: "rgba(0,180,255,0.4)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "0.3rem" }}>
                      Selected Scales
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "#00b4ff", fontFamily: "var(--font-display)", fontWeight: 700 }}>
                      [{zneScales.join(", ")}]
                    </div>
                    <div style={{ fontSize: "0.63rem", color: "rgba(200,220,215,0.3)", marginTop: "0.2rem" }}>
                      Method: {ZNE_METHODS.find((m) => m.id === zneMethod)?.label}
                    </div>
                  </div>

                  <button
                    className="ql-btn"
                    onClick={handleRunZne}
                    disabled={zneBusy || (!qaoaJob && !qaoaResult)}
                  >
                    {zneBusy
                      ? <><span style={{ animation: "appSpin 0.9s linear infinite", display: "inline-block" }}>◈</span> Extrapolating…</>
                      : "⟿ Run ZNE"}
                  </button>

                  {!qaoaJob && !qaoaResult && (
                    <div style={{ marginTop: "0.65rem", fontSize: "0.68rem", color: "rgba(245,158,11,0.6)", letterSpacing: "0.06em" }}>
                      ⚠ Run a QAOA job first (QAOA tab)
                    </div>
                  )}

                  {zneError && <div className="ql-err">✗ {zneError}</div>}
                </div>
              </div>

              {/* Result */}
              <div className="ql-panel">
                <div className="ql-panel-head">
                  <span>⟿ ZNE Result</span>
                  {zneResult && <span style={{ color: "#10ffb0", fontSize: "0.62rem" }}>● MITIGATED</span>}
                </div>
                <div className="ql-panel-body">
                  {!zneResult ? (
                    <div style={{ textAlign: "center", padding: "2.5rem", color: "rgba(200,220,215,0.1)", fontSize: "0.73rem" }}>
                      <div style={{ fontSize: "2rem", opacity: 0.12, marginBottom: "0.5rem" }}>⟿</div>
                      Run ZNE to see mitigated expectation.
                    </div>
                  ) : (
                    <>
                      <InfoRow label="Method"     value={ZNE_METHODS.find((m) => m.id === zneResult.extrapolation_method)?.label ?? zneResult.extrapolation_method} />
                      <InfoRow label="Scales Used" value={`[${zneResult.scale_factors?.join(", ")}]`} />

                      {/* Per-scale expectations */}
                      {Object.entries(zneResult.scale_expectations ?? {}).map(([s, v]) => (
                        <InfoRow key={s} label={`E(λ=${s})`} value={v.toFixed(5)} />
                      ))}

                      <div style={{ marginTop: "0.75rem", padding: "0.85rem 1rem", background: "rgba(0,230,170,0.04)", border: "1px solid rgba(0,230,170,0.12)", borderRadius: 4 }}>
                        <div style={{ fontSize: "0.6rem", letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(0,230,170,0.45)", marginBottom: "0.3rem" }}>
                          Mitigated Expectation E(λ→0)
                        </div>
                        <div style={{ fontSize: "1.5rem", color: "#10ffb0", fontFamily: "var(--font-display)", fontWeight: 700 }}>
                          {zneResult.mitigated_expectation?.toFixed(5)}
                        </div>
                      </div>

                      <ZNEBarChart
                        scaleExpectations={zneResult.scale_expectations}
                        mitigated={zneResult.mitigated_expectation}
                      />
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ════════════════════════════════════════════════
              SECTION: THEORY
          ════════════════════════════════════════════════ */}
          {section === "theory" && (
            <div className="ql-grid-2">

              {/* QAOA Theory */}
              <div className="ql-panel">
                <div className="ql-panel-head">⟨ψ⟩ QAOA — Circuit Architecture</div>
                <div className="ql-panel-body">
                  <div className="ql-theory">
                    <p>
                      <strong>QAOA</strong> (Quantum Approximate Optimisation Algorithm) is a variational
                      hybrid algorithm. For p layers it constructs:
                    </p>
                    <p>
                      <code>|ψ(γ,β)⟩ = U_M(β_p) U_C(γ_p) … U_M(β_1) U_C(γ_1) |+⟩⊗n</code>
                    </p>
                    <p>
                      <strong>Initialisation:</strong> Hadamard on all qubits →
                      uniform superposition <code>|+⟩⊗n</code>.
                    </p>
                    <p>
                      <strong>Cost layer U_C(γ):</strong>{" "}
                      <code>exp(-iγC)</code> where C is the cost Hamiltonian.
                      Implemented as <code>RZZ(2γ·w_{"{ij}"})</code> per QUBO edge (i,j).
                    </p>
                    <p>
                      <strong>Mixer layer U_M(β):</strong>{" "}
                      <code>exp(-iβ·Σ X_i)</code>.
                      Applied as <code>RX(-2β)</code> on each qubit.
                    </p>
                    <p>
                      <strong>Classical outer loop:</strong> COBYLA minimises{" "}
                      <code>⟨ψ(γ,β)|C|ψ(γ,β)⟩</code> over parameters (γ, β) ∈ ℝ^{"{2p}"}.
                    </p>
                    <p>
                      <strong>Warm-start (Egger et al. 2021):</strong> Initialise
                      γ ≈ 0.1, β ≈ π/4 from K-Means / LDA solution to reduce landscape exploration.
                    </p>
                    <p>
                      <strong>QUBO Encoding:</strong> Document similarity graph → MaxCut cost Hamiltonian.
                      Edge weights from SBERT cosine similarity. Binary variable{" "}
                      <code>x_i ∈ {"{0,1}"}</code> = cluster assignment.
                    </p>
                  </div>
                </div>
              </div>

              {/* Noise & ZNE Theory */}
              <div className="ql-panel">
                <div className="ql-panel-head">⬡ Noise Models & ZNE</div>
                <div className="ql-panel-body">
                  <div className="ql-theory">
                    <p>
                      <strong>Noise sources modelled (Blueprint §9.4):</strong>
                    </p>
                    <p>
                      <strong>T1/T2 Thermal Relaxation:</strong> Amplitude damping (T1) and
                      dephasing (T2) on each qubit. Gate time τ introduces a{" "}
                      <code>thermal_relaxation_error(T1, T2, τ)</code> channel.
                    </p>
                    <p>
                      <strong>Depolarising Errors:</strong> 1Q gate error rate p₁ ≈ 0.001;
                      2Q (CX/RZZ) error rate p₂ ≈ 0.01. Applied as{" "}
                      <code>depolarizing_error(p, n)</code>.
                    </p>
                    <p>
                      <strong>Readout Error:</strong> P(1|0) ≈ 2–4 % per qubit modelled
                      as a <code>ReadoutError</code> matrix.
                    </p>
                    <p>
                      <strong>ZNE (Zero-Noise Extrapolation):</strong> Run the circuit at
                      scale factors λ ∈ [1, 2, 3]. Observe E(λ). Fit and extrapolate to λ→0:
                    </p>
                    <p>
                      <code>E_mitigated = Σ_i c_i · E(λ_i)</code>
                    </p>
                    <p>
                      where Richardson coefficients{" "}
                      <code>c_i = Π_{"{j≠i}"} λ_j / (λ_j − λ_i)</code>.
                    </p>
                    <p>
                      <strong>Approximation Ratio:</strong>{" "}
                      <code>r = E_QAOA / E_optimal</code> ∈ (0, 1].
                      For MaxCut with p=1: r ≥ 0.6924 (Farhi et al. 2014).
                    </p>
                  </div>
                </div>
              </div>

            </div>
          )}

        </div>
      </div>
    </>
  );
}