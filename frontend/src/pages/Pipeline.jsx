/**
 * Pipeline Page — Launch, configure, and monitor hybrid pipeline runs.
 * Connects to §4.1 (Ingest), §4.3 (Quantum), §4.4 (Hybrid) endpoints.
 */

import { useState, useEffect } from "react";
import { ingestApi, hybridApi, quantumApi, pollUntilDone } from "../services/api";

const PIPELINE_DEFAULTS = {
  corpus_id:        "20ng",
  method:           "hybrid",
  num_topics:       5,
  num_clusters:     5,
  qaoa_layers:      2,
  shots:            1024,
  noise_profile:    "depolarizing",
  enable_noise:     true,
  enable_mitigation: true,
};

export default function Pipeline() {
  const [config, setConfig]       = useState(PIPELINE_DEFAULTS);
  const [status, setStatus]       = useState("idle"); // idle|running|done|failed
  const [jobId, setJobId]         = useState(null);
  const [jobResult, setJobResult] = useState(null);
  const [log, setLog]             = useState([]);
  const [progress, setProgress]   = useState(0);

  const addLog = (msg) => setLog((l) => [...l, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const handleLaunch = async () => {
    setStatus("running");
    setLog([]);
    setJobResult(null);
    setProgress(0);
    addLog("Initiating hybrid pipeline…");

    try {
      addLog(`Loading dataset: ${config.corpus_id}`);
      const loaded = await ingestApi.loadDataset(config.corpus_id, 200, null);
      addLog(`Corpus loaded: ${loaded.doc_count ?? "?"} documents.`);
      setProgress(15);

      addLog("Launching hybrid pipeline…");
      const run = await hybridApi.run(config);
      const id  = run.run_id ?? run.job_id;
      setJobId(id);
      addLog(`Pipeline job created: ${id}`);
      setProgress(30);

      addLog("Polling job status…");
      const result = await pollUntilDone(id, (r) => {
        addLog(`Status: ${r.status} | elapsed: ${r.elapsed_s ?? "?"}s`);
        setProgress((p) => Math.min(p + 10, 90));
      }, { intervalMs: 3000, timeoutMs: 600_000 });

      setJobResult(result);
      setProgress(100);
      setStatus(result.status === "failed" ? "failed" : "done");
      addLog(`Pipeline ${result.status}.`);
    } catch (err) {
      setStatus("failed");
      addLog(`Error: ${err.message ?? "Unknown error"}`);
    }
  };

  const field = (key, label, type = "number", min, max) => (
    <div className="field-group" key={key}>
      <label className="field-label">{label}</label>
      {type === "checkbox" ? (
        <input
          type="checkbox"
          checked={config[key]}
          onChange={(e) => setConfig((c) => ({ ...c, [key]: e.target.checked }))}
          style={{ accentColor: "#00e6aa", width: 16, height: 16 }}
        />
      ) : (
        <input
          className="hqc-input"
          type={type}
          value={config[key]}
          min={min}
          max={max}
          onChange={(e) =>
            setConfig((c) => ({
              ...c,
              [key]: type === "number" ? Number(e.target.value) : e.target.value,
            }))
          }
        />
      )}
    </div>
  );

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@700;800&display=swap');

        .pipeline-page {
          min-height: calc(100vh - 60px);
          background: #06080e;
          padding: 80px 1.5rem 3rem;
          font-family: 'Share Tech Mono', monospace;
          color: #c8dcd8;
        }

        .pipeline-page::before {
          content: '';
          position: fixed;
          inset: 0;
          background-image:
            linear-gradient(rgba(0,230,170,0.012) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,230,170,0.012) 1px, transparent 1px);
          background-size: 40px 40px;
          pointer-events: none;
          z-index: 0;
        }

        .pipeline-content {
          position: relative;
          z-index: 1;
          max-width: 1200px;
          margin: 0 auto;
        }

        .pipeline-title {
          font-family: 'Exo 2', sans-serif;
          font-weight: 800;
          font-size: clamp(1.4rem, 3vw, 2rem);
          color: #e8f4f0;
          margin-bottom: 0.4rem;
        }
        .pipeline-title span { color: #00e6aa; }

        .pipeline-sub {
          font-size: 0.68rem;
          color: rgba(0,230,170,0.4);
          letter-spacing: 0.18em;
          text-transform: uppercase;
          margin-bottom: 2.5rem;
        }

        .pipeline-grid {
          display: grid;
          grid-template-columns: 380px 1fr;
          gap: 1.5rem;
        }

        @media (max-width: 900px) { .pipeline-grid { grid-template-columns: 1fr; } }

        .config-panel, .monitor-panel {
          background: #080b12;
          border: 1px solid rgba(0,230,170,0.1);
          border-radius: 6px;
          overflow: hidden;
        }

        .panel-head {
          padding: 1rem 1.5rem;
          border-bottom: 1px solid rgba(0,230,170,0.08);
          font-size: 0.7rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: rgba(0,230,170,0.6);
        }

        .config-body {
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .field-group {
          display: flex;
          flex-direction: column;
          gap: 0.4rem;
        }

        .field-label {
          font-size: 0.62rem;
          letter-spacing: 0.15em;
          text-transform: uppercase;
          color: rgba(0,230,170,0.45);
        }

        .hqc-input {
          background: #0d1117;
          border: 1px solid rgba(0,230,170,0.15);
          border-radius: 4px;
          color: #c8dcd8;
          padding: 0.55rem 0.75rem;
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.78rem;
          outline: none;
          transition: border-color 0.15s;
          width: 100%;
        }

        .hqc-input:focus { border-color: rgba(0,230,170,0.35); }

        .launch-btn {
          width: 100%;
          margin-top: 0.5rem;
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
          text-transform: uppercase;
        }

        .launch-btn:hover:not(:disabled) {
          background: linear-gradient(135deg, rgba(0,230,170,0.2), rgba(0,180,255,0.12));
          box-shadow: 0 0 20px rgba(0,230,170,0.1);
        }

        .launch-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        .progress-wrap {
          height: 3px;
          background: rgba(0,230,170,0.08);
          border-radius: 2px;
          overflow: hidden;
          margin: 1rem 1.5rem 0;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #00e6aa, #00b4ff);
          box-shadow: 0 0 8px #00e6aa;
          transition: width 0.4s ease;
          border-radius: 2px;
        }

        .log-area {
          padding: 1.25rem 1.5rem;
          height: 300px;
          overflow-y: auto;
          font-size: 0.72rem;
          line-height: 1.8;
          color: rgba(0,230,170,0.6);
          display: flex;
          flex-direction: column;
          gap: 0.1rem;
        }

        .log-line:last-child { color: #00e6aa; }

        .result-box {
          margin: 0 1.5rem 1.5rem;
          padding: 1rem;
          background: rgba(0,230,170,0.04);
          border: 1px solid rgba(0,230,170,0.12);
          border-radius: 4px;
          font-size: 0.72rem;
          color: rgba(200,220,215,0.6);
        }

        .result-metric {
          display: flex;
          justify-content: space-between;
          padding: 0.3rem 0;
          border-bottom: 1px solid rgba(0,230,170,0.05);
        }

        .result-val { color: #00e6aa; font-weight: 600; }

        .status-tag {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          padding: 0.3rem 0.75rem;
          border-radius: 3px;
          font-size: 0.65rem;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }
      `}</style>

      <div className="pipeline-page">
        <div className="pipeline-content">
          <div className="pipeline-title">Pipeline <span>Configurator</span></div>
          <div className="pipeline-sub">§4.4 Hybrid Pipeline · §9 Config Structure</div>

          <div className="pipeline-grid">
            {/* Config Panel */}
            <div className="config-panel">
              <div className="panel-head">◈ Pipeline Configuration</div>
              <div className="config-body">
                {field("corpus_id",    "Corpus ID",     "text")}
                {field("num_topics",   "Num Topics",    "number", 2, 50)}
                {field("num_clusters", "Num Clusters",  "number", 2, 20)}
                {field("qaoa_layers",  "QAOA p-layers", "number", 1, 5)}
                {field("shots",        "Quantum Shots", "number", 256, 8192)}
                <div className="field-group">
                  <label className="field-label">Noise Profile</label>
                  <select
                    className="hqc-input"
                    value={config.noise_profile}
                    onChange={(e) => setConfig((c) => ({ ...c, noise_profile: e.target.value }))}
                    style={{ cursor: "pointer", appearance: "none" }}
                  >
                    <option value="depolarizing">Depolarizing</option>
                    <option value="thermal">Thermal Relaxation</option>
                    <option value="device_fake">Device Fake (FakeManila)</option>
                  </select>
                </div>
                <div style={{ display: "flex", gap: "1.5rem" }}>
                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.72rem", color: "rgba(200,220,215,0.55)", cursor: "pointer" }}>
                    <input type="checkbox" checked={config.enable_noise} onChange={(e) => setConfig((c) => ({ ...c, enable_noise: e.target.checked }))} style={{ accentColor: "#00e6aa" }} />
                    Enable Noise
                  </label>
                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.72rem", color: "rgba(200,220,215,0.55)", cursor: "pointer" }}>
                    <input type="checkbox" checked={config.enable_mitigation} onChange={(e) => setConfig((c) => ({ ...c, enable_mitigation: e.target.checked }))} style={{ accentColor: "#00e6aa" }} />
                    ZNE Mitigation
                  </label>
                </div>
                <button
                  className="launch-btn"
                  onClick={handleLaunch}
                  disabled={status === "running"}
                >
                  {status === "running" ? "⟳ Running Pipeline…" : "▶ Launch Pipeline"}
                </button>
              </div>
            </div>

            {/* Monitor Panel */}
            <div className="monitor-panel">
              <div className="panel-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>◉ Run Monitor</span>
                {jobId && (
                  <span style={{ color: "rgba(0,180,255,0.6)", fontSize: "0.65rem" }}>
                    JOB: {jobId}
                  </span>
                )}
              </div>

              {status === "running" && (
                <div className="progress-wrap">
                  <div className="progress-fill" style={{ width: `${progress}%` }} />
                </div>
              )}

              <div className="log-area">
                {log.length === 0 ? (
                  <span style={{ color: "rgba(200,220,215,0.2)" }}>
                    Awaiting pipeline launch…
                  </span>
                ) : (
                  log.map((l, i) => (
                    <div key={i} className="log-line">{l}</div>
                  ))
                )}
              </div>

              {jobResult && status === "done" && (
                <div className="result-box">
                  {[
                    ["Silhouette Score",   jobResult.silhouette_hybrid?.toFixed(4)],
                    ["Topic Coherence Cv", jobResult.topic_coherence_cv?.toFixed(4)],
                    ["QAOA Final Cost",    jobResult.qaoa_final_cost?.toFixed(4)],
                    ["Noise TVD",          jobResult.noise_tvd?.toFixed(4)],
                  ].filter(([,v]) => v != null).map(([k, v]) => (
                    <div key={k} className="result-metric">
                      <span>{k}</span>
                      <span className="result-val">{v}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
