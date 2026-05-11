/**
 * App.jsx — HQC Topic Model Research Console
 * Root component: global styles, router, layout shell, lazy page loading.
 * Blueprint v1.0 compliant — routes map to §4 API layers.
 */

import { Suspense, lazy } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Navbar from "./components/Layout/Navbar";
import Dashboard from "./pages/Dashboard";

// ── Lazy-load heavier pages ────────────────────────────────────────────────────
const Pipeline   = lazy(() => import("./pages/Pipeline"));
const Results    = lazy(() => import("./pages/Results"));
const QuantumLab = lazy(() => import("./pages/QuantumLab"));

// ── Page Loading Fallback ──────────────────────────────────────────────────────
function PageLoader() {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "calc(100vh - 60px)",
      gap: "1rem",
      background: "#06080e",
      fontFamily: "'Share Tech Mono', monospace",
    }}>
      <div style={{
        fontSize: "2rem",
        color: "rgba(0,230,170,0.4)",
        animation: "appSpin 1.2s linear infinite",
      }}>◈</div>
      <div style={{
        fontSize: "0.7rem",
        letterSpacing: "0.2em",
        color: "rgba(0,230,170,0.25)",
        textTransform: "uppercase",
      }}>
        Loading module…
      </div>
    </div>
  );
}

// ── Not Found Page ─────────────────────────────────────────────────────────────
function NotFound() {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "calc(100vh - 60px)",
      gap: "1rem",
      background: "#06080e",
      fontFamily: "'Share Tech Mono', monospace",
      textAlign: "center",
      padding: "2rem",
    }}>
      <div style={{ fontSize: "4rem", opacity: 0.15 }}>⟨ψ⟩</div>
      <div style={{
        fontSize: "0.65rem",
        letterSpacing: "0.25em",
        color: "rgba(255,68,102,0.5)",
        textTransform: "uppercase",
      }}>
        404 — Quantum State Not Found
      </div>
      <div style={{ color: "rgba(200,220,215,0.3)", fontSize: "0.8rem" }}>
        This route collapsed on measurement.
      </div>
      <a
        href="/"
        style={{
          marginTop: "1rem",
          padding: "0.5rem 1.5rem",
          border: "1px solid rgba(0,230,170,0.2)",
          borderRadius: "4px",
          color: "#00e6aa",
          textDecoration: "none",
          fontSize: "0.75rem",
          letterSpacing: "0.12em",
          fontFamily: "'Share Tech Mono', monospace",
        }}
      >
        → Return to Dashboard
      </a>
    </div>
  );
}

// ── App Root ──────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <>
      {/* ── Global Styles ── */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;800&display=swap');

        /* ── CSS Custom Properties ── */
        :root {
          --color-bg:           #06080e;
          --color-surface:      #080b12;
          --color-surface-2:    #0d1117;
          --color-border:       rgba(0, 230, 170, 0.1);
          --color-border-hover: rgba(0, 230, 170, 0.3);

          --color-primary:   #00e6aa;
          --color-secondary: #00b4ff;
          --color-warning:   #f59e0b;
          --color-danger:    #ff4466;
          --color-success:   #10ffb0;

          --color-text:     #c8dcd8;
          --color-text-dim: rgba(200, 220, 215, 0.45);
          --color-text-xs:  rgba(200, 220, 215, 0.25);

          --font-mono:    'Share Tech Mono', 'Courier New', monospace;
          --font-display: 'Exo 2', sans-serif;

          --nav-height: 60px;
          --radius: 4px;
          --radius-lg: 8px;

          --shadow-glow-green: 0 0 20px rgba(0, 230, 170, 0.15);
          --shadow-glow-blue:  0 0 20px rgba(0, 180, 255, 0.15);

          --transition-fast: 0.15s ease;
          --transition-med:  0.25s ease;
        }

        /* ── Reset & Base ── */
        *, *::before, *::after {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        html {
          scroll-behavior: smooth;
          -webkit-text-size-adjust: 100%;
        }

        body {
          background: var(--color-bg);
          color: var(--color-text);
          font-family: var(--font-mono);
          font-size: 16px;
          line-height: 1.6;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
          min-height: 100vh;
          overflow-x: hidden;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--color-bg); }
        ::-webkit-scrollbar-thumb {
          background: rgba(0, 230, 170, 0.15);
          border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(0, 230, 170, 0.3);
        }

        /* ── Selection ── */
        ::selection {
          background: rgba(0, 230, 170, 0.2);
          color: #00e6aa;
        }

        /* ── Typography helpers ── */
        h1, h2, h3, h4, h5, h6 {
          font-family: var(--font-display);
          color: var(--color-text);
          line-height: 1.2;
        }

        a { color: inherit; }

        code, pre {
          font-family: var(--font-mono);
          font-size: 0.875em;
        }

        /* ── Focus visible ── */
        :focus-visible {
          outline: 1px solid rgba(0, 230, 170, 0.5);
          outline-offset: 2px;
          border-radius: 2px;
        }

        /* ── Utility animations ── */
        @keyframes appSpin {
          to { transform: rotate(360deg); }
        }

        @keyframes appFadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        @keyframes appPulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.5; }
        }

        /* ── Page entry animation ── */
        .page-enter {
          animation: appFadeIn 0.3s ease forwards;
        }

        /* ── Common layout wrappers ── */
        .page-container {
          min-height: calc(100vh - var(--nav-height));
          padding: calc(var(--nav-height) + 2rem) 1.5rem 3rem;
          max-width: 1400px;
          margin: 0 auto;
        }

        /* ── Common panel styles (reusable across pages) ── */
        .panel {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          overflow: hidden;
        }

        .panel-header {
          padding: 1rem 1.5rem;
          border-bottom: 1px solid var(--color-border);
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .panel-title {
          font-size: 0.7rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: rgba(0, 230, 170, 0.6);
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        /* ── Common form elements ── */
        .hqc-input {
          width: 100%;
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          color: var(--color-text);
          padding: 0.6rem 0.75rem;
          font-family: var(--font-mono);
          font-size: 0.8rem;
          outline: none;
          transition: border-color var(--transition-fast);
        }

        .hqc-input:focus {
          border-color: var(--color-border-hover);
        }

        .hqc-input::placeholder {
          color: var(--color-text-dim);
        }

        /* ── Button variants ── */
        .btn-primary {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.55rem 1.25rem;
          background: rgba(0, 230, 170, 0.1);
          border: 1px solid rgba(0, 230, 170, 0.3);
          border-radius: var(--radius);
          color: var(--color-primary);
          font-family: var(--font-mono);
          font-size: 0.78rem;
          letter-spacing: 0.1em;
          cursor: pointer;
          text-transform: uppercase;
          transition: all var(--transition-fast);
        }

        .btn-primary:hover {
          background: rgba(0, 230, 170, 0.16);
          border-color: rgba(0, 230, 170, 0.5);
          box-shadow: var(--shadow-glow-green);
        }

        .btn-primary:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .btn-secondary {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          background: transparent;
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          color: var(--color-text-dim);
          font-family: var(--font-mono);
          font-size: 0.75rem;
          letter-spacing: 0.08em;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .btn-secondary:hover {
          border-color: var(--color-border-hover);
          color: var(--color-text);
        }

        /* ── Badge ── */
        .badge {
          display: inline-flex;
          align-items: center;
          gap: 0.3rem;
          padding: 0.2rem 0.55rem;
          border-radius: 3px;
          font-size: 0.62rem;
          letter-spacing: 0.12em;
          font-weight: 600;
          text-transform: uppercase;
        }

        .badge-success {
          background: rgba(16, 255, 176, 0.08);
          color: var(--color-success);
          border: 1px solid rgba(16, 255, 176, 0.18);
        }

        .badge-warning {
          background: rgba(245, 158, 11, 0.08);
          color: var(--color-warning);
          border: 1px solid rgba(245, 158, 11, 0.18);
        }

        .badge-danger {
          background: rgba(255, 68, 102, 0.08);
          color: var(--color-danger);
          border: 1px solid rgba(255, 68, 102, 0.18);
        }

        .badge-info {
          background: rgba(0, 180, 255, 0.08);
          color: var(--color-secondary);
          border: 1px solid rgba(0, 180, 255, 0.18);
        }

        /* ── Tooltip ── */
        [data-tooltip] {
          position: relative;
          cursor: default;
        }

        [data-tooltip]::after {
          content: attr(data-tooltip);
          position: absolute;
          bottom: calc(100% + 6px);
          left: 50%;
          transform: translateX(-50%);
          background: #111827;
          border: 1px solid var(--color-border);
          border-radius: 3px;
          padding: 0.3rem 0.6rem;
          font-size: 0.65rem;
          white-space: nowrap;
          opacity: 0;
          pointer-events: none;
          transition: opacity var(--transition-fast);
          z-index: 999;
          letter-spacing: 0.06em;
        }

        [data-tooltip]:hover::after { opacity: 1; }

        /* ── Grid helpers ── */
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; }

        @media (max-width: 900px) {
          .grid-2, .grid-3 { grid-template-columns: 1fr; }
        }

        /* ── Monospace data display ── */
        .mono {
          font-family: var(--font-mono);
          letter-spacing: 0.05em;
        }

        /* ── Divider ── */
        .divider {
          height: 1px;
          background: var(--color-border);
          margin: 1.5rem 0;
        }
      `}</style>

      {/* ── Router Shell ── */}
      <BrowserRouter>
        <Navbar />
        <main>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/"          element={<Dashboard />} />
              <Route path="/pipeline"  element={<Pipeline />} />
              <Route path="/results"   element={<Results />} />
              <Route path="/quantum"   element={<QuantumLab />} />
              <Route path="/404"       element={<NotFound />} />
              <Route path="*"          element={<Navigate to="/404" replace />} />
            </Routes>
          </Suspense>
        </main>
      </BrowserRouter>
    </>
  );
}
