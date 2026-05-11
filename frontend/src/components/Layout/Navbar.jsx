/**
 * Navbar — HQC Topic Model Research Console
 * Dark science-terminal aesthetic with quantum pulse animations.
 */

import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { healthApi } from "../../services/api";

const NAV_LINKS = [
  { path: "/",           label: "Dashboard",   icon: "⬡" },
  { path: "/pipeline",   label: "Pipeline",    icon: "◈" },
  { path: "/results",    label: "Results",     icon: "◉" },
  { path: "/quantum",    label: "Quantum Lab", icon: "⟨ψ⟩" },
];

export default function Navbar() {
  const { pathname } = useLocation();
  const [apiStatus, setApiStatus] = useState("checking"); // checking | online | offline
  const [mobileOpen, setMobileOpen] = useState(false);
  const [tick, setTick] = useState(0);

  // Health check on mount + every 30s
  useEffect(() => {
    const check = async () => {
      try {
        await healthApi.check();
        setApiStatus("online");
      } catch {
        setApiStatus("offline");
      }
    };
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  // Animate the quantum clock tick
  useEffect(() => {
    const id = setInterval(() => setTick((t) => (t + 1) % 8), 800);
    return () => clearInterval(id);
  }, []);

  const statusColor = {
    checking: "#f59e0b",
    online:   "#10ffb0",
    offline:  "#ff4466",
  }[apiStatus];

  const statusLabel = {
    checking: "CONNECTING",
    online:   "API ONLINE",
    offline:  "API OFFLINE",
  }[apiStatus];

  // Quantum clock animation frames
  const clockFrames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧"];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

        .hqc-nav {
          position: fixed;
          top: 0; left: 0; right: 0;
          z-index: 1000;
          background: rgba(6, 8, 14, 0.96);
          backdrop-filter: blur(12px);
          border-bottom: 1px solid rgba(0, 230, 170, 0.12);
          font-family: 'Share Tech Mono', monospace;
        }

        .hqc-nav::before {
          content: '';
          position: absolute;
          bottom: 0; left: 0; right: 0;
          height: 1px;
          background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(0, 230, 170, 0.6) 30%,
            rgba(0, 180, 255, 0.4) 60%,
            transparent 100%
          );
          animation: scanline 4s ease-in-out infinite;
        }

        @keyframes scanline {
          0%, 100% { opacity: 0.3; transform: scaleX(0.7); }
          50% { opacity: 1; transform: scaleX(1); }
        }

        .nav-inner {
          max-width: 1400px;
          margin: 0 auto;
          padding: 0 1.5rem;
          height: 60px;
          display: flex;
          align-items: center;
          gap: 2rem;
        }

        /* ── Brand ── */
        .brand {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          text-decoration: none;
          flex-shrink: 0;
        }

        .brand-glyph {
          width: 36px; height: 36px;
          border: 1px solid rgba(0, 230, 170, 0.4);
          border-radius: 4px;
          display: flex; align-items: center; justify-content: center;
          font-size: 1.2rem;
          color: #00e6aa;
          position: relative;
          overflow: hidden;
          background: rgba(0, 230, 170, 0.04);
        }

        .brand-glyph::after {
          content: '';
          position: absolute;
          inset: 0;
          background: linear-gradient(135deg, rgba(0,230,170,0.15) 0%, transparent 60%);
          animation: glyphPulse 2s ease-in-out infinite;
        }

        @keyframes glyphPulse {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; }
        }

        .brand-text {
          display: flex;
          flex-direction: column;
          line-height: 1.1;
        }

        .brand-title {
          font-family: 'Exo 2', sans-serif;
          font-weight: 700;
          font-size: 0.9rem;
          letter-spacing: 0.08em;
          color: #e8f4f0;
          text-transform: uppercase;
        }

        .brand-sub {
          font-size: 0.62rem;
          color: rgba(0, 230, 170, 0.55);
          letter-spacing: 0.15em;
          text-transform: uppercase;
        }

        /* ── Nav Links ── */
        .nav-links {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          flex: 1;
          list-style: none;
          margin: 0;
          padding: 0;
        }

        .nav-link {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.45rem 1rem;
          border-radius: 4px;
          text-decoration: none;
          font-size: 0.8rem;
          letter-spacing: 0.08em;
          color: rgba(200, 220, 215, 0.55);
          border: 1px solid transparent;
          transition: all 0.18s ease;
          position: relative;
          white-space: nowrap;
        }

        .nav-link .link-icon {
          font-size: 1rem;
          opacity: 0.7;
          transition: opacity 0.18s;
        }

        .nav-link:hover {
          color: #00e6aa;
          background: rgba(0, 230, 170, 0.05);
          border-color: rgba(0, 230, 170, 0.18);
        }

        .nav-link:hover .link-icon { opacity: 1; }

        .nav-link.active {
          color: #00e6aa;
          background: rgba(0, 230, 170, 0.08);
          border-color: rgba(0, 230, 170, 0.28);
        }

        .nav-link.active::before {
          content: '';
          position: absolute;
          bottom: -1px; left: 20%; right: 20%;
          height: 2px;
          background: #00e6aa;
          border-radius: 2px 2px 0 0;
          box-shadow: 0 0 8px #00e6aa;
        }

        /* ── Right cluster ── */
        .nav-right {
          display: flex;
          align-items: center;
          gap: 1rem;
          margin-left: auto;
          flex-shrink: 0;
        }

        .quantum-clock {
          font-size: 0.75rem;
          color: rgba(0, 200, 255, 0.4);
          letter-spacing: 0.1em;
          user-select: none;
        }

        .status-badge {
          display: flex;
          align-items: center;
          gap: 0.45rem;
          padding: 0.3rem 0.75rem;
          border-radius: 20px;
          font-size: 0.68rem;
          letter-spacing: 0.12em;
          border: 1px solid;
          transition: all 0.3s ease;
          white-space: nowrap;
        }

        .status-dot {
          width: 6px; height: 6px;
          border-radius: 50%;
          flex-shrink: 0;
        }

        .status-dot.pulse {
          animation: statusPulse 1.5s ease-in-out infinite;
        }

        @keyframes statusPulse {
          0%, 100% { box-shadow: 0 0 0 0 currentColor; opacity: 1; }
          50% { box-shadow: 0 0 0 4px transparent; opacity: 0.6; }
        }

        /* ── Version chip ── */
        .version-chip {
          font-size: 0.63rem;
          color: rgba(150, 170, 165, 0.4);
          letter-spacing: 0.1em;
          border-left: 1px solid rgba(255,255,255,0.08);
          padding-left: 1rem;
        }

        /* ── Mobile ── */
        .mobile-toggle {
          display: none;
          background: none;
          border: 1px solid rgba(0, 230, 170, 0.2);
          color: #00e6aa;
          padding: 0.4rem 0.6rem;
          border-radius: 4px;
          cursor: pointer;
          font-size: 1rem;
          margin-left: auto;
          font-family: 'Share Tech Mono', monospace;
        }

        .mobile-menu {
          display: none;
          flex-direction: column;
          padding: 1rem 1.5rem;
          border-top: 1px solid rgba(0, 230, 170, 0.08);
          background: rgba(6, 8, 14, 0.98);
          gap: 0.25rem;
        }

        .mobile-menu.open { display: flex; }

        .mobile-link {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.6rem 0.75rem;
          border-radius: 4px;
          text-decoration: none;
          font-size: 0.82rem;
          color: rgba(200, 220, 215, 0.65);
          letter-spacing: 0.06em;
          border: 1px solid transparent;
        }

        .mobile-link.active {
          color: #00e6aa;
          background: rgba(0, 230, 170, 0.07);
          border-color: rgba(0, 230, 170, 0.2);
        }

        /* ── Responsive ── */
        @media (max-width: 900px) {
          .nav-links { display: none; }
          .nav-right { display: none; }
          .mobile-toggle { display: block; }
        }

        @media (max-width: 480px) {
          .brand-sub { display: none; }
        }
      `}</style>

      <nav className="hqc-nav">
        <div className="nav-inner">
          {/* Brand */}
          <Link to="/" className="brand">
            <div className="brand-glyph">⟨ψ⟩</div>
            <div className="brand-text">
              <span className="brand-title">HQC · TopicModel</span>
              <span className="brand-sub">v1.0 · Research Console</span>
            </div>
          </Link>

          {/* Desktop Nav Links */}
          <ul className="nav-links">
            {NAV_LINKS.map(({ path, label, icon }) => (
              <li key={path}>
                <Link
                  to={path}
                  className={`nav-link ${pathname === path ? "active" : ""}`}
                >
                  <span className="link-icon">{icon}</span>
                  {label}
                </Link>
              </li>
            ))}
          </ul>

          {/* Right cluster */}
          <div className="nav-right">
            <span className="quantum-clock">{clockFrames[tick]} QSIM</span>

            <div
              className="status-badge"
              style={{
                color: statusColor,
                borderColor: `${statusColor}33`,
                background: `${statusColor}0a`,
              }}
            >
              <div
                className={`status-dot ${apiStatus === "online" ? "pulse" : ""}`}
                style={{ background: statusColor, color: statusColor }}
              />
              {statusLabel}
            </div>

            <span className="version-chip">BLUEPRINT v1.0</span>
          </div>

          {/* Mobile toggle */}
          <button
            className="mobile-toggle"
            onClick={() => setMobileOpen((o) => !o)}
            aria-label="Toggle navigation"
          >
            {mobileOpen ? "✕" : "☰"}
          </button>
        </div>

        {/* Mobile menu */}
        <div className={`mobile-menu ${mobileOpen ? "open" : ""}`}>
          {NAV_LINKS.map(({ path, label, icon }) => (
            <Link
              key={path}
              to={path}
              className={`mobile-link ${pathname === path ? "active" : ""}`}
              onClick={() => setMobileOpen(false)}
            >
              <span>{icon}</span>
              {label}
            </Link>
          ))}
          <div style={{
            marginTop: "0.75rem",
            paddingTop: "0.75rem",
            borderTop: "1px solid rgba(0,230,170,0.1)",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            fontSize: "0.7rem",
            color: statusColor,
            letterSpacing: "0.1em",
          }}>
            <div style={{
              width: 6, height: 6,
              borderRadius: "50%",
              background: statusColor,
            }} />
            {statusLabel}
          </div>
        </div>
      </nav>
    </>
  );
}
