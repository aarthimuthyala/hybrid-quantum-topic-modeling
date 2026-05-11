/**
 * TopicViewer — Topic Word Visualisation Component
 * Renders discovered topics as horizontal word-weight bars with topic navigation.
 * Accepts topic data from §4.2 (LDA/NMF) and §4.4 (Hybrid pipeline) responses.
 *
 * Expected props:
 *   topics: Array<{ topic_id, label, words: Array<{ text, weight }> }>
 *   modelType?: "LDA" | "NMF" | "Hybrid"
 *   numTopics?: number
 *   loading?: boolean
 */

import { useState, useMemo } from "react";

// ─── Gradient palette per topic index ────────────────────────────────────────
const TOPIC_GRADIENTS = [
  ["#00e6aa", "#00b4ff"],
  ["#f59e0b", "#ff6b6b"],
  ["#a78bfa", "#00b4ff"],
  ["#10ffb0", "#00e6aa"],
  ["#ff6b6b", "#f59e0b"],
  ["#00b4ff", "#a78bfa"],
  ["#f59e0b", "#10ffb0"],
  ["#ff4466", "#a78bfa"],
  ["#00e6aa", "#f59e0b"],
  ["#a78bfa", "#10ffb0"],
];

function getGradient(idx) {
  const [a, b] = TOPIC_GRADIENTS[idx % TOPIC_GRADIENTS.length];
  return `linear-gradient(90deg, ${a}, ${b})`;
}
function getColor(idx) {
  return TOPIC_GRADIENTS[idx % TOPIC_GRADIENTS.length][0];
}

// ─── Skeleton loader ──────────────────────────────────────────────────────────
function TopicSkeleton() {
  return (
    <div style={{ padding: "0 0 0.5rem" }}>
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.3rem 0" }}>
          <div className="skeleton" style={{ width: 80, height: 14, borderRadius: 3 }} />
          <div className="skeleton" style={{ flex: 1, height: 5, borderRadius: 3 }} />
          <div className="skeleton" style={{ width: 36, height: 12, borderRadius: 3 }} />
        </div>
      ))}
    </div>
  );
}

// ─── Single word bar row ──────────────────────────────────────────────────────
function WordBar({ word, weight, maxWeight, gradient, rank }) {
  const pct = maxWeight > 0 ? (weight / maxWeight) * 100 : 0;
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: "0.75rem",
      padding: "0.28rem 0",
      animation: `appFadeIn 0.3s ease ${rank * 0.04}s both`,
    }}>
      {/* Rank */}
      <span style={{
        fontSize: "0.58rem",
        color: "rgba(200,220,215,0.2)",
        width: 14,
        textAlign: "right",
        flexShrink: 0,
        letterSpacing: "0.04em",
      }}>
        {rank}
      </span>

      {/* Word label */}
      <span style={{
        fontSize: "0.75rem",
        color: "#c8dcd8",
        width: 96,
        flexShrink: 0,
        textAlign: "right",
        letterSpacing: "0.02em",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        fontFamily: "var(--font-mono)",
      }}>
        {word}
      </span>

      {/* Bar track */}
      <div style={{
        flex: 1,
        height: 6,
        background: "rgba(255,255,255,0.04)",
        borderRadius: 3,
        overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`,
          height: "100%",
          background: gradient,
          borderRadius: 3,
          transition: "width 0.55s ease",
          boxShadow: `0 0 6px rgba(0,230,170,0.25)`,
        }} />
      </div>

      {/* Weight value */}
      <span style={{
        fontSize: "0.65rem",
        color: "rgba(200,220,215,0.3)",
        minWidth: 44,
        textAlign: "right",
        letterSpacing: "0.04em",
        fontFamily: "var(--font-mono)",
        flexShrink: 0,
      }}>
        {weight < 0.001 ? weight.toExponential(2) : weight.toFixed(4)}
      </span>
    </div>
  );
}

// ─── Single topic panel ───────────────────────────────────────────────────────
function TopicPanel({ topic, gradient, color, isActive }) {
  const words = topic.words ?? [];
  const maxWeight = words.length ? Math.max(...words.map((w) => w.weight ?? w.value ?? 0)) : 1;

  if (!isActive) return null;

  return (
    <div>
      {/* Top words inline preview */}
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "0.4rem",
        marginBottom: "1.25rem",
      }}>
        {words.slice(0, 6).map((w, i) => (
          <span key={i} style={{
            padding: "0.22rem 0.6rem",
            borderRadius: "20px",
            border: `1px solid ${color}33`,
            background: `${color}0a`,
            color,
            fontSize: "0.7rem",
            letterSpacing: "0.04em",
            fontFamily: "var(--font-mono)",
          }}>
            {w.text ?? w.word}
          </span>
        ))}
      </div>

      {/* Word bars — top 10 */}
      {words.slice(0, 10).map((w, i) => (
        <WordBar
          key={w.text ?? w.word ?? i}
          word={w.text ?? w.word ?? ""}
          weight={w.weight ?? w.value ?? 0}
          maxWeight={maxWeight}
          gradient={gradient}
          rank={i + 1}
        />
      ))}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function TopicViewer({ topics = [], modelType = "LDA", numTopics, loading = false }) {
  const [activeTopic, setActiveTopic] = useState(0);
  const [view, setView]               = useState("bars"); // bars | grid

  const displayTopics = useMemo(() => {
    if (!topics?.length) return [];
    return topics.slice(0, numTopics ?? topics.length);
  }, [topics, numTopics]);

  // ── Loading state ────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={{ fontFamily: "var(--font-mono)" }}>
        <div style={{ fontSize: "0.62rem", letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(0,230,170,0.3)", marginBottom: "1rem" }}>
          <span style={{ display: "inline-block", animation: "appSpin 0.9s linear infinite", marginRight: "0.5rem" }}>◈</span>
          Computing topics…
        </div>
        <TopicSkeleton />
      </div>
    );
  }

  // ── Empty state ──────────────────────────────────────────────────────────────
  if (!displayTopics.length) {
    return (
      <div style={{ textAlign: "center", padding: "2.5rem", color: "rgba(200,220,215,0.2)", fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
        <div style={{ fontSize: "2rem", marginBottom: "0.5rem", opacity: 0.2 }}>◫</div>
        No topic data available for this run.
      </div>
    );
  }

  const currentTopic = displayTopics[activeTopic] ?? displayTopics[0];
  const gradient     = getGradient(activeTopic);
  const color        = getColor(activeTopic);

  return (
    <div style={{ fontFamily: "var(--font-mono)" }}>

      {/* ── Header row ── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span style={{
            fontSize: "0.6rem", letterSpacing: "0.18em",
            textTransform: "uppercase", color: "rgba(0,230,170,0.4)",
          }}>
            {modelType} · {displayTopics.length} Topics
          </span>
        </div>

        {/* View toggle */}
        <div style={{ display: "flex", gap: "0.25rem" }}>
          {[["bars", "≡"], ["grid", "⊞"]].map(([v, icon]) => (
            <button
              key={v}
              onClick={() => setView(v)}
              style={{
                background: view === v ? "rgba(0,230,170,0.1)" : "transparent",
                border: `1px solid ${view === v ? "rgba(0,230,170,0.3)" : "rgba(0,230,170,0.1)"}`,
                borderRadius: 3,
                color: view === v ? "#00e6aa" : "rgba(200,220,215,0.3)",
                padding: "0.25rem 0.55rem",
                cursor: "pointer",
                fontSize: "0.85rem",
                fontFamily: "var(--font-mono)",
                transition: "all 0.15s",
              }}
            >
              {icon}
            </button>
          ))}
        </div>
      </div>

      {/* ── Topic selector tabs ── */}
      <div style={{
        display: "flex",
        gap: "0.3rem",
        flexWrap: "wrap",
        marginBottom: "1.25rem",
        paddingBottom: "1rem",
        borderBottom: "1px solid rgba(0,230,170,0.06)",
      }}>
        {displayTopics.map((t, i) => {
          const tColor = getColor(i);
          const isActive = activeTopic === i;
          return (
            <button
              key={t.topic_id ?? i}
              onClick={() => setActiveTopic(i)}
              style={{
                padding: "0.3rem 0.7rem",
                borderRadius: 3,
                border: `1px solid ${isActive ? `${tColor}44` : "rgba(0,230,170,0.08)"}`,
                background: isActive ? `${tColor}12` : "transparent",
                color: isActive ? tColor : "rgba(200,220,215,0.35)",
                fontSize: "0.68rem",
                letterSpacing: "0.06em",
                cursor: "pointer",
                fontFamily: "var(--font-mono)",
                transition: "all 0.15s",
                whiteSpace: "nowrap",
              }}
            >
              {t.label ?? `Topic ${i + 1}`}
            </button>
          );
        })}
      </div>

      {/* ── Bar view ── */}
      {view === "bars" && (
        <>
          {/* Active topic header */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
            <div style={{
              width: 3, height: 32,
              background: gradient,
              borderRadius: 2,
              flexShrink: 0,
            }} />
            <div>
              <div style={{ fontSize: "0.85rem", color: "#e8f4f0", fontFamily: "var(--font-display)", fontWeight: 700 }}>
                {currentTopic.label ?? `Topic ${activeTopic + 1}`}
              </div>
              <div style={{ fontSize: "0.6rem", color: "rgba(200,220,215,0.3)", letterSpacing: "0.1em", textTransform: "uppercase", marginTop: "0.1rem" }}>
                Top {Math.min(10, currentTopic.words?.length ?? 0)} terms by weight
              </div>
            </div>
          </div>

          <TopicPanel
            topic={currentTopic}
            gradient={gradient}
            color={color}
            isActive={true}
          />
        </>
      )}

      {/* ── Grid view — all topics overview ── */}
      {view === "grid" && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: "0.75rem",
        }}>
          {displayTopics.map((t, i) => {
            const tColor = getColor(i);
            const topWords = (t.words ?? []).slice(0, 5);
            return (
              <div
                key={t.topic_id ?? i}
                onClick={() => { setActiveTopic(i); setView("bars"); }}
                style={{
                  padding: "0.85rem",
                  background: "rgba(0,0,0,0.2)",
                  border: `1px solid ${tColor}22`,
                  borderRadius: 4,
                  cursor: "pointer",
                  transition: "border-color 0.15s, background 0.15s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = `${tColor}44`;
                  e.currentTarget.style.background  = `${tColor}08`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = `${tColor}22`;
                  e.currentTarget.style.background  = "rgba(0,0,0,0.2)";
                }}
              >
                {/* Topic label */}
                <div style={{ fontSize: "0.68rem", color: tColor, letterSpacing: "0.06em", marginBottom: "0.5rem", fontFamily: "var(--font-display)", fontWeight: 600 }}>
                  {t.label ?? `Topic ${i + 1}`}
                </div>
                {/* Mini word list */}
                <div style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
                  {topWords.map((w, wi) => (
                    <div key={wi} style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                      <div style={{
                        width: `${Math.max(8, ((w.weight ?? w.value ?? 0) / (topWords[0]?.weight ?? 1)) * 50)}px`,
                        height: 3,
                        background: getGradient(i),
                        borderRadius: 2,
                        flexShrink: 0,
                      }} />
                      <span style={{ fontSize: "0.65rem", color: "rgba(200,220,215,0.55)" }}>
                        {w.text ?? w.word}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}