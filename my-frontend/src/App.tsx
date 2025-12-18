// App.tsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion, useSpring } from 'framer-motion';

const activityNames: Record<number, string> = {
  1: 'Standing still',
  2: 'Sitting and relaxing',
  3: 'Lying down',
  4: 'Walking',
  5: 'Climbing stairs',
  6: 'Waist bends forward',
  7: 'Frontal elevation of arms',
  8: 'Knees bending (crouching)',
  9: 'Cycling',
  10: 'Jogging',
  11: 'Running',
  12: 'Jump front & back',
};

type ApiResponse = {
  status: string;
  count: number;
  prediction: {
    label: number | null;
    confidence: number | null;
    distribution: Record<string, number>;
  };
  metadata: { input_type: string };
  timeline_bars?: { label: number; start_sec: number; end_sec: number; confidence?: number }[];
  timeline?: { label: number; start_sec: number; end_sec: number; confidence?: number }[];
  percentage_by_activity?: Record<string, number>;
  purity?: number;
  duration_seconds?: Record<string, number>;
  confidence_by_activity?: Record<string, number>;
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ApiResponse | null>(null);
  const [useJson, setUseJson] = useState(false);
  const [jsonRows, setJsonRows] = useState<string>('');
  const [featureNames, setFeatureNames] = useState<string>('');
  const [activeNav, setActiveNav] = useState<'cargar' | 'resultados'>('cargar');
  const cargarRef = useRef<HTMLDivElement | null>(null);
  const resultadosRef = useRef<HTMLDivElement | null>(null);

  const scrollTo = (el: HTMLElement | null) => {
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const handleSend = async () => {
    setError(null);
    setData(null);
    if (!useJson && !file) {
      setError('Seleccione un archivo .log primero');
      return;
    }
    setLoading(true);
    try {
      let res: Response;
      if (useJson) {
        const rows = JSON.parse(jsonRows) as number[][];
        const names = featureNames.split(',').map((s) => s.trim()).filter(Boolean);
        const payload: any = { rows };
        if (names.length) payload.feature_names = names;
        res = await fetch('http://localhost:8000/predict/json', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      } else {
        const fd = new FormData();
        if (file) fd.append('file', file);
        res = await fetch('http://localhost:8000/predict/file', { method: 'POST', body: fd });
      }
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Error en el servidor');
      setData(json);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const distributionEntries = data ? Object.entries(data.prediction?.distribution || {}) : [];
  const distributionEntriesSorted = useMemo(() => distributionEntries.sort((a, b) => b[1] - a[1]), [distributionEntries]);
  const timelineBars = (data?.timeline_bars as { label: number; start_sec: number; end_sec: number; confidence?: number; purity?: number }[] | undefined) ||
    (data?.timeline as { label: number; start_sec: number; end_sec: number; confidence?: number; purity?: number }[] | undefined) ||
    [];
  const percentages = data?.percentage_by_activity as Record<string, number> | undefined;
  const durations = data?.duration_seconds as Record<string, number> | undefined;
  const confidenceByActivity = data?.confidence_by_activity as Record<string, number> | undefined;

  // Compute global purity (duration-weighted across segments)
  const globalPurity = useMemo(() => {
    if (!timelineBars || timelineBars.length === 0) return null as number | null;
    let wsum = 0;
    let dsum = 0;
    for (const s of timelineBars) {
      const dur = Math.max(0, (s.end_sec - s.start_sec));
      if (typeof s.purity === 'number') {
        wsum += dur * (s.purity as number);
        dsum += dur;
      }
    }
    return dsum > 0 ? (wsum / dsum) : null;
  }, [timelineBars]);

  return (
    <div className="flex h-screen w-full bg-[#f5f6f8] text-[#1e293b]">
      <aside className="w-64 bg-[#0b1220] text-slate-200 flex flex-col border-r border-slate-700/40">
        <div className="px-6 py-6 text-lg font-semibold tracking-wide border-b border-slate-700/50">
          MHealth Dashboard
        </div>
        <nav className="flex-1 px-4 py-6 text-sm space-y-2">
          <button
            className={(activeNav === 'cargar' ? 'bg-slate-700/40 text-slate-100 ' : 'hover:bg-slate-700/40 ') + 'w-full text-left px-3 py-2 rounded-lg transition'}
            onClick={() => { setActiveNav('cargar'); scrollTo(cargarRef.current); }}
          >Cargar Datos</button>
          <button
            className={(activeNav === 'resultados' ? 'bg-slate-700/40 text-slate-100 ' : 'hover:bg-slate-700/40 ') + 'w-full text-left px-3 py-2 rounded-lg transition'}
            onClick={() => { setActiveNav('resultados'); scrollTo(resultadosRef.current); }}
          >Resultados</button>
        </nav>
        <div className="px-6 py-4 border-t border-slate-700/40 text-xs text-slate-400">v1.0 — Sistema Clínico</div>
      </aside>

      <div className="flex-1 flex flex-col overflow-y-auto">
        <header className="h-16 bg-gradient-to-r from-indigo-600 via-sky-600 to-cyan-500 text-white flex items-center justify-between px-6 shadow">
          <h1 className="text-xl font-semibold tracking-tight">Reconocimiento de Actividad Humana</h1>
          <span className="text-white/90 text-sm">Módulo de predicción — Sensores MHealth</span>
        </header>

        <main className="p-6 space-y-6 max-w-7xl mx-auto w-full">
          <section ref={cargarRef} className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-[#0f172a]">Carga de datos</h2>
                <p className="text-slate-500 text-sm">Seleccione un archivo .log o introduzca datos JSON.</p>
              </div>
              <div className="flex items-center bg-slate-100 rounded-full p-1 text-sm">
                <button className={!useJson ? 'px-3 py-1 rounded-full bg-white shadow-sm border border-slate-200' : 'px-3 py-1 text-slate-600'} onClick={() => setUseJson(false)}>Archivo .log</button>
                <button className={useJson ? 'px-3 py-1 rounded-full bg-white shadow-sm border border-slate-200' : 'px-3 py-1 text-slate-600'} onClick={() => setUseJson(true)}>JSON</button>
              </div>
            </div>

            {!useJson ? (
              <div className="border-2 border-dashed border-slate-300 rounded-2xl p-6 bg-slate-50 hover:bg-slate-100 transition">
                <div className="flex items-center gap-4">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-indigo-100 text-indigo-600">📄</div>
                  <input type="file" accept=".log" className="text-sm" onChange={e => setFile(e.target.files?.[0] || null)} />
                  <span className="text-sm text-slate-700 truncate">{file?.name || 'Arrastre o seleccione un archivo .log'}</span>
                  <button onClick={handleSend} disabled={loading || !file} className="ml-auto px-5 py-2 rounded-lg bg-gradient-to-r from-indigo-600 to-sky-600 text-white shadow hover:opacity-95 disabled:opacity-60 transition">
                    {loading ? 'Procesando…' : 'Analizar actividad'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <textarea value={jsonRows} onChange={e => setJsonRows(e.target.value)} placeholder="Pegue JSON para rows (e.g., [[0.1,0.2,...],[...]])" className="w-full h-28 border border-slate-300 rounded-md p-3 font-mono text-xs" />
                <input value={featureNames} onChange={e => setFeatureNames(e.target.value)} placeholder="feature_names (coma separadas, opcional)" className="w-full border border-slate-300 rounded-md p-3 text-sm" />
                <button onClick={handleSend} disabled={loading} className="px-5 py-2 rounded-lg bg-gradient-to-r from-indigo-600 to-sky-600 text-white shadow hover:opacity-95 transition">
                  {loading ? 'Procesando…' : 'Analizar actividad'}
                </button>
              </div>
            )}

            {error && (
              <div className="bg-rose-50 text-rose-700 border border-rose-200 rounded-md px-3 py-2 text-sm">{error}</div>
            )}
            {data && (
              <div className="bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-md px-3 py-2 text-sm">Análisis completado</div>
            )}
          </section>

          {data && (
            <section ref={resultadosRef} className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-6">
              <h2 className="text-lg font-semibold text-[#0f172a]">Resultados de predicción</h2>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card label="Total de muestras" value={data.count} icon="📦" />
                <Card label="Actividades detectadas" value={distributionEntriesSorted.length} icon="🏷️" />
                <Card label="Actividad principal" value={activityNames[Number(data?.prediction?.label ?? 1)]} icon="⭐" />
                {globalPurity != null && (
                  <Card label="Pureza global" value={`${(globalPurity * 100).toFixed(1)}%`} icon="🧪" />
                )}
              </div>

              <div className="space-y-4">
                <h3 className="text-sm text-slate-700 font-medium">Línea de tiempo (ventanas compactadas)</h3>
                <TimelineView timeline={timelineBars} />
                <table className="w-full text-sm border border-slate-200 rounded-lg overflow-hidden">
                  <thead className="bg-[#0f172a] text-white text-xs">
                    <tr>
                      <th className="p-3 text-left">Actividad</th>
                      <th className="p-3 text-left">Duración total (s)</th>
                      <th className="p-3 text-left">Confianza promedio</th>
                      <th className="p-3 text-left">Pureza promedio</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">{Object.entries(durations || {}).sort((a, b) => Number(b[1]) - Number(a[1])).map(([k, v]) => (
                    <tr key={k} className="hover:bg-slate-50">
                      <td className="p-3">{activityNames[Number(k)]}</td>
                      <td className="p-3">{Number(v).toFixed(2)}</td>
                      <td className="p-3">{((((confidenceByActivity || {})[k] || 0) * 100)).toFixed(1)}%</td>
                      <td className="p-3">{(() => {
                        const segs = timelineBars.filter(s => String(s.label) === String(k));
                        const vals = segs.map(s => s.purity).filter(p => typeof p === 'number');
                        const avg = vals.length ? (vals.reduce((a, b) => a + (b as number), 0) / vals.length) : null;
                        return avg != null ? `${(avg * 100).toFixed(1)}%` : 'N/A';
                      })()}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </section>
          )}

        </main>

      </div>
    </div>
  );
}

/* ---------- Card component ---------- */
function Card({ label, value, icon }: { label: string; value: any; icon?: string }) {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-500">{label}</div>
        {icon && <div className="text-lg">{icon}</div>}
      </div>
      <div className="text-2xl font-semibold text-[#0f172a] mt-1">{value}</div>
    </div>
  );
}

/* ---------- colors ---------- */
const colors: Record<number, string> = {
  1: '#60a5fa', 2: '#34d399', 3: '#fbbf24', 4: '#f97316', 5: '#a78bfa', 6: '#ef4444',
  7: '#22d3ee', 8: '#f472b6', 9: '#84cc16', 10: '#c084fc', 11: '#fb7185', 12: '#14b8a6'
};

/* ---------- TimelineView v2 ---------- */
function TimelineView({ timeline }: { timeline: { label: number; start_sec: number; end_sec: number; confidence?: number; purity?: number }[] }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(800);
  const h = 56;
  const [hover, setHover] = useState<{ x: number; y: number; label: number; duration: number; confidence?: number; purity?: number } | null>(null);
  const [view, setView] = useState<{ start: number; end: number } | null>(null);
  const [cursorX, setCursorX] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [lastDragX, setLastDragX] = useState<number | null>(null);
  const rAF = useRef<number | null>(null);

  // Resize observer to adapt width
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setW(Math.max(320, Math.floor(entry.contentRect.width)));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // compact consecutive segments same label (duration-weighted confidence)
  const compact = useMemo(() => {
    if (!timeline || timeline.length === 0) return [] as { label: number; start_sec: number; end_sec: number; confidence?: number; purity?: number }[];
    const sorted = [...timeline].sort((a, b) => a.start_sec - b.start_sec);
    const out: { label: number; start_sec: number; end_sec: number; confidence?: number; purity?: number }[] = [];
    let cur = { ...sorted[0] };
    let wsumConf = (cur.end_sec - cur.start_sec) * (cur.confidence ?? 0);
    let wsumPur = (cur.end_sec - cur.start_sec) * (cur.purity ?? 1);
    let dsum = (cur.end_sec - cur.start_sec);
    for (let i = 1; i < sorted.length; i++) {
      const s = sorted[i];
      if (s.label === cur.label && Math.abs(s.start_sec - cur.end_sec) < 1e-6) {
        cur.end_sec = s.end_sec;
        wsumConf += (s.end_sec - s.start_sec) * (s.confidence ?? 0);
        wsumPur += (s.end_sec - s.start_sec) * (s.purity ?? 1);
        dsum += (s.end_sec - s.start_sec);
      } else {
        cur.confidence = dsum > 0 ? (wsumConf / dsum) : cur.confidence;
        cur.purity = dsum > 0 ? (wsumPur / dsum) : cur.purity;
        out.push(cur);
        cur = { ...s };
        wsumConf = (s.end_sec - s.start_sec) * (s.confidence ?? 0);
        wsumPur = (s.end_sec - s.start_sec) * (s.purity ?? 1);
        dsum = (s.end_sec - s.start_sec);
      }
    }
    cur.confidence = dsum > 0 ? (wsumConf / dsum) : cur.confidence;
    cur.purity = dsum > 0 ? (wsumPur / dsum) : cur.purity;
    out.push(cur);
    return out;
  }, [timeline]);

  // clamp micro gaps
  const gapClamp = useMemo(() => {
    if (compact.length === 0) return [] as { label: number; start_sec: number; end_sec: number; confidence?: number; purity?: number }[];
    const out: { label: number; start_sec: number; end_sec: number; confidence?: number; purity?: number }[] = [];
    for (let i = 0; i < compact.length; i++) {
      const prev = out[out.length - 1];
      const cur = { ...compact[i] };
      if (prev) {
        const gap = cur.start_sec - prev.end_sec;
        if (gap > 0 && gap < 1e-3) cur.start_sec = prev.end_sec;
      }
      out.push(cur);
    }
    return out;
  }, [compact]);

  const minStart = gapClamp.length ? Math.min(...gapClamp.map(t => t.start_sec)) : 0;
  const maxEnd = gapClamp.length ? Math.max(...gapClamp.map(t => t.end_sec)) : 1;
  const total = Math.max(0.001, maxEnd - minStart);

  const viewStart = view ? Math.max(minStart, Math.min(view.start, maxEnd)) : minStart;
  const viewEnd = view ? Math.max(viewStart + 0.001, Math.min(view.end, maxEnd)) : maxEnd;
  const viewSpan = Math.max(0.001, viewEnd - viewStart);

  // playhead smoothing
  const playheadX = useSpring(cursorX ?? -1000, { stiffness: 170, damping: 20 });

  // WHEEL: zoom towards cursor or pan when not ctrl
  const onWheel: React.WheelEventHandler<HTMLDivElement> = (evt) => {
    evt.preventDefault();
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = evt.clientX - rect.left;
    const innerW = Math.max(1, w - 48);
    const px = (mx - 24) / innerW;
    const tAtCursor = viewStart + px * viewSpan;

    // If shift pressed, horizontal scroll-like => pan
    if (evt.shiftKey && Math.abs(evt.deltaY) > 0) {
      const pixelsToSeconds = viewSpan / innerW;
      const deltaSec = evt.deltaY * pixelsToSeconds;
      const newStart = Math.max(minStart, Math.min(maxEnd - viewSpan, viewStart + deltaSec));
      setView({ start: newStart, end: newStart + viewSpan });
      return;
    }

    // Otherwise zoom
    const factor = evt.deltaY < 0 ? 0.85 : 1.15;
    const newSpan = Math.min(Math.max(viewSpan * factor, 0.05), total);
    const newStart = tAtCursor - px * newSpan;
    const newEnd = newStart + newSpan;
    setView({ start: Math.max(minStart, newStart), end: Math.min(maxEnd, newEnd) });
  };

  // Drag to pan
  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as Element).setPointerCapture?.(e.pointerId);
    setDragging(true);
    setLastDragX(e.clientX);
  };
  const onPointerUp = (e?: React.PointerEvent) => {
    setDragging(false);
    setLastDragX(null);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging || lastDragX == null) return;
    const dx = e.clientX - lastDragX;
    setLastDragX(e.clientX);
    const innerW = Math.max(1, w - 48);
    const deltaSec = -(dx / innerW) * viewSpan;
    const newStart = Math.max(minStart, Math.min(maxEnd - viewSpan, viewStart + deltaSec));
    setView({ start: newStart, end: newStart + viewSpan });
  };

  // Mouse move for playhead & hover
  const onMouseMove = (e: React.MouseEvent) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const localX = e.clientX - rect.left;
    // Use rAF to avoid flooding React state updates
    if (rAF.current) cancelAnimationFrame(rAF.current);
    rAF.current = requestAnimationFrame(() => setCursorX(localX));
  };
  const onMouseLeave = () => {
    setCursorX(null);
    setHover(null);
  };

  // convert cursorX to time helper
  const cursorTime = (cx: number | null) => {
    if (cx == null) return null;
    const innerW = Math.max(1, w - 48);
    const p = (cx - 24) / innerW;
    return viewStart + p * viewSpan;
  };

  // minimap click/drag to change view
  const miniRef = useRef<HTMLDivElement | null>(null);
  const onMiniClick = (e: React.MouseEvent) => {
    const rect = miniRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = (e.clientX - rect.left) / rect.width;
    const mid = minStart + px * total;
    const span = viewSpan;
    const newStart = Math.max(minStart, Math.min(maxEnd - span, mid - span / 2));
    setView({ start: newStart, end: newStart + span });
  };

  // generate small gradients ids
  const gradId = (i: number) => `g_${i}_${Math.abs((gapClamp[i]?.label || 0))}`;

  return (
    <div ref={ref} className="w-full" onWheel={onWheel}>
      <div className="flex items-center justify-between mb-2 gap-2 text-xs">
        <div className="flex gap-2">
          <button className="px-3 py-1 rounded-lg border bg-white hover:bg-slate-50 shadow-sm" onClick={() => {
            const mid = (viewStart + viewEnd) / 2;
            const newSpan = Math.max(0.001, viewSpan / 2);
            const newStart = Math.max(minStart, Math.min(maxEnd - newSpan, mid - newSpan / 2));
            setView({ start: newStart, end: newStart + newSpan });
          }}>Zoom +</button>
          <button className="px-3 py-1 rounded-lg border bg-white hover:bg-slate-50 shadow-sm" onClick={() => {
            const mid = (viewStart + viewEnd) / 2;
            const newSpan = Math.min(total, viewSpan * 2);
            const newStart = Math.max(minStart, Math.min(maxEnd - newSpan, mid - newSpan / 2));
            setView({ start: newStart, end: newStart + newSpan });
          }}>Zoom −</button>
          <button className="px-3 py-1 rounded-lg border bg-white hover:bg-slate-50 shadow-sm" onClick={() => setView(null)}>Reset</button>
        </div>
        <div className="text-slate-500 text-xs">Vista: {viewStart.toFixed(2)}s → {viewEnd.toFixed(2)}s</div>
      </div>

      <svg
        width={w}
        height={h}
        className="rounded-lg border border-slate-200 shadow-sm bg-white"
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerMove={(e: any) => { onPointerMove(e); onMouseMove(e); }}
        onMouseLeave={onMouseLeave}
      >
        {/* defs for gradients */}
        <defs>
          {gapClamp.map((s, i) => {
            const pur = typeof s.purity === 'number' ? s.purity as number : 1;
            const o1 = 0.55 + 0.45 * pur;
            const o2 = 0.35 + 0.45 * pur;
            return (
              <linearGradient id={gradId(i)} key={i}>
                <stop offset="0%" stopColor={colors[s.label] || '#3b82f6'} stopOpacity={o1} />
                <stop offset="100%" stopColor={colors[s.label] || '#3b82f6'} stopOpacity={o2} />
              </linearGradient>
            );
          })}
        </defs>

        {/* subtle backdrop */}
        <rect x={0} y={0} width={w} height={h} fill="#ffffff" />

        {/* time background banding */}
        {[0, 1, 2].map(i => (
          <rect key={i} x={24} y={8 + i * ((h - 16) / 3)} width={w - 48} height={(h - 16) / 3} fill={i % 2 === 0 ? '#fbfdff' : '#ffffff'} />
        ))}

        {/* render segments */}
        {gapClamp.map((t, i) => {
          // project to view window
          const segStart = Math.max(viewStart, Math.min(t.start_sec, viewEnd));
          const segEnd = Math.max(viewStart, Math.min(t.end_sec, viewEnd));
          if (segEnd <= viewStart || segStart >= viewEnd) return null;
          const innerW = Math.max(1, w - 48);
          const x = 24 + ((segStart - viewStart) / viewSpan) * innerW;
          const width = Math.max(2, ((segEnd - segStart) / viewSpan) * innerW);
          return (
            <g key={i}>
              <rect
                x={x}
                y={10}
                width={width}
                height={h - 26}
                fill={`url(#${gradId(i)})`}
                stroke="#0f172a10"
                    onMouseEnter={(e) => setHover({ x: (e as any).clientX, y: (e as any).clientY, label: t.label, duration: t.end_sec - t.start_sec, confidence: t.confidence, purity: t.purity })}
                onMouseLeave={() => setHover(null)}
              />
              {/* inner highlight */}
              <rect x={x} y={10} width={Math.max(2, width * 0.18)} height={3} fill="#ffffff40" pointerEvents="none" />
            </g>
          );
        })}

        {/* tick lines + labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
          const innerW = Math.max(1, w - 48);
          const x = 24 + p * innerW;
          const tsec = (viewStart + p * viewSpan).toFixed(2);
          return (
            <g key={i}>
              <line x1={x} y1={h - 10} x2={x} y2={h - 4} stroke="#e6eef8" />
              <text x={x} y={h - 12} fontSize={11} fill="#64748b" textAnchor="middle">{tsec}s</text>
            </g>
          );
        })}

        {/* playhead (follows cursorX) */}
        {cursorX != null && cursorX >= 24 && cursorX <= w - 24 && (
          <motion.g style={{ translateX: playheadX }} pointerEvents="none">
            <line x1={cursorX} y1={8} x2={cursorX} y2={h - 8} stroke="#0ea5e9" strokeDasharray="4 3" strokeWidth={1.2} />
            <rect x={cursorX - 28} y={6} width={56} height={18} fill="#0ea5e9" />
            <text x={cursorX} y={19} fontSize={11} fill="#fff" textAnchor="middle">
              {(() => {
                const t = cursorTime(cursorX);
                return t == null ? '' : `${t.toFixed(2)}s`;
              })()}
            </text>
          </motion.g>
        )}
      </svg>

      {/* tooltip */}
      {hover && (
        <div style={{ position: 'fixed', left: hover.x + 12, top: hover.y + 12, zIndex: 60 }} className="pointer-events-none bg-white/95 backdrop-blur-sm border border-slate-200 shadow-lg rounded-md px-3 py-2 text-xs text-[#0f172a]">
          <div className="font-semibold flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: colors[hover.label] || '#3b82f6' }} />
            {activityNames[hover.label]}
          </div>
          <div className="mt-1 text-slate-600">Duración: <span className="font-medium text-[#0f172a]">{hover.duration.toFixed(2)}s</span></div>
          <div className="text-slate-600">Confianza: <span className="font-medium text-[#0f172a]">{hover.confidence != null ? (hover.confidence * 100).toFixed(1) + '%' : 'N/A'}</span></div>
          <div className="text-slate-600">Pureza: <span className="font-medium text-[#0f172a]">{hover.purity != null ? (hover.purity * 100).toFixed(1) + '%' : 'N/A'}</span></div>
        </div>
      )}

      {/* legend */}
      <div className="flex flex-wrap gap-2 mt-3">
        {[...new Set(compact.map(s => s.label))].map((lbl) => (
          <span key={lbl} className="inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full border border-slate-200 bg-white shadow-sm">
            <span style={{ backgroundColor: colors[lbl] || '#3b82f6' }} className="inline-block w-3 h-3 rounded-full" />
            {activityNames[lbl]}
          </span>
        ))}
      </div>

      {/* minimap */}
      <div className="mt-3">
        <div ref={miniRef} onClick={onMiniClick} className="h-10 rounded-lg border border-slate-200 bg-white cursor-pointer relative" style={{ userSelect: 'none' }}>
          {/* minimap bars */}
          <svg width="100%" height="40" viewBox={`0 0 ${Math.max(320, w)} 40`} preserveAspectRatio="none" style={{ display: 'block' }}>
            <rect x={0} y={0} width="100%" height={40} fill="#fff" />
            {gapClamp.map((s, i) => {
              const x = ((s.start_sec - minStart) / total) * Math.max(320, w);
              const width = Math.max(1, ((s.end_sec - s.start_sec) / total) * Math.max(320, w));
              return <rect key={i} x={x} y={8} width={width} height={24} fill={colors[s.label] || '#3b82f6'} opacity={0.9} />;
            })}
            {/* view window */}
            <rect x={((viewStart - minStart) / total) * Math.max(320, w)} y={4} width={(viewSpan / total) * Math.max(320, w)} height={32} fill="#0f172a10" stroke="#0f172a30" />
          </svg>
        </div>
      </div>
    </div>
  );
}
