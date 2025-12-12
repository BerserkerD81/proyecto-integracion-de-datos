import { useEffect, useMemo, useRef, useState } from 'react'

const activityNames: Record<number, string> = {
  0: 'Sin actividad',
  1: 'Sentado',
  2: 'Acostado',
  3: 'Caminando',
  4: 'Subiendo escaleras',
  5: 'Flexión cintura',
  6: 'Elevación brazos',
  7: 'Flexión rodillas',
  8: 'Ciclismo',
  9: 'Trotando',
  10: 'Corriendo',
  11: 'Saltando',
  12: 'Otra',
}

type ApiResponse = {
  status: string
  count: number
  prediction: {
    label: number | null
    confidence: number | null
    distribution: Record<string, number>
  }
  metadata: { input_type: string }
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<ApiResponse | null>(null)
  const [useJson, setUseJson] = useState(false)
  const [jsonRows, setJsonRows] = useState<string>('')
  const [featureNames, setFeatureNames] = useState<string>('')
  const [activeActivities, setActiveActivities] = useState<Set<string>>(new Set(Object.keys(activityNames)))
  const [hoverActivity, setHoverActivity] = useState<string | null>(null)
  const [activeNav, setActiveNav] = useState<'cargar'|'resultados'>('cargar')
  const cargarRef = useRef<HTMLDivElement | null>(null)
  const resultadosRef = useRef<HTMLDivElement | null>(null)

  const scrollTo = (el: HTMLElement | null) => {
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const handleSend = async () => {
    setError(null)
    setData(null)
    if (!useJson && !file) {
      setError('Seleccione un archivo .log primero')
      return
    }
    setLoading(true)
    try {
      let res: Response
      if (useJson) {
        const rows = JSON.parse(jsonRows) as number[][]
        const names = featureNames.split(',').map(s => s.trim()).filter(Boolean)
        const payload: any = { rows }
        if (names.length) payload.feature_names = names
        res = await fetch('http://localhost:8000/predict/json', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
      } else {
        const fd = new FormData()
        if (file) fd.append('file', file)
        res = await fetch('http://localhost:8000/predict/file', { method: 'POST', body: fd })
      }
      const json = await res.json()
      if (!res.ok) throw new Error(json.detail || 'Error en el servidor')
      setData(json)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const distributionEntries = data ? Object.entries(data.prediction?.distribution || {}) : []
  const visibleDistributionEntries = useMemo(() => distributionEntries.filter(([k]) => activeActivities.has(String(k))), [distributionEntries, activeActivities])
  const distributionEntriesSorted = useMemo(() => visibleDistributionEntries.sort((a,b) => b[1]-a[1]), [visibleDistributionEntries])

  return (
    <div className="flex h-screen w-full bg-[#f5f6f8] text-[#1e293b]">

      <aside className="w-64 bg-[#0f172a] text-slate-200 flex flex-col border-r border-slate-700/40">
        <div className="px-6 py-6 text-lg font-semibold tracking-wide border-b border-slate-700/50">
          MHealth Dashboard
        </div>
        <nav className="flex-1 px-4 py-6 text-sm space-y-2">
          <button
            className={(activeNav==='cargar' ? 'bg-slate-700/40 text-slate-100 ' : 'hover:bg-slate-700/40 ') + 'w-full text-left px-3 py-2 rounded-lg transition'}
            onClick={() => { setActiveNav('cargar'); scrollTo(cargarRef.current) }}
          >Cargar Datos</button>
          <button
            className={(activeNav==='resultados' ? 'bg-slate-700/40 text-slate-100 ' : 'hover:bg-slate-700/40 ') + 'w-full text-left px-3 py-2 rounded-lg transition'}
            onClick={() => { setActiveNav('resultados'); scrollTo(resultadosRef.current) }}
          >Resultados</button>
        </nav>
        <div className="px-6 py-4 border-t border-slate-700/40 text-xs text-slate-400">v1.0 — Sistema Clínico</div>
      </aside>

      <div className="flex-1 flex flex-col overflow-y-auto">

        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 shadow-sm">
          <h1 className="text-xl font-semibold tracking-tight text-[#0f172a]">Reconocimiento de Actividad Humana</h1>
          <span className="text-slate-500 text-sm">Módulo de predicción — Sensores MHealth</span>
        </header>

        <main className="p-6 space-y-6 max-w-7xl mx-auto w-full">

          <section ref={cargarRef} className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-[#0f172a]">Carga de datos</h2>
                <p className="text-slate-500 text-sm">Seleccione un archivo .log o introduzca datos JSON.</p>
              </div>
              <div className="flex items-center bg-slate-100 rounded-lg p-1 text-sm">
                <button className={!useJson ? 'px-3 py-1 rounded-md bg-white shadow-sm border border-slate-200' : 'px-3 py-1 text-slate-600'} onClick={() => setUseJson(false)}>Archivo .log</button>
                <button className={useJson ? 'px-3 py-1 rounded-md bg-white shadow-sm border border-slate-200' : 'px-3 py-1 text-slate-600'} onClick={() => setUseJson(true)}>JSON</button>
              </div>
            </div>

            {!useJson ? (
              <div className="border-2 border-dashed border-slate-300 rounded-xl p-6 bg-slate-50 hover:bg-slate-100 transition">
                <div className="flex items-center gap-3">
                  <input type="file" accept=".log" className="text-sm" onChange={e => setFile(e.target.files?.[0] || null)} />
                  <span className="text-sm text-slate-700">{file?.name || 'Arrastre aquí o seleccione un archivo .log'}</span>
                  <button onClick={handleSend} disabled={loading || !file} className="ml-auto px-4 py-2 rounded-md bg-[#1e3a8a] text-white shadow hover:bg-[#1e40af] disabled:opacity-60 transition">
                    {loading ? 'Procesando…' : 'Analizar actividad'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <textarea value={jsonRows} onChange={e => setJsonRows(e.target.value)} placeholder="Pegue JSON para rows (e.g., [[0.1,0.2,...],[...]])" className="w-full h-28 border border-slate-300 rounded-md p-3 font-mono text-xs" />
                <input value={featureNames} onChange={e => setFeatureNames(e.target.value)} placeholder="feature_names (coma separadas, opcional)" className="w-full border border-slate-300 rounded-md p-3 text-sm" />
                <button onClick={handleSend} disabled={loading} className="px-4 py-2 rounded-md bg-[#1e3a8a] text-white shadow hover:bg-[#1e40af] transition">
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

          <section className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <h3 className="text-sm text-slate-700 mb-2 font-medium">Actividades disponibles</h3>
            <Legend activeActivities={activeActivities} onToggle={(k: string) =>{
              setActiveActivities(prev => {
                const next = new Set(prev)
                if (next.has(k)) next.delete(k)
                else next.add(k)
                return next
              })
            }} onHover={setHoverActivity} />
          </section>

          {data && (
            <section ref={resultadosRef} className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6">
              <h2 className="text-lg font-semibold text-[#0f172a]">Resultados de predicción</h2>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card label="Total de muestras" value={data.count} />
                <Card label="Actividades detectadas" value={distributionEntriesSorted.length} />
                <Card label="Actividad principal" value={activityNames[Number(Object.entries(data.prediction.distribution).sort((a,b) => b[1]-a[1])[0]?.[0] || 0)]} />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">

                <div>
                  <table className="w-full text-sm border border-slate-200 rounded-lg overflow-hidden">
                    <thead className="bg-[#0f172a] text-white text-xs">
                      <tr>
                        <th className="p-3 text-left">ID</th>
                        <th className="p-3 text-left">Actividad</th>
                        <th className="p-3 text-left">Porcentaje</th>
                        <th className="p-3 text-left">Conf. promedio</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200">{distributionEntriesSorted.map(([k, v]) => (
                      <tr key={k} className="hover:bg-slate-50">
                        <td className="p-3 font-medium">{k}</td>
                        <td className="p-3">{activityNames[Number(k)]}</td>
                        <td className="p-3">{(v*100).toFixed(1)}%</td>
                        <td className="p-3">{data.prediction.confidence != null ? (data.prediction.confidence*100).toFixed(1)+'%' : 'N/A'}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>

                <ResponsiveBarChart distribution={Object.fromEntries(visibleDistributionEntries)} highlightKey={hoverActivity} />

              </div>
            </section>
          )}

        </main>

      </div>
    </div>
  )
}

function Card({label, value}:{label:string, value:any}){
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
      <div className="text-sm text-slate-500">{label}</div>
      <div className="text-2xl font-semibold text-[#0f172a]">{value}</div>
    </div>
  )
}

function ResponsiveBarChart({ distribution, highlightKey }: { distribution: Record<string, number>, highlightKey?: string | null }) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [size, setSize] = useState<{w:number,h:number}>({ w: 720, h: 280 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const cr = entry.contentRect
        setSize({ w: Math.max(320, Math.floor(cr.width)), h: 280 })
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const entries = Object.entries(distribution)
  const maxV = Math.max(...entries.map(([,v]) => v), 0.001)
  const padding = { left: 40, right: 20, top: 20, bottom: 30 }
  const innerW = Math.max(0, size.w - padding.left - padding.right)
  const innerH = Math.max(0, size.h - padding.top - padding.bottom)
  const barW = entries.length > 0 ? innerW / entries.length : innerW
  const toX = (i: number) => padding.left + i * barW
  const toY = (v: number) => padding.top + innerH - (v / maxV) * innerH

  return (
    <div ref={ref} className="w-full">
      <svg width={size.w} height={size.h}>
        <rect x={0} y={0} width={size.w} height={size.h} fill="#ffffff" />
        {entries.map(([k, v], i) => {
          const x = toX(i) + 6
          const y = toY(v)
          const h = padding.top + innerH - y
          const isHighlight = highlightKey != null && String(k) === String(highlightKey)
          return (
            <g key={k}>
              <rect
                x={x}
                y={y}
                width={Math.max(0, barW - 12)}
                height={h}
                rx={4}
                fill={isHighlight ? '#1e3a8a' : '#3b82f6'}
                opacity={isHighlight ? 1 : 0.85}
              />
              <text
                x={x + Math.max(0, barW - 12) / 2}
                y={y - 6}
                textAnchor="middle"
                fontSize={11}
                fill="#334155"
              >{(v * 100).toFixed(0)}%</text>
              <text
                x={x + Math.max(0, barW - 12) / 2}
                y={padding.top + innerH + 4}
                transform={`rotate(-90 ${x + Math.max(0, barW - 12) / 2} ${padding.top + innerH + 4})`}
                textAnchor="end"
                fontSize={11}
                fill="#64748b"
              >{String(k)}</text>
            </g>
          )
        })}
        <line x1={padding.left} y1={padding.top + innerH} x2={padding.left + innerW} y2={padding.top + innerH} stroke="#cbd5e1" />
        <text x={padding.left - 8} y={padding.top} textAnchor="end" fontSize={11} fill="#64748b">100%</text>
        <text x={padding.left - 8} y={padding.top + innerH} textAnchor="end" fontSize={11} fill="#64748b">0%</text>
      </svg>
    </div>
  )
}

function Legend({ activeActivities, onToggle, onHover }: { activeActivities: Set<string>, onToggle: (k: string) => void, onHover: (k: string | null) => void }) {
  const items = Object.entries(activityNames)
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
      {items.map(([id, name]) => {
        const active = activeActivities.has(String(id))
        return (
          <button
            key={id}
            onMouseEnter={() => onHover(String(id))}
            onMouseLeave={() => onHover(null)}
            onClick={() => onToggle(String(id))}
            className={
              (active ? 'bg-slate-100 text-[#0f172a] ' : 'bg-white text-slate-500 ') +
              'border border-slate-200 rounded-md px-3 py-2 text-sm flex items-center justify-between hover:bg-slate-50 transition'
            }
          >
            <span>{name}</span>
            <span className={active ? 'text-[#1e3a8a]' : 'text-slate-400'}>{active ? 'On' : 'Off'}</span>
          </button>
        )
      })}
    </div>
  )
}
