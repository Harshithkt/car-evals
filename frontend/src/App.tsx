import { useCallback, useEffect, useRef, useState } from 'react'
import type { ComponentType, SVGProps } from 'react'
import { api } from './lib/api'
import type { Analysis, ClassMeta, Health, Metrics } from './lib/api'
import {
  CarMark, IconActivity, IconBook, IconChart, IconFlask, IconGrid, IconLayers, IconScan, IconSearch, IconXCircle,
} from './components/icons'
import Landing, { Footer } from './pages/Landing'
import Recognize from './pages/Recognize'
import Batch from './pages/Batch'
import type { BatchItem } from './pages/Batch'
import StressLab from './pages/StressLab'
import Pipeline from './pages/Pipeline'
import Comparison from './pages/Comparison'
import Robustness from './pages/Robustness'
import Failures from './pages/Failures'
import Catalogue from './pages/Catalogue'
import About from './pages/About'

type Icon = ComponentType<SVGProps<SVGSVGElement>>
const TABS: { id: string; label: string; icon: Icon; group: 'Tools' | 'Evaluation' | 'Reference' }[] = [
  { id: 'recognize', label: 'Recognize', icon: IconScan, group: 'Tools' },
  { id: 'batch', label: 'Batch', icon: IconGrid, group: 'Tools' },
  { id: 'stress', label: 'Stress lab', icon: IconFlask, group: 'Tools' },
  { id: 'pipeline', label: 'Pipeline', icon: IconLayers, group: 'Tools' },
  { id: 'comparison', label: 'Model comparison', icon: IconChart, group: 'Evaluation' },
  { id: 'robustness', label: 'Robustness & confidence', icon: IconActivity, group: 'Evaluation' },
  { id: 'failures', label: 'Failure analysis', icon: IconXCircle, group: 'Evaluation' },
  { id: 'catalogue', label: 'Catalogue', icon: IconSearch, group: 'Reference' },
  { id: 'about', label: 'About', icon: IconBook, group: 'Reference' },
]
const HISTORY_MAX = 8

/** Hash routes: `#/` = landing, `#/app/<tab>` = app. Old `#<tab>` links still work. */
function parseRoute(): { view: 'landing' } | { view: 'app'; tab: string } {
  const h = window.location.hash.replace(/^#\/?/, '')
  const m = h.match(/^app\/?([a-z]*)/)
  if (m) return { view: 'app', tab: TABS.some((t) => t.id === m[1]) ? m[1] : 'recognize' }
  if (TABS.some((t) => t.id === h)) return { view: 'app', tab: h }
  return { view: 'landing' }
}

export default function App() {
  const [route, setRoute] = useState(parseRoute)
  const [health, setHealth] = useState<Health | null>(null)
  const [healthErr, setHealthErr] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [plots, setPlots] = useState<string[] | null>(null)
  const [classes, setClasses] = useState<ClassMeta[]>([])
  const [current, setCurrent] = useState<Analysis | null>(null)
  const [history, setHistory] = useState<Analysis[]>([])
  const [batch, setBatch] = useState<BatchItem[]>([])
  const currentRef = useRef(current)
  const historyRef = useRef(history)
  currentRef.current = current
  historyRef.current = history

  const loadHealth = useCallback(() => {
    api.health().then((h) => { setHealth(h); setHealthErr(null) }).catch((e) => setHealthErr(String(e.message ?? e)))
  }, [])

  useEffect(() => {
    loadHealth()
    api.metrics().then(setMetrics).catch(() => setMetrics({}))
    api.plots().then(setPlots).catch(() => setPlots([]))
    api.classes().then(setClasses).catch(() => setClasses([]))
  }, [loadHealth])

  useEffect(() => {
    const onHash = () => { setRoute(parseRoute()); window.scrollTo({ top: 0 }) }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // reports always print in the light theme
  useEffect(() => {
    const before = () => { document.documentElement.dataset.theme = 'light' }
    const after = () => { delete document.documentElement.dataset.theme }
    window.addEventListener('beforeprint', before)
    window.addEventListener('afterprint', after)
    return () => { window.removeEventListener('beforeprint', before); window.removeEventListener('afterprint', after) }
  }, [])

  useEffect(() => {
    const t = route.view === 'app' ? TABS.find((x) => x.id === route.tab)?.label : null
    document.title = t ? `${t} · P5 Vehicle Recognition` : 'P5 Vehicle Recognition'
  }, [route])

  /** Show a new analysis; `record` = keep it in the history strip (live camera frames are not kept). */
  const commit = useCallback((a: Analysis, record: boolean) => {
    const prev = currentRef.current
    if (record) {
      const next = [a, ...historyRef.current.filter((h) => h.id !== a.id)].slice(0, HISTORY_MAX)
      historyRef.current.filter((h) => !next.includes(h) && h.preview !== prev?.preview).forEach((h) => URL.revokeObjectURL(h.preview))
      setHistory(next)
      historyRef.current = next
    }
    if (prev && prev.preview !== a.preview && !historyRef.current.some((h) => h.preview === prev.preview)) URL.revokeObjectURL(prev.preview)
    setCurrent(a)
  }, [])

  const clearHistory = useCallback(() => {
    historyRef.current.forEach((h) => { if (h.preview !== currentRef.current?.preview) URL.revokeObjectURL(h.preview) })
    setHistory([])
  }, [])

  const go = (tab: string) => { window.location.hash = `/app/${tab}` }
  const home = () => { window.location.hash = '/' }

  if (route.view === 'landing') return <Landing metrics={metrics} health={health} go={go} />

  const tab = route.tab
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-line bg-surface/85 backdrop-blur-md print:hidden">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4 sm:px-6">
          <button onClick={home} className="flex items-center gap-2.5 rounded-lg pr-2" title="Back to the home page">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-white"><CarMark className="h-5 w-5" /></span>
            <span className="hidden text-left leading-tight sm:block">
              <span className="block text-sm font-semibold tracking-[-0.01em] text-ink">Vehicle Recognition</span>
              <span className="block text-[11px] text-muted">P5 · Stanford Cars · 196 classes</span>
            </span>
          </button>
          <div className="ml-auto">
            <HealthPill health={health} err={healthErr} onRetry={loadHealth} />
          </div>
        </div>
        <nav className="mx-auto flex max-w-7xl items-center gap-1 overflow-x-auto px-3 pb-2 sm:px-5" aria-label="Sections">
          {TABS.map((t, i) => {
            const active = tab === t.id
            const newGroup = i > 0 && TABS[i - 1].group !== t.group
            return (
              <span key={t.id} className="flex shrink-0 items-center gap-1">
                {newGroup && <span className="mx-1.5 h-5 w-px bg-line" aria-hidden />}
                <button
                  onClick={() => go(t.id)}
                  aria-current={active ? 'page' : undefined}
                  title={t.group}
                  className={`inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm transition-colors ${
                    active ? 'bg-accent-wash/50 font-medium text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink'
                  }`}
                >
                  <t.icon className={`h-4 w-4 ${active ? 'text-accent' : 'text-muted'}`} />
                  {t.label}
                  {t.id === 'batch' && batch.some((b) => b.state === 'running') && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />}
                </button>
              </span>
            )
          })}
        </nav>
      </header>

      {health && !health.model_loaded && (
        <div className="mx-auto mt-4 w-full max-w-7xl px-4 sm:px-6 print:hidden">
          <div className="rounded-xl border border-critical/40 bg-critical/10 px-4 py-3 text-sm text-critical-ink">
            The backend is running but the model failed to load: {health.error ?? 'unknown error'}
          </div>
        </div>
      )}
      {healthErr && (
        <div className="mx-auto mt-4 w-full max-w-7xl px-4 sm:px-6 print:hidden">
          <div className="rounded-xl border border-critical/40 bg-critical/10 px-4 py-3 text-sm text-critical-ink">
            Cannot reach the backend. Start it with <code className="font-mono">uvicorn backend.app:app --reload</code> from the project folder, then{' '}
            <button onClick={loadHealth} className="underline">retry</button>.
          </div>
        </div>
      )}

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 print:max-w-none print:p-0">
        {tab === 'recognize' && (
          <Recognize health={health} current={current} history={history} commit={commit} restore={setCurrent} clearHistory={clearHistory}
            classes={classes} onShowPipeline={() => go('pipeline')} />
        )}
        {tab === 'batch' && <Batch items={batch} setItems={setBatch} maxMb={health?.max_upload_mb ?? 10} hasOnnx={health?.engines?.includes('onnx') ?? false} />}
        {tab === 'stress' && <StressLab current={current} />}
        {tab === 'pipeline' && <Pipeline result={current?.result ?? null} plots={plots} onGoRecognize={() => go('recognize')} />}
        {tab === 'comparison' && <Comparison metrics={metrics} plots={plots} />}
        {tab === 'robustness' && <Robustness metrics={metrics} plots={plots} />}
        {tab === 'failures' && <Failures metrics={metrics} plots={plots} />}
        {tab === 'catalogue' && <Catalogue classes={classes} />}
        {tab === 'about' && <About metrics={metrics} plots={plots} />}
      </main>
      <div className="print:hidden"><Footer go={go} wide /></div>
    </div>
  )
}

function HealthPill({ health, err, onRetry }: { health: Health | null; err: string | null; onRetry: () => void }) {
  if (err)
    return (
      <button onClick={onRetry} className="inline-flex items-center gap-1.5 rounded-full border border-critical/40 bg-critical/10 px-3 py-1 text-xs text-critical-ink">
        <span className="h-1.5 w-1.5 rounded-full bg-critical" /> Backend offline · retry
      </button>
    )
  if (!health) return <span className="text-xs text-muted">Connecting…</span>
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1 text-xs text-ink-2" title={health.warnings.join('\n') || undefined}>
      <span className="relative flex h-2 w-2">
        {health.model_loaded && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-good opacity-50" />}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${health.model_loaded ? 'bg-good' : 'bg-critical'}`} />
      </span>
      <span className="font-medium text-ink">{health.model_loaded ? 'Model ready' : 'Model not loaded'}</span>
      {health.device && <span className="hidden text-muted sm:inline">{health.device.toUpperCase()} · {health.backend}</span>}
    </span>
  )
}
