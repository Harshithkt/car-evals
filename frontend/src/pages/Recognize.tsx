import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { Analysis, ClassMeta, Health, Sample } from '../lib/api'
import { ms, pct } from '../lib/format'
import { downloadJSON, slimPrediction, stamp, uid } from '../lib/download'
import { CameraPanel } from '../components/CameraPanel'
import { VehicleCard } from '../components/VehicleCard'
import {
  IconAlert, IconArrowRight, IconCamera, IconCheck, IconClock, IconDownload, IconHistory, IconLayers, IconPrinter, IconScan, IconTrash, IconUpload,
} from '../components/icons'
import { Card, PageHeader, STATUS_COLOR, Toggle } from '../components/ui'

const ACCEPT = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']

export default function Recognize({ health, current, history, commit, restore, clearHistory, classes, onShowPipeline }: {
  health: Health | null
  current: Analysis | null
  history: Analysis[]
  commit: (a: Analysis, record: boolean) => void
  restore: (a: Analysis) => void
  clearHistory: () => void
  classes: ClassMeta[]
  onShowPipeline: () => void
}) {
  const [mode, setMode] = useState<'upload' | 'camera'>('upload')
  const [samples, setSamples] = useState<Sample[]>([])
  const [busy, setBusy] = useState(false)
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [drag, setDrag] = useState(false)
  const [gradcam, setGradcam] = useState(true)
  const [maxVehicles, setMaxVehicles] = useState(5)
  const [engine, setEngine] = useState<'torch' | 'onnx'>('torch')
  const [view, setView] = useState<'annotated' | 'gradcam' | 'original'>('annotated')
  const input = useRef<HTMLInputElement>(null)
  const resultsCol = useRef<HTMLDivElement>(null)
  const maxMb = health?.max_upload_mb ?? 10
  const hasOnnx = health?.engines?.includes('onnx') ?? false
  const camAvailable = (health?.gradcam_available ?? true) && engine === 'torch'
  const thresholds = health?.thresholds ?? {}
  const result = current?.result ?? null

  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples([]))
  }, [])

  // paste an image from the clipboard
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const f = Array.from(e.clipboardData?.files ?? []).find((x) => x.type.startsWith('image/'))
      if (f) run(f, f.name || 'pasted-image.png', { source: 'upload' })
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  })

  async function run(file: Blob, name: string, opts: { gt?: string | null; source: Analysis['source']; live?: boolean }) {
    setError(null)
    if (file.type && !ACCEPT.includes(file.type)) {
      setError(`Unsupported file type (${file.type}). Use JPG, PNG, WEBP or BMP.`)
      return
    }
    if (file.size > maxMb * 1024 * 1024) {
      setError(`File is ${(file.size / 1e6).toFixed(1)} MB; the limit is ${maxMb} MB.`)
      return
    }
    const preview = URL.createObjectURL(file)
    if (!opts.live) setPending(preview)
    setBusy(true)
    if (!opts.live && mode === 'upload' && window.innerWidth < 1024)
      requestAnimationFrame(() => resultsCol.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    try {
      const r = await api.predict(file, name, {
        gradcam: gradcam && camAvailable && !opts.live,
        stages: !opts.live,
        maxVehicles,
        // live camera frames use the INT8 ONNX engine when it is loaded (~10× faster classification on CPU)
        engine: opts.live && hasOnnx ? 'onnx' : engine,
        timeoutMs: opts.live ? 20_000 : 90_000,
      })
      commit({ id: uid(), at: Date.now(), name, preview, result: r, groundTruth: opts.gt ?? null, source: opts.source }, !opts.live)
      if (!opts.live) setView('annotated')
    } catch (e) {
      URL.revokeObjectURL(preview)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      setPending(null)
    }
  }

  async function runSample(s: Sample) {
    try {
      const b = await (await fetch(api.sampleUrl(s))).blob()
      await run(b, `${s.name}.jpg`, { gt: s.ground_truth, source: 'sample' })
    } catch {
      setError('Could not load the sample image from the backend.')
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files?.[0]
    if (f) run(f, f.name, { source: 'upload' })
  }

  const preview = pending ?? current?.preview ?? null
  const shown = busy && pending ? pending : view === 'gradcam' && result?.gradcam ? result.gradcam : view === 'original' ? preview : result?.annotated_image
  const groundTruth = current?.groundTruth ?? null
  const correct = groundTruth && result?.vehicles[0] ? result.vehicles[0].label === groundTruth : null

  return (
    <div>
      {/* print-only report header */}
      <div className="mb-6 hidden border-b border-line pb-4 print:block">
        <div className="text-xs uppercase tracking-[0.08em] text-muted">P5 Fine-Grained Vehicle Recognition · analysis report</div>
        <div className="mt-1 text-xl font-semibold text-ink">{current?.name ?? 'Image'}</div>
        <div className="text-xs text-ink-2">
          {current && new Date(current.at).toLocaleString()} · {result?.device.toUpperCase()} · {result?.backend} · thresholds τ = {thresholds.tau}, τ_make = {thresholds.tau_make}
        </div>
      </div>

      <div className="print:hidden">
        <PageHeader eyebrow="Recognize" title="Identify vehicles in a photo">
          Upload a photo or use your camera. The system detects every car and reports its make, model and year with a calibrated confidence.
        </PageHeader>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] print:block">
        {/* ---------- left: input ---------- */}
        <div className="space-y-5 print:hidden">
          <Card>
            <div className="mb-4 grid grid-cols-2 rounded-lg bg-surface-2 p-0.5 text-sm" role="tablist" aria-label="Input source">
              {([['upload', 'Upload', IconUpload], ['camera', 'Camera', IconCamera]] as const).map(([m, label, I]) => (
                <button key={m} role="tab" aria-selected={mode === m} onClick={() => setMode(m)}
                  className={`inline-flex items-center justify-center gap-2 rounded-md py-1.5 font-medium transition ${mode === m ? 'bg-surface text-ink shadow-sm' : 'text-ink-2 hover:text-ink'}`}>
                  <I className="h-4 w-4" /> {label}
                </button>
              ))}
            </div>

            {mode === 'upload' ? (
              <div
                onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
                onDragLeave={() => setDrag(false)}
                onDrop={onDrop}
                onClick={() => !busy && input.current?.click()}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
                role="button"
                tabIndex={0}
                aria-label="Upload an image"
                className={`group flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-9 text-center transition-all ${
                  drag ? 'scale-[1.01] border-accent bg-accent-wash/30' : 'border-line hover:border-accent/60 hover:bg-surface-2/60'
                }`}
              >
                <span className={`mb-3 flex h-12 w-12 items-center justify-center rounded-full transition-colors ${drag ? 'bg-accent text-white' : 'bg-accent-wash/50 text-accent group-hover:bg-accent group-hover:text-white'}`}>
                  <IconUpload className="h-5 w-5" />
                </span>
                <div className="text-sm font-medium text-ink">{busy ? 'Analysing…' : <>Drop an image, or <span className="text-accent">browse</span></>}</div>
                <div className="mt-1 text-xs text-muted">JPG, PNG, WEBP or BMP · up to {maxMb} MB · or paste</div>
                <input ref={input} type="file" accept={ACCEPT.join(',')} className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f, f.name, { source: 'upload' }); e.target.value = '' }} />
              </div>
            ) : (
              <div className="rounded-xl bg-surface-2/60 px-4 py-4 text-[13px] leading-relaxed text-ink-2">
                Start the camera on the right, then <b className="text-ink">Capture</b> a single frame (with Grad-CAM and all stages) or switch on
                <b className="text-ink"> Live mode</b> to re-analyse continuously. Live frames skip stages and Grad-CAM to stay fast and are not saved to history.
              </div>
            )}

            <div className="mt-5 space-y-3 border-t border-grid pt-4">
              <Toggle checked={gradcam && camAvailable} onChange={setGradcam} label="Grad-CAM explanation" disabled={!camAvailable} />
              <label className="flex items-center justify-between text-sm text-ink-2">
                Max vehicles per image
                <select value={maxVehicles} onChange={(e) => setMaxVehicles(Number(e.target.value))}
                  className="rounded-lg border border-line bg-surface px-2.5 py-1 text-ink focus:outline-2 focus:outline-accent">
                  {[1, 2, 3, 5, 8, 10].map((n) => <option key={n}>{n}</option>)}
                </select>
              </label>
              {hasOnnx && (
                <label className="flex items-center justify-between text-sm text-ink-2">
                  Classifier engine
                  <select value={engine} onChange={(e) => setEngine(e.target.value as 'torch' | 'onnx')}
                    className="rounded-lg border border-line bg-surface px-2.5 py-1 text-ink focus:outline-2 focus:outline-accent">
                    <option value="torch">PyTorch FP32</option>
                    <option value="onnx">ONNX INT8 (fast)</option>
                  </select>
                </label>
              )}
              {engine === 'onnx' && <p className="text-xs text-muted">INT8 is 3.3× smaller and much faster on CPU (top-1 94.7% vs 94.6% FP32 on 1,000 test crops). Grad-CAM needs PyTorch.</p>}
              {mode === 'camera' && hasOnnx && <p className="text-xs text-muted">Live mode always uses the INT8 engine for speed.</p>}
              {!hasOnnx && !camAvailable && <p className="text-xs text-muted">Grad-CAM needs the PyTorch backend (BACKEND=torch).</p>}
            </div>
          </Card>

          {samples.length > 0 && mode === 'upload' && (
            <Card title="Sample images" subtitle="Unseen Stanford Cars test images">
              <div className="grid grid-cols-3 gap-2">
                {samples.map((s) => (
                  <button key={s.name} onClick={() => runSample(s)} disabled={busy} title={s.ground_truth ?? s.name}
                    className="group relative overflow-hidden rounded-lg ring-1 ring-line transition hover:ring-2 hover:ring-accent focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50">
                    <img src={api.sampleUrl(s)} alt={s.ground_truth ?? s.name} className="aspect-[4/3] w-full object-cover transition-transform duration-300 group-hover:scale-105" />
                  </button>
                ))}
              </div>
            </Card>
          )}

          {history.length > 0 && (
            <Card
              title={<span className="inline-flex items-center gap-2"><IconHistory className="h-4 w-4 text-muted" /> Recent analyses</span>}
              actions={<button onClick={clearHistory} className="inline-flex items-center gap-1 text-xs text-ink-2 hover:text-critical-ink"><IconTrash className="h-3.5 w-3.5" /> Clear</button>}
            >
              <ul className="-mx-2 space-y-1">
                {history.map((h) => {
                  const v = h.result.vehicles[0]
                  return (
                    <li key={h.id}>
                      <button onClick={() => { restore(h); setView('annotated') }}
                        className={`flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left transition hover:bg-surface-2 ${current?.id === h.id ? 'bg-accent-wash/30' : ''}`}>
                        <img src={h.preview} alt="" className="h-10 w-14 shrink-0 rounded-md object-cover ring-1 ring-line" />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-[13px] font-medium text-ink">{v ? (v.status === 'confident' ? v.label : `${v.make_guess}?`) : 'No vehicle'}</span>
                          <span className="block truncate text-[11px] text-muted">{h.source} · {new Date(h.at).toLocaleTimeString()}</span>
                        </span>
                        {v && <span className="flex items-center gap-1.5 text-xs tabular text-ink-2"><span className="h-2 w-2 rounded-full" style={{ background: STATUS_COLOR[v.status] }} />{pct(v.confidence)}</span>}
                      </button>
                    </li>
                  )
                })}
              </ul>
            </Card>
          )}
        </div>

        {/* ---------- right: results ---------- */}
        <div ref={resultsCol} className="min-w-0 scroll-mt-28 space-y-5">
          {error && (
            <div role="alert" className="flex items-start gap-3 rounded-xl border border-critical/40 bg-critical/10 px-4 py-3 text-sm text-critical-ink print:hidden">
              <IconAlert className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {mode === 'camera' && (
            <div className="print:hidden">
              <CameraPanel busy={busy} result={current?.source === 'camera' ? result : null}
                onCapture={(blob, live) => run(blob, `camera-${stamp()}.jpg`, { source: 'camera', live })} />
            </div>
          )}

          {mode === 'upload' && !result && !busy && <EmptyState onSample={samples[4] ? () => runSample(samples[4]) : undefined} />}

          {mode === 'upload' && (result || busy) && (
            <Card
              className="print:break-inside-avoid"
              title={result && !busy ? `${result.vehicles.length} vehicle${result.vehicles.length === 1 ? '' : 's'} detected` : 'Analysing image…'}
              subtitle={
                groundTruth && !busy ? (
                  <span className="inline-flex flex-wrap items-center gap-1.5">
                    Ground truth: <span className="font-medium text-ink">{groundTruth}</span>
                    {correct !== null && (
                      <span className={`inline-flex items-center gap-1 rounded-full px-1.5 py-px text-[11px] font-medium ${correct ? 'bg-good/12 text-good-ink' : 'bg-critical/10 text-critical-ink'}`}>
                        {correct && <IconCheck className="h-3 w-3" />}{correct ? 'match' : 'mismatch'}
                      </span>
                    )}
                  </span>
                ) : current?.name ?? undefined
              }
              actions={
                result && !busy && (
                  <div className="flex flex-wrap items-center gap-2 print:hidden">
                    <div className="inline-flex rounded-lg bg-surface-2 p-0.5 text-xs" role="tablist">
                      {(['annotated', 'gradcam', 'original'] as const).map((v) => (
                        <button key={v} role="tab" aria-selected={view === v} disabled={v === 'gradcam' && !result.gradcam} onClick={() => setView(v)}
                          className={`rounded-md px-3 py-1.5 font-medium transition disabled:opacity-40 ${view === v ? 'bg-surface text-ink shadow-sm' : 'text-ink-2 hover:text-ink'}`}
                          title={v === 'gradcam' && !result.gradcam ? (result.gradcam_note ?? 'Turn on Grad-CAM and run again') : undefined}>
                          {v === 'gradcam' ? 'Grad-CAM' : v === 'annotated' ? 'Detections' : 'Original'}
                        </button>
                      ))}
                    </div>
                    <button onClick={() => window.print()} title="Print or save as PDF" className="inline-flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-xs font-medium text-ink-2 hover:text-ink">
                      <IconPrinter className="h-3.5 w-3.5" /> Report
                    </button>
                    <button onClick={() => current && downloadJSON({ ...slimPrediction(result), ground_truth: groundTruth, thresholds }, `vehicle-analysis-${stamp()}.json`)}
                      title="Download the result as JSON" className="inline-flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-xs font-medium text-ink-2 hover:text-ink">
                      <IconDownload className="h-3.5 w-3.5" /> JSON
                    </button>
                  </div>
                )
              }
            >
              <div className="relative flex min-h-56 items-center justify-center overflow-hidden rounded-xl bg-surface-2">
                {shown && (
                  <img src={shown} alt={busy ? 'Uploaded image' : 'Recognition result'}
                    className={`h-auto max-h-[520px] w-full max-w-[760px] object-contain transition ${busy ? 'scale-[1.01] opacity-50 blur-[1px]' : ''}`} />
                )}
                {busy && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                    <span className="h-9 w-9 animate-spin rounded-full border-[3px] border-accent/25 border-t-accent" />
                    <span className="rounded-full bg-surface/90 px-3 py-1 text-sm font-medium text-ink shadow-sm">Detecting · classifying · explaining</span>
                  </div>
                )}
              </div>
              {view === 'gradcam' && result?.gradcam && !busy && (
                <p className="mt-3 text-xs text-muted">Grad-CAM of the ConvNeXt classifier on the first vehicle's crop. Warm colours mark the regions that drove the prediction.</p>
              )}
              {result && !busy && (
                <div className="mt-4 flex flex-wrap items-center gap-2 text-xs text-ink-2 tabular">
                  <Chip label="Detection" value={ms(result.timings_ms.detection)} />
                  <Chip label="Classification" value={ms(result.timings_ms.classification)} />
                  <Chip label="Total" value={ms(result.timings_ms.total)} icon />
                  <span className="text-muted">{result.device.toUpperCase()} · {result.backend}</span>
                  <button onClick={onShowPipeline} className="ml-auto inline-flex items-center gap-1.5 font-medium text-accent hover:underline print:hidden">
                    <IconLayers className="h-3.5 w-3.5" /> Intermediate results <IconArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}
            </Card>
          )}

          {!(busy && mode === 'upload') && result && (mode === 'upload' || current?.source === 'camera') &&
            result.vehicles.map((v, i) => (
              <VehicleCard key={`${current!.id}-${i}`} v={v} index={i} total={result.vehicles.length} preview={current!.preview}
                imageSize={result.image_size} classes={classes} thresholds={thresholds} />
            ))}
          {result && result.vehicles.length === 0 && !busy && (
            <Card><p className="text-sm text-ink-2">No vehicle was found in this image.</p></Card>
          )}
        </div>
      </div>
    </div>
  )
}

function Chip({ label, value, icon }: { label: string; value: string; icon?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1">
      {icon && <IconClock className="h-3.5 w-3.5 text-muted" />}
      {label} <b className="font-semibold text-ink">{value}</b>
    </span>
  )
}

function EmptyState({ onSample }: { onSample?: () => void }) {
  const steps = [
    ['Detect', 'YOLOv8 finds every vehicle, with classical fallbacks'],
    ['Classify', 'ConvNeXt + classical features score 196 classes'],
    ['Explain', 'Calibrated confidence, top-5, runner-up analysis and Grad-CAM'],
  ]
  return (
    <Card>
      <div className="grid items-center gap-8 py-4 md:grid-cols-[1fr_1.1fr]">
        <div className="relative overflow-hidden rounded-xl bg-surface-2">
          <img src="/landing/hero.jpg" alt="" className="w-full opacity-90" />
          <div className="absolute rounded-[3px] border-2 border-[#22c55e]" style={{ left: '4.47%', top: '21.11%', width: '92.48%', height: '70.96%' }}>
            <span className="scanline" />
          </div>
        </div>
        <div>
          <h2 className="text-lg font-semibold tracking-[-0.01em] text-ink">No image analysed yet</h2>
          <p className="mt-1 text-sm text-ink-2">Upload a photo on the left, drop one on the upload box, paste from the clipboard, or switch to the camera.</p>
          <ol className="mt-5 space-y-3">
            {steps.map(([t, d], i) => (
              <li key={t} className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent-wash/60 text-xs font-semibold text-ink tabular">{i + 1}</span>
                <div className="text-sm"><span className="font-medium text-ink">{t}.</span> <span className="text-ink-2">{d}</span></div>
              </li>
            ))}
          </ol>
          {onSample && (
            <button onClick={onSample} className="mt-6 inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90">
              <IconScan className="h-4 w-4" /> Try a sample image
            </button>
          )}
        </div>
      </div>
    </Card>
  )
}
