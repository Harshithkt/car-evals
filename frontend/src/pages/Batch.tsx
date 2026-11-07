import { useRef, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { api } from '../lib/api'
import type { Prediction, Status } from '../lib/api'
import { ms, pct } from '../lib/format'
import { downloadCSV, downloadJSON, slimPrediction, stamp, uid } from '../lib/download'
import { BarList } from '../components/charts'
import { IconCheck, IconDownload, IconGrid, IconImage, IconPlay, IconStop, IconTrash, IconUpload, IconXCircle } from '../components/icons'
import { Card, DET_SOURCE, PageHeader, STATUS_COLOR, StatTile, StatusBadge, Toggle } from '../components/ui'

export interface BatchItem {
  id: string
  name: string
  file: Blob
  url: string
  state: 'queued' | 'running' | 'done' | 'error'
  result?: Prediction
  error?: string
  ms?: number
  gt?: string | null
}

const ACCEPT = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']

export default function Batch({ items, setItems, maxMb, hasOnnx }: {
  items: BatchItem[]
  setItems: Dispatch<SetStateAction<BatchItem[]>>
  maxMb: number
  hasOnnx: boolean
}) {
  const [fast, setFast] = useState(true)
  const [running, setRunning] = useState(false)
  const [drag, setDrag] = useState(false)
  const [open, setOpen] = useState<BatchItem | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const cancel = useRef(false)
  const input = useRef<HTMLInputElement>(null)

  function add(files: { file: Blob; name: string; gt?: string | null }[]) {
    const ok = files.filter((f) => (!f.file.type || ACCEPT.includes(f.file.type)) && f.file.size <= maxMb * 1024 * 1024)
    const skipped = files.length - ok.length
    setNote(skipped ? `${skipped} file${skipped > 1 ? 's were' : ' was'} skipped (not a JPG/PNG/WEBP/BMP image, or over ${maxMb} MB).` : null)
    setItems((prev) => [...prev, ...ok.map((f) => ({ id: uid(), name: f.name, file: f.file, url: URL.createObjectURL(f.file), state: 'queued' as const, gt: f.gt ?? null }))])
  }

  async function addSamples() {
    const samples = await api.samples().catch(() => [])
    const files = await Promise.all(samples.map(async (s) => ({ file: await (await fetch(api.sampleUrl(s))).blob(), name: `${s.name}.jpg`, gt: s.ground_truth })))
    add(files)
  }

  async function runAll() {
    cancel.current = false
    setRunning(true)
    // snapshot of what still has to run; state updates go through setItems so the queue survives tab switches
    const todo = items.filter((i) => i.state === 'queued' || i.state === 'error')
    for (const it of todo) {
      if (cancel.current) break
      setItems((prev) => prev.map((p) => (p.id === it.id ? { ...p, state: 'running', error: undefined } : p)))
      const t0 = performance.now()
      try {
        const r = await api.predict(it.file, it.name, { gradcam: false, stages: false, maxVehicles: 5, engine: hasOnnx && fast ? 'onnx' : 'torch' })
        const dt = performance.now() - t0
        setItems((prev) => prev.map((p) => (p.id === it.id ? { ...p, state: 'done', result: r, ms: dt } : p)))
      } catch (e) {
        setItems((prev) => prev.map((p) => (p.id === it.id ? { ...p, state: 'error', error: e instanceof Error ? e.message : String(e) } : p)))
      }
    }
    setRunning(false)
  }

  function clear() {
    items.forEach((i) => URL.revokeObjectURL(i.url))
    setItems([])
    setNote(null)
  }

  const done = items.filter((i) => i.state === 'done')
  const vehicles = done.flatMap((i) => i.result!.vehicles)
  const statusCount = (s: Status) => vehicles.filter((v) => v.status === s).length
  const withGt = done.filter((i) => i.gt)
  const correct = withGt.filter((i) => i.result!.vehicles[0]?.label === i.gt).length
  const avgMs = done.length ? done.reduce((a, b) => a + (b.ms ?? 0), 0) / done.length : null
  const remaining = items.filter((i) => i.state === 'queued' || i.state === 'running').length
  const progress = items.length ? (items.length - remaining) / items.length : 0
  const makes = Object.entries(vehicles.reduce<Record<string, number>>((acc, v) => {
    const k = v.status === 'confident' ? v.make : `${v.make_guess} (uncertain)`
    acc[k] = (acc[k] ?? 0) + 1
    return acc
  }, {})).sort((a, b) => b[1] - a[1]).slice(0, 8)

  const exportCSV = () => downloadCSV(done.flatMap((i) => i.result!.vehicles.map((v, k) => ({
    file: i.name, vehicle: k + 1, make: v.make, model: v.model, year: v.year, label: v.label, confidence: v.confidence.toFixed(4),
    status: v.status, make_guess: v.make_guess, make_confidence: v.make_confidence.toFixed(4), det_source: v.det_source,
    det_conf: v.det_conf.toFixed(4), x1: Math.round(v.box[0]), y1: Math.round(v.box[1]), x2: Math.round(v.box[2]), y2: Math.round(v.box[3]),
    ground_truth: i.gt ?? '', correct: i.gt ? String(k === 0 && v.label === i.gt) : '', request_ms: Math.round(i.ms ?? 0),
  }))), `vehicle-batch-${stamp()}.csv`)
  const exportJSON = () => downloadJSON(done.map((i) => ({ file: i.name, ground_truth: i.gt, request_ms: i.ms, ...slimPrediction(i.result!) })), `vehicle-batch-${stamp()}.json`)

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Tools" title="Batch analysis"
        actions={items.length > 0 && (
          <div className="flex flex-wrap gap-2">
            <button onClick={exportCSV} disabled={!done.length} className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink-2 hover:text-ink disabled:opacity-40"><IconDownload className="h-4 w-4" /> CSV</button>
            <button onClick={exportJSON} disabled={!done.length} className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink-2 hover:text-ink disabled:opacity-40"><IconDownload className="h-4 w-4" /> JSON</button>
          </div>
        )}>
        Analyse many images in one go: a parking lot survey, a dataset sample, or a folder of photos. Results stay here while you switch tabs.
      </PageHeader>

      <Card>
        <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-center">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); add(Array.from(e.dataTransfer.files).map((f) => ({ file: f, name: f.name }))) }}
            onClick={() => input.current?.click()}
            role="button" tabIndex={0} aria-label="Add images"
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
            className={`flex cursor-pointer items-center gap-4 rounded-xl border-2 border-dashed px-5 py-5 transition ${drag ? 'border-ink bg-surface-2' : 'border-line hover:border-mist hover:bg-surface-2/60'}`}
          >
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-surface-2 text-ink"><IconUpload className="h-5 w-5" /></span>
            <div>
              <div className="text-sm font-medium text-ink">Drop images here, or <span className="text-ink">browse</span></div>
              <div className="text-xs text-muted">Select as many as you like · JPG, PNG, WEBP, BMP · up to {maxMb} MB each</div>
            </div>
            <input ref={input} type="file" multiple accept={ACCEPT.join(',')} className="hidden"
              onChange={(e) => { add(Array.from(e.target.files ?? []).map((f) => ({ file: f, name: f.name }))); e.target.value = '' }} />
          </div>
          <div className="flex flex-wrap gap-2">
            <button onClick={addSamples} disabled={running} className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-3.5 py-2.5 text-sm text-ink-2 hover:text-ink disabled:opacity-50">
              <IconImage className="h-4 w-4" /> Add sample images
            </button>
            {running ? (
              <button onClick={() => { cancel.current = true }} className="inline-flex items-center gap-2 rounded-lg bg-critical px-4 py-2.5 text-sm font-medium text-white">
                <IconStop className="h-4 w-4" /> Stop after current
              </button>
            ) : (
              <button onClick={runAll} disabled={!items.some((i) => i.state === 'queued' || i.state === 'error')}
                className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-40">
                <IconPlay className="h-4 w-4" /> Analyse {items.filter((i) => i.state === 'queued' || i.state === 'error').length || ''} image{items.filter((i) => i.state === 'queued' || i.state === 'error').length === 1 ? '' : 's'}
              </button>
            )}
            {items.length > 0 && !running && (
              <button onClick={clear} className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2.5 text-sm text-ink-2 hover:text-critical-ink"><IconTrash className="h-4 w-4" /> Clear</button>
            )}
          </div>
        </div>
        {hasOnnx && (
          <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-grid pt-4">
            <Toggle checked={fast} onChange={setFast} label="Fast mode: ONNX INT8 classifier" disabled={running} />
            <span className="text-xs text-muted">3.3× smaller and much faster on CPU; top-1 94.7% vs 94.6% FP32 on 1,000 test crops. Turn off to use PyTorch FP32.</span>
          </div>
        )}
        {note && <p className="mt-3 text-xs text-warning-ink">{note}</p>}
        {items.length > 0 && (
          <div className="mt-5">
            <div className="mb-1.5 flex justify-between text-xs text-ink-2">
              <span>{items.length - remaining} of {items.length} processed{running && avgMs ? ` · about ${Math.ceil((remaining * avgMs) / 1000)} s left` : ''}</span>
              <span className="tabular">{pct(progress, 0)}</span>
            </div>
            <div className="h-1.5 rounded-full bg-surface-2"><div className="h-1.5 rounded-full bg-accent transition-all" style={{ width: `${progress * 100}%` }} /></div>
          </div>
        )}
      </Card>

      {items.length === 0 ? (
        <Card>
          <div className="flex flex-col items-center gap-3 py-12 text-center">
            <IconGrid className="h-10 w-10 text-muted" />
            <div className="text-sm text-ink-2">No images yet. Add some above, or click <b className="text-ink">Add sample images</b> to try it.</div>
          </div>
        </Card>
      ) : (
        <>
          {done.length > 0 && (
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <div className="grid grid-cols-2 gap-3">
                <StatTile label="Images analysed" value={done.length} note={`${items.filter((i) => i.state === 'error').length} failed`} />
                <StatTile label="Vehicles found" value={vehicles.length} note={`${(vehicles.length / Math.max(1, done.length)).toFixed(1)} per image`} />
                <StatTile label={withGt.length ? 'Accuracy vs ground truth' : 'Confident predictions'}
                  value={withGt.length ? pct(correct / withGt.length, 0) : pct(statusCount('confident') / Math.max(1, vehicles.length), 0)}
                  note={withGt.length ? `${correct} of ${withGt.length} labelled images` : `${statusCount('confident')} of ${vehicles.length} vehicles`} />
                <StatTile label="Average request" value={ms(avgMs)} note={`incl. upload · ${done[0]?.result?.backend === "onnx" ? "ONNX INT8" : "PyTorch FP32"}`} />
              </div>
              <Card title="Makes found" subtitle="Vehicles per make across the batch">
                {makes.length ? <BarList items={makes.map(([k, v]) => ({ label: k, value: v }))} fmt={(v) => String(v)} /> : <p className="text-sm text-ink-2">No vehicles yet.</p>}
                <div className="mt-4 flex h-2 overflow-hidden rounded-full bg-surface-2" aria-label="Decision mix">
                  {(['confident', 'uncertain_model', 'low_confidence'] as const).map((s) => (
                    <span key={s} style={{ width: `${(statusCount(s) / Math.max(1, vehicles.length)) * 100}%`, background: STATUS_COLOR[s] }} title={`${s}: ${statusCount(s)}`} />
                  ))}
                </div>
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
                  {(['confident', 'uncertain_model', 'low_confidence'] as const).map((s) => (
                    <span key={s} className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: STATUS_COLOR[s] }} />{s.replace('_', ' ')} <b className="tabular text-ink">{statusCount(s)}</b></span>
                  ))}
                </div>
              </Card>
            </div>
          )}

          <Card title="Results" subtitle="Click a row to see the annotated image">
            <div className="-mx-1 overflow-x-auto">
              <table className="w-full min-w-[720px] text-sm">
                <thead>
                  <tr className="border-b border-grid text-left text-[12px] text-muted">
                    <th className="px-2 py-2 font-medium">Image</th>
                    <th className="px-2 py-2 font-medium">Prediction (first vehicle)</th>
                    <th className="px-2 py-2 font-medium">Status</th>
                    <th className="px-2 py-2 text-right font-medium">Confidence</th>
                    <th className="px-2 py-2 text-right font-medium">Vehicles</th>
                    <th className="px-2 py-2 font-medium">Detector</th>
                    <th className="px-2 py-2 text-right font-medium">Time</th>
                  </tr>
                </thead>
                <tbody className="tabular">
                  {items.map((it) => {
                    const v = it.result?.vehicles[0]
                    const ok = it.gt && v ? v.label === it.gt : null
                    return (
                      <tr key={it.id} onClick={() => it.result && setOpen(it)} className={`border-b border-grid last:border-0 ${it.result ? 'cursor-pointer hover:bg-surface-2/60' : ''}`}>
                        <td className="px-2 py-2">
                          <div className="flex items-center gap-3">
                            <img src={it.url} alt="" className="h-10 w-14 rounded-md object-cover ring-1 ring-line" />
                            <span className="max-w-[12rem] truncate text-ink-2">{it.name}</span>
                          </div>
                        </td>
                        <td className="px-2 py-2">
                          {it.state === 'queued' && <span className="text-muted">Queued</span>}
                          {it.state === 'running' && <span className="inline-flex items-center gap-2 text-ink"><span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-accent/25 border-t-accent" /> Analysing…</span>}
                          {it.state === 'error' && <span className="text-critical-ink">{it.error}</span>}
                          {it.state === 'done' && (v ? (
                            <span className="inline-flex items-center gap-1.5 text-ink">
                              {v.status === 'confident' ? v.label : `${v.make_guess}?`}
                              {ok !== null && (ok ? <IconCheck className="h-4 w-4 text-good-ink" /> : <IconXCircle className="h-4 w-4 text-critical-ink" />)}
                            </span>
                          ) : <span className="text-muted">No vehicle found</span>)}
                        </td>
                        <td className="px-2 py-2">{v && <StatusBadge status={v.status} />}</td>
                        <td className="px-2 py-2 text-right text-ink-2">{v ? pct(v.confidence) : '—'}</td>
                        <td className="px-2 py-2 text-right text-ink-2">{it.result ? it.result.vehicles.length : '—'}</td>
                        <td className="px-2 py-2 text-ink-2">{v ? DET_SOURCE[v.det_source]?.label ?? v.det_source : '—'}</td>
                        <td className="px-2 py-2 text-right text-ink-2">{it.ms ? ms(it.ms) : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {open?.result && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4" onClick={() => setOpen(null)}>
          <div className="max-h-full max-w-4xl overflow-auto rounded-2xl bg-surface p-4" onClick={(e) => e.stopPropagation()}>
            <div className="mb-3 flex items-center justify-between gap-4">
              <div className="truncate text-sm font-medium text-ink">{open.name}</div>
              <button onClick={() => setOpen(null)} className="text-sm text-ink-2 hover:text-ink">Close</button>
            </div>
            <img src={open.result.annotated_image} alt={open.name} className="max-h-[70vh] w-auto max-w-full rounded-lg" />
            <ul className="mt-3 space-y-1 text-sm">
              {open.result.vehicles.map((v, i) => (
                <li key={i} className="flex flex-wrap items-center gap-2 text-ink-2">
                  <span className="tabular text-muted">#{i + 1}</span> <span className="text-ink">{v.label}</span> <StatusBadge status={v.status} /> <span className="tabular">{pct(v.confidence)}</span>
                </li>
              ))}
            </ul>
            {open.gt && <p className="mt-2 text-xs text-muted">Ground truth: {open.gt}</p>}
          </div>
        </div>
      )}
    </div>
  )
}
