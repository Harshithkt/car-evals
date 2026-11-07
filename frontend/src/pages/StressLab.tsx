import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { Analysis, Prediction, Sample, Vehicle } from '../lib/api'
import { pct } from '../lib/format'
import { KINDS, PRESETS, baseCanvas, loadImage, perturb, toBlob } from '../lib/perturb'
import type { Kind } from '../lib/perturb'
import { Lines } from '../components/charts'
import { IconArrowRight, IconCheck, IconCompare, IconFlask, IconPlay, IconSun, IconUpload, IconXCircle } from '../components/icons'
import { Card, Label, PageHeader, StatusBadge } from '../components/ui'

interface SweepPoint { s: number; thumb: string; result: Prediction }

const LEVELS = [0, 0.2, 0.4, 0.6, 0.8, 1]
const first = (p?: Prediction | null): Vehicle | undefined => p?.vehicles[0]
const probOf = (v: Vehicle | undefined, label: string) => v?.top5.find((t) => t.label === label)?.confidence ?? 0

export default function StressLab({ current }: { current: Analysis | null }) {
  const [samples, setSamples] = useState<Sample[]>([])
  const [source, setSource] = useState<{ name: string; gt: string | null } | null>(null)
  const [base, setBase] = useState<HTMLCanvasElement | null>(null)
  const [baseline, setBaseline] = useState<Prediction | null>(null)
  const [kind, setKind] = useState<Kind>('night')
  const [severity, setSeverity] = useState(0.7)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [compare, setCompare] = useState<Prediction | null>(null)
  const [sweep, setSweep] = useState<SweepPoint[]>([])
  const [busy, setBusy] = useState<null | 'baseline' | 'compare' | 'sweep'>(null)
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => { api.samples().then(setSamples).catch(() => setSamples([])) }, [])

  async function pick(src: string, name: string, gt: string | null) {
    setError(null); setCompare(null); setSweep([]); setBaseline(null)
    try {
      setBusy('baseline')
      const img = await loadImage(src)
      const c = baseCanvas(img)
      setBase(c)
      setSource({ name, gt })
      const r = await api.predict(await toBlob(c), name, { gradcam: false, stages: true, maxVehicles: 1 })
      setBaseline(r)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  // live preview of the perturbation
  useEffect(() => {
    if (!base) return
    const t = setTimeout(() => setPreviewUrl(perturb(base, kind, severity, first(baseline)?.box).toDataURL('image/jpeg', 0.85)), 60)
    return () => clearTimeout(t)
  }, [base, kind, severity, baseline])

  async function runCompare() {
    if (!base) return
    setBusy('compare'); setError(null)
    try {
      const blob = await toBlob(perturb(base, kind, severity, first(baseline)?.box))
      setCompare(await api.predict(blob, `${kind}-${Math.round(severity * 100)}.jpg`, { gradcam: false, stages: true, maxVehicles: 1 }))
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy(null) }
  }

  async function runSweep() {
    if (!base) return
    setBusy('sweep'); setError(null); setSweep([])
    try {
      for (const s of LEVELS) {
        const c = perturb(base, kind, s, first(baseline)?.box)
        const r = await api.predict(await toBlob(c), `${kind}-${s}.jpg`, { gradcam: false, stages: false, maxVehicles: 1 })
        const thumb = document.createElement('canvas')
        const k = 240 / c.width
        thumb.width = 240; thumb.height = Math.round(c.height * k)
        thumb.getContext('2d')!.drawImage(c, 0, 0, thumb.width, thumb.height)
        setSweep((prev) => [...prev, { s, thumb: thumb.toDataURL('image/jpeg', 0.8), result: r }])
      }
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy(null) }
  }

  const ref = source?.gt ?? first(baseline)?.label ?? ''
  const kindInfo = KINDS.find((k) => k.id === kind)!

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Tools" title="Stress lab">
        Degrade a photo the way the real world does (darkness, fog, blur, low-res CCTV, occlusion, a tilted camera), then re-run the full system and see
        what breaks. Everything happens live on one image. For test-set robustness numbers, see Robustness & confidence.
      </PageHeader>

      {/* ---------- 1. source ---------- */}
      <Card title="1 · Pick an image" subtitle="It is resized to at most 1024 px so every variant uses identical pixels">
        <div className="flex flex-wrap items-center gap-3">
          {samples.map((s) => (
            <button key={s.name} onClick={() => pick(api.sampleUrl(s), `${s.name}.jpg`, s.ground_truth)} disabled={!!busy} title={s.ground_truth ?? s.name}
              className={`overflow-hidden rounded-lg ring-1 transition hover:ring-2 hover:ring-accent disabled:opacity-50 ${source?.name === `${s.name}.jpg` ? 'ring-2 ring-accent' : 'ring-line'}`}>
              <img src={api.sampleUrl(s)} alt={s.ground_truth ?? s.name} className="h-14 w-20 object-cover" />
            </button>
          ))}
          {current && (
            <button onClick={() => pick(current.preview, current.name, current.groundTruth)} disabled={!!busy}
              className="inline-flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-sm text-ink-2 hover:text-ink disabled:opacity-50">
              <img src={current.preview} alt="" className="h-8 w-11 rounded object-cover" /> Last recognised image
            </button>
          )}
          <button onClick={() => input.current?.click()} disabled={!!busy} className="inline-flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-sm text-ink-2 hover:text-ink disabled:opacity-50">
            <IconUpload className="h-4 w-4" /> Upload
          </button>
          <input ref={input} type="file" accept="image/jpeg,image/png,image/webp,image/bmp" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) pick(URL.createObjectURL(f), f.name, null); e.target.value = '' }} />
        </div>
        {busy === 'baseline' && <p className="mt-3 text-sm text-accent">Running the clean baseline…</p>}
        {baseline && (
          <p className="mt-3 text-sm text-ink-2">
            Clean baseline: <b className="text-ink">{first(baseline)?.label ?? 'no vehicle'}</b> at {pct(first(baseline)?.confidence)}
            {source?.gt && <> · ground truth <b className="text-ink">{source.gt}</b></>}
          </p>
        )}
      </Card>

      {error && <div role="alert" className="rounded-xl border border-critical/40 bg-critical/10 px-4 py-3 text-sm text-critical-ink">{error}</div>}

      {base && (
        <>
          {/* ---------- 2. perturbation ---------- */}
          <Card title="2 · Choose a real-world condition">
            <div className="mb-4 flex flex-wrap gap-2">
              <span className="self-center text-xs text-muted">Presets:</span>
              {PRESETS.map((p) => (
                <button key={p.label} onClick={() => { setKind(p.kind); setSeverity(p.severity) }}
                  className={`rounded-full border px-3 py-1 text-xs transition ${kind === p.kind && severity === p.severity ? 'border-accent bg-accent-wash/40 text-ink' : 'border-line text-ink-2 hover:text-ink'}`}>
                  {p.label}
                </button>
              ))}
            </div>
            <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
              <div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {KINDS.map((k) => (
                    <button key={k.id} onClick={() => setKind(k.id)}
                      className={`rounded-xl border px-3 py-2.5 text-left transition ${kind === k.id ? 'border-accent bg-accent-wash/30 ring-1 ring-accent' : 'border-line hover:bg-surface-2'}`}>
                      <div className="text-[13px] font-medium text-ink">{k.label}</div>
                      <div className="text-[11px] text-muted">{k.challenge}</div>
                    </button>
                  ))}
                </div>
                <label className="mt-5 block">
                  <div className="mb-1 flex justify-between text-sm"><span className="text-ink-2">Severity</span><span className="tabular font-medium text-ink">{Math.round(severity * 100)}%</span></div>
                  <input type="range" min={0} max={1} step={0.05} value={severity} onChange={(e) => setSeverity(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
                </label>
                <p className="mt-2 text-xs text-muted">{kindInfo.desc}</p>
                <div className="mt-5 flex flex-wrap gap-2">
                  <button onClick={runCompare} disabled={!!busy || !baseline} className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50">
                    <IconCompare className="h-4 w-4" /> {busy === 'compare' ? 'Running…' : 'Compare with original'}
                  </button>
                  <button onClick={runSweep} disabled={!!busy || !baseline} className="inline-flex items-center gap-2 rounded-lg border border-line px-4 py-2.5 text-sm font-medium text-ink hover:bg-surface-2 disabled:opacity-50">
                    <IconPlay className="h-4 w-4" /> {busy === 'sweep' ? `Sweeping… ${sweep.length}/${LEVELS.length}` : 'Sweep 0 → 100%'}
                  </button>
                </div>
              </div>
              <div className="overflow-hidden rounded-xl bg-surface-2">
                {previewUrl && <img src={previewUrl} alt="Perturbed preview" className="w-full" />}
                <div className="px-3 py-2 text-xs text-ink-2">Preview · {kindInfo.label} at {Math.round(severity * 100)}%</div>
              </div>
            </div>
          </Card>

          {/* ---------- 3. comparison ---------- */}
          {compare && baseline && (
            <Card title="3 · Before vs after" subtitle={`${kindInfo.label} at ${Math.round(severity * 100)}%`}>
              <div className="grid gap-5 md:grid-cols-2">
                <Side title="Original" p={baseline} reference={ref} />
                <Side title={`${kindInfo.label} ${Math.round(severity * 100)}%`} p={compare} reference={ref} delta={(first(compare)?.confidence ?? 0) - (first(baseline)?.confidence ?? 0)} />
              </div>
              <Verdict before={baseline} after={compare} reference={ref} />
            </Card>
          )}

          {/* ---------- sweep ---------- */}
          {sweep.length > 0 && (
            <Card title={`Severity sweep · ${kindInfo.label}`} subtitle={`Reference label: ${ref || '—'} (${source?.gt ? 'ground truth' : 'clean prediction'})`}>
              <Lines
                data={sweep.map((p) => ({ severity: `${Math.round(p.s * 100)}%`, top1: first(p.result)?.confidence ?? 0, ref: probOf(first(p.result), ref) }))}
                xKey="severity" xLabel="Severity"
                series={[{ key: 'ref', name: 'p(reference label)' }, { key: 'top1', name: 'Top-1 confidence (any label)' }]}
                fmt={(v) => pct(v, 0)} domain={[0, 1]} height={260}
              />
              <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
                {sweep.map((p) => {
                  const v = first(p.result)
                  const ok = v?.label === ref
                  return (
                    <figure key={p.s} className="overflow-hidden rounded-xl border border-line bg-page">
                      <img src={p.thumb} alt={`${Math.round(p.s * 100)}%`} className="aspect-[4/3] w-full object-cover" />
                      <figcaption className="space-y-1 p-2 text-[11px]">
                        <div className="flex items-center justify-between"><span className="font-semibold tabular text-ink">{Math.round(p.s * 100)}%</span>
                          {ok ? <IconCheck className="h-3.5 w-3.5 text-good-ink" /> : <IconXCircle className="h-3.5 w-3.5 text-critical-ink" />}</div>
                        <div className="line-clamp-2 text-ink-2">{v ? v.label : 'No vehicle'}</div>
                        <div className="tabular text-muted">{v ? pct(v.confidence, 0) : '—'}</div>
                      </figcaption>
                    </figure>
                  )
                })}
              </div>
              {sweep.length === LEVELS.length && <SweepSummary sweep={sweep} reference={ref} />}
            </Card>
          )}
        </>
      )}

      {!base && busy !== 'baseline' && (
        <Card>
          <div className="flex flex-col items-center gap-3 py-12 text-center">
            <IconFlask className="h-10 w-10 text-muted" />
            <div className="max-w-md text-sm text-ink-2">Pick a sample image above to begin. The lab runs a clean baseline first, then lets you stress it.</div>
          </div>
        </Card>
      )}
    </div>
  )
}

function Side({ title, p, reference, delta }: { title: string; p: Prediction; reference: string; delta?: number }) {
  const v = first(p)
  const fix = String(p.classical_info?.illumination_fix ?? 'none')
  return (
    <div className="min-w-0">
      <Label>{title}</Label>
      <img src={p.annotated_image} alt={title} className="w-full rounded-xl ring-1 ring-line" />
      {v ? (
        <div className="mt-3 space-y-2 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={v.status} />
            {v.label === reference ? <span className="inline-flex items-center gap-1 text-xs text-good-ink"><IconCheck className="h-3.5 w-3.5" /> matches reference</span>
              : <span className="inline-flex items-center gap-1 text-xs text-critical-ink"><IconXCircle className="h-3.5 w-3.5" /> differs</span>}
          </div>
          <div className="font-medium text-ink">{v.label}</div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2 tabular">
            <span>Confidence <b className="text-ink">{pct(v.confidence)}</b>{delta !== undefined && <span className={delta < 0 ? 'text-critical-ink' : 'text-good-ink'}> ({delta >= 0 ? '+' : ''}{(delta * 100).toFixed(1)} pts)</span>}</span>
            <span>Make <b className="text-ink">{v.make_guess} {pct(v.make_confidence, 0)}</b></span>
            <span>Detector <b className="text-ink">{v.det_source.replace(/_/g, ' ')}</b> {pct(v.det_conf, 0)}</span>
          </div>
          <div className="inline-flex items-center gap-1.5 rounded-md bg-surface-2 px-2 py-1 text-xs text-ink-2">
            <IconSun className="h-3.5 w-3.5" /> Illumination normalisation: <b className="text-ink">{fix}</b>
            {typeof p.classical_info?.brightness === 'number' && <span className="text-muted">· brightness {Math.round(p.classical_info.brightness as number)}</span>}
          </div>
        </div>
      ) : <p className="mt-3 text-sm text-ink-2">No vehicle detected.</p>}
    </div>
  )
}

function Verdict({ before, after, reference }: { before: Prediction; after: Prediction; reference: string }) {
  const a = first(before)
  const b = first(after)
  if (!a || !b) return null
  const kept = b.label === reference
  const makeKept = b.make_guess === (a.status === 'confident' ? a.make : a.make_guess)
  const fix = String(after.classical_info?.illumination_fix ?? 'none')
  return (
    <div className={`mt-5 flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${kept ? 'border-good/40 bg-good/8 text-good-ink' : makeKept ? 'border-warning/50 bg-warning/10 text-warning-ink' : 'border-critical/40 bg-critical/8 text-critical-ink'}`}>
      <IconArrowRight className="mt-0.5 h-4 w-4 shrink-0" />
      <span>
        {kept ? 'The system kept the correct answer under this condition.' : makeKept ? 'The exact model changed, but the make survived. ' : 'The prediction broke under this condition. '}
        {!kept && b.status !== 'confident' && 'It flagged its own uncertainty instead of reporting a confident wrong answer. '}
        {!kept && b.status === 'confident' && 'It was still confident, so this is a silent failure worth reporting. '}
        {fix !== 'none' && ` The classical illumination normalisation was triggered (${fix}).`}
      </span>
    </div>
  )
}

function SweepSummary({ sweep, reference }: { sweep: SweepPoint[]; reference: string }) {
  const breaking = sweep.find((p) => first(p.result)?.label !== reference)
  const confidentWrong = sweep.filter((p) => first(p.result)?.label !== reference && first(p.result)?.status === 'confident').length
  return (
    <p className="mt-4 rounded-xl bg-surface-2 px-4 py-3 text-sm text-ink-2">
      {breaking ? <>The prediction first changes at <b className="text-ink">{Math.round(breaking.s * 100)}%</b> severity.</> : <>The prediction survived <b className="text-ink">every level</b>, up to 100%.</>}
      {' '}{confidentWrong > 0 ? `${confidentWrong} level${confidentWrong > 1 ? 's' : ''} gave a confident wrong answer.` : 'No level produced a confident wrong answer.'}
    </p>
  )
}
