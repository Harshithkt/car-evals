import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { ClassMeta, LLMOpinion, Status, Vehicle } from '../lib/api'
import { pct } from '../lib/format'
import { bodyType, indexByName, runnerUpInsight } from '../lib/classes'
import { BarList } from './charts'
import { IconAlert, IconCheck, IconHelp, IconInfo, IconSliders, IconSparkles } from './icons'
import { Card, DET_SOURCE, Label, Meter, STATUS_COLOR, StatusBadge } from './ui'

const STATUS_BOX: Record<Status, string> = {
  confident: 'border-good/40 bg-good/8 text-good-ink',
  uncertain_model: 'border-warning/50 bg-warning/10 text-warning-ink',
  low_confidence: 'border-critical/40 bg-critical/8 text-critical-ink',
}

/** Same rule as inference.py `decide()`: p(top-1) ≥ τ → confident, else p(make) ≥ τ_make → uncertain model, else low confidence. */
export const decideStatus = (conf: number, makeConf: number, tau: number, tauMake: number): Status =>
  conf >= tau ? 'confident' : makeConf >= tauMake ? 'uncertain_model' : 'low_confidence'

/** The vehicle cropped out of the original image with CSS only (no canvas, no extra request). */
export function CropThumb({ src, box, size, className = '' }: { src: string; box: number[]; size: [number, number]; className?: string }) {
  const [x1, y1, x2, y2] = box
  const [W, H] = size
  const bw = Math.max(1, x2 - x1)
  const bh = Math.max(1, y2 - y1)
  return (
    <div
      role="img"
      aria-label="Detected vehicle crop"
      className={`rounded-lg bg-surface-2 bg-no-repeat ring-1 ring-line ${className}`}
      style={{
        aspectRatio: `${bw} / ${bh}`,
        backgroundImage: `url("${src}")`,
        backgroundSize: `${(W / bw) * 100}% ${(H / bh) * 100}%`,
        backgroundPosition: `${W > bw ? (x1 / (W - bw)) * 100 : 0}% ${H > bh ? (y1 / (H - bh)) * 100 : 0}%`,
      }}
    />
  )
}

/** One LLM call per vehicle per analysis, shared by every render of that card (switching tabs does not ask again). */
const LLM_CACHE = new Map<string, Promise<LLMOpinion>>()
const known = (s?: string) => !!s && s.trim().toLowerCase() !== 'unknown'

function useLLMOpinion(key: string, v: Vehicle, preview: string | null | undefined, imageSize: [number, number] | undefined, auto: boolean) {
  const [res, setRes] = useState<LLMOpinion | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const ask = useCallback(async () => {
    if (!preview || !imageSize) return
    setBusy(true)
    setErr(null)
    try {
      let p = LLM_CACHE.get(key)
      if (!p) {
        p = cropBlob(preview, v.box, imageSize).then((b) => api.llmOpinion(b))
        LLM_CACHE.set(key, p)
        p.catch(() => LLM_CACHE.delete(key))   // a failed call may be retried
      }
      setRes(await p)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [key, preview, imageSize, v.box])
  useEffect(() => {
    if (auto) void ask()
  }, [auto, ask])
  return { res, err, busy, ask }
}

export function VehicleCard({ v, index, total, preview, imageSize, classes, thresholds, llm = false, autoLLM = false, cacheKey }: {
  v: Vehicle
  /** the server has a vision LLM configured (optional second opinion) */
  llm?: boolean
  /** ask the LLM automatically when the trained model's confidence is low (off for live camera frames) */
  autoLLM?: boolean
  /** identifies this vehicle in this analysis, so the LLM is asked only once */
  cacheKey?: string
  index: number
  total: number
  preview?: string | null
  imageSize?: [number, number]
  classes: ClassMeta[]
  thresholds: { tau?: number; tau_make?: number }
}) {
  const src = DET_SOURCE[v.det_source] ?? { label: v.det_source }
  const confident = v.status === 'confident'
  const StatusIcon = confident ? IconCheck : v.status === 'uncertain_model' ? IconInfo : IconAlert
  const byName = useMemo(() => indexByName(classes), [classes])
  const insight = useMemo(() => runnerUpInsight(v.top5, byName), [v.top5, byName])
  const body = bodyType(v.model)

  // Low confidence usually means the car is outside the trained classes: then the vision LLM's answer leads the card.
  const autoAsk = llm && autoLLM && v.status === 'low_confidence' && !!preview && !!imageSize
  const op = useLLMOpinion(cacheKey ?? `${v.box.join(',')}-${v.label}`, v, preview, imageSize, autoAsk)
  const a = op.res?.answer
  const llmLead = autoAsk && (op.busy || (!!a && known(a.make)))
  const llmShort = op.res?.model.split('/').pop()

  return (
    <Card className="print:break-inside-avoid">
      <div className="flex flex-wrap items-start gap-4">
        {preview && imageSize && <CropThumb src={preview} box={v.box} size={imageSize} className="w-28 shrink-0 sm:w-36" />}
        <div className="min-w-0 flex-1">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            {total > 1 && <span className="rounded-md bg-surface-2 px-2 py-0.5 text-xs font-medium text-ink-2">Vehicle {index + 1}</span>}
            {llmLead
              ? <span className="inline-flex items-center gap-1 rounded-md border border-ink/30 bg-surface-2 px-2 py-0.5 text-xs font-medium text-ink"><IconSparkles className="h-3.5 w-3.5" /> Vision LLM answer</span>
              : <StatusBadge status={v.status} />}
            {!llmLead && body !== 'Other' && <span className="rounded-md border border-line px-2 py-0.5 text-xs text-ink-2">{body}</span>}
          </div>
          {llmLead ? (
            <>
              <h3 className={`text-xl font-semibold leading-tight tracking-[-0.015em] text-ink sm:text-2xl ${op.busy ? 'animate-pulse text-ink-2' : ''}`}>
                {op.busy ? 'Asking a vision LLM…' : [a!.make, known(a!.model) ? a!.model : ''].filter(Boolean).join(' ')}
              </h3>
              <div className="mt-1 text-sm text-ink-2">
                {op.busy ? 'The trained model was not sure, so a vision LLM is looking at this car.' : [a!.generation_or_years, a!.body_type].filter(known).join(' · ')}
              </div>
            </>
          ) : (
            <>
              <h3 className="text-xl font-semibold leading-tight tracking-[-0.015em] text-ink sm:text-2xl">
                {confident ? `${v.make} ${v.model}` : <>Likely a {v.make_guess}</>}
              </h3>
              <div className="mt-1 text-sm text-ink-2">{confident ? v.year : `Best guess: ${v.label}`}</div>
            </>
          )}
        </div>
        <div className="text-right">
          {llmLead ? (
            <>
              <div className="text-2xl font-semibold capitalize leading-none tracking-[-0.02em] text-ink">{op.busy ? '…' : a!.confidence || 'n/a'}</div>
              <div className="mt-1 text-xs text-muted">LLM's own confidence</div>
            </>
          ) : (
            <>
              <div className="text-3xl font-semibold leading-none tracking-[-0.02em] text-ink">{pct(confident ? v.confidence : v.make_confidence)}</div>
              <div className="mt-1 text-xs text-muted">{confident ? 'class confidence' : 'make confidence'}</div>
            </>
          )}
        </div>
      </div>

      {!llmLead && (
        <div className="mt-4">
          <Meter value={v.confidence} color={STATUS_COLOR[v.status]} />
        </div>
      )}

      <dl className="mt-5 grid grid-cols-[1fr_1.5fr_0.8fr] gap-2 text-sm sm:gap-3">
        {(llmLead && a
          ? [['Make', a.make, 'from the LLM'], ['Model', known(a.model) ? a.model : '—', null], ['Years', known(a.generation_or_years) ? a.generation_or_years : '—', null]]
          : [['Make', v.make, `${pct(v.make_confidence)} make conf.`], ['Model', v.model, null], ['Year', v.year || '—', null]]
        ).map(([k, val, note]) => (
          <div key={k} className="min-w-0 rounded-xl border border-line bg-page px-3.5 py-2.5">
            <dt className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">{k}</dt>
            <dd className={`mt-0.5 break-words font-medium ${confident || (llmLead && a) ? 'text-ink' : 'text-ink-2'}`}>{op.busy && llmLead ? '…' : val}</dd>
            {note && <dd className="text-[11px] text-muted">{note}</dd>}
          </div>
        ))}
      </dl>

      {llmLead ? (
        <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-line bg-surface-2/60 px-3.5 py-2.5 text-sm text-ink-2">
          <IconInfo className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            <span className="font-medium text-ink">Trained model was not sure:</span> best guess {v.label} ({pct(v.confidence)}, low confidence).
            This car may be outside its known classes, so the answer above comes from {llmShort ? <b className="font-medium text-ink">{llmShort}</b> : 'a vision LLM'}.
            The LLM is not part of the evaluated system and can be wrong.
            {a?.clues && <span className="mt-1 block text-[13px]">LLM's clues: {a.clues}</span>}
            {op.res && <span className="mt-1 block text-[11px] text-muted">{op.res.model} · {(op.res.latency_ms / 1000).toFixed(1)} s</span>}
          </span>
        </div>
      ) : (
        <div className={`mt-4 flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-sm ${STATUS_BOX[v.status]}`}>
          <StatusIcon className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{v.message}</span>
        </div>
      )}

      <div className="mt-6 grid gap-6 border-t border-grid pt-5 md:grid-cols-[minmax(0,1fr)_minmax(0,15rem)]">
        <div className="min-w-0">
          <Label>Top-5 predictions</Label>
          <BarList items={v.top5.map((t) => ({ label: t.label, value: t.confidence }))} fmt={(x) => pct(x)} max={1} highlight={(l) => l === v.top5[0]?.label} />
        </div>
        <div className="text-sm md:border-l md:border-grid md:pl-6">
          <Label>Detector</Label>
          <dl className="space-y-2">
            <div className="flex justify-between gap-2"><dt className="text-ink-2">Source</dt><dd className="text-right font-medium text-ink">{src.label}</dd></div>
            <div className="flex justify-between gap-2"><dt className="text-ink-2">Confidence</dt><dd className="tabular font-medium text-ink">{pct(v.det_conf)}</dd></div>
            <div className="flex justify-between gap-2"><dt className="text-ink-2">Box (px)</dt><dd className="tabular text-xs text-ink-2">{v.box.map((b) => Math.round(b)).join(', ')}</dd></div>
          </dl>
          {src.note && (
            <div className="mt-3 flex gap-2 rounded-lg border border-warning/50 bg-warning/10 px-2.5 py-2 text-xs text-warning-ink">
              <IconAlert className="mt-px h-3.5 w-3.5 shrink-0" /> {src.note}
            </div>
          )}
        </div>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        {insight && insight.text && <RunnerUp insight={insight} />}
        <WhatIf v={v} thresholds={thresholds} />
      </div>
      {llm && preview && imageSize && !llmLead && <LLMSecondOpinion v={v} op={op} />}
    </Card>
  )
}

const KIND_LABEL = { year: 'Year confusion', model: 'Same make', lookalike: 'Look-alike', different: 'Unrelated' } as const

function RunnerUp({ insight }: { insight: NonNullable<ReturnType<typeof runnerUpInsight>> }) {
  const tone = insight.strength === 'narrow' ? 'text-warning-ink' : 'text-good-ink'
  return (
    <div className="rounded-xl border border-line bg-page p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-ink"><IconHelp className="h-4 w-4 text-ink" /> Why not the runner-up?</div>
      <div className="mt-2 text-[13px] text-ink-2">
        <span className="font-medium text-ink">{insight.runnerUp}</span>
        <span className="ml-1.5 rounded-full border border-line px-1.5 py-px text-[11px]">{KIND_LABEL[insight.kind]}</span>
      </div>
      <p className="mt-2 text-[13px] leading-relaxed text-ink-2">{insight.text}</p>
      <div className="mt-3 text-xs text-muted">
        Margin over runner-up: <span className={`font-semibold tabular ${tone}`}>{pct(insight.margin)}</span> ({insight.strength})
      </div>
    </div>
  )
}

function WhatIf({ v, thresholds }: { v: Vehicle; thresholds: { tau?: number; tau_make?: number } }) {
  const t0 = thresholds.tau ?? 0.5
  const m0 = thresholds.tau_make ?? 0.5
  const [tau, setTau] = useState(t0)
  const [tauMake, setTauMake] = useState(m0)
  const status = decideStatus(v.confidence, v.make_confidence, tau, tauMake)
  const changed = status !== v.status
  return (
    <div className="rounded-xl border border-line bg-page p-4 print:hidden">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-medium text-ink"><IconSliders className="h-4 w-4 text-ink" /> Threshold playground</div>
        {(tau !== t0 || tauMake !== m0) && (
          <button onClick={() => { setTau(t0); setTauMake(m0) }} className="text-xs text-ink underline decoration-mist underline-offset-4 hover:decoration-ink">Reset to deployed</button>
        )}
      </div>
      <p className="mt-1 text-xs text-muted">Re-applies the system's decision rule to this vehicle in your browser. The model is not re-run.</p>
      <Slider label="τ (class)" value={tau} onChange={setTau} marker={v.confidence} markerLabel="p(class)" />
      <Slider label="τ_make" value={tauMake} onChange={setTauMake} marker={v.make_confidence} markerLabel="p(make)" />
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-ink-2">
        Decision: <StatusBadge status={status} />
        {changed && <span className="text-muted">(deployed: {v.status.replace('_', ' ')})</span>}
      </div>
    </div>
  )
}

function Slider({ label, value, onChange, marker, markerLabel }: { label: string; value: number; onChange: (v: number) => void; marker: number; markerLabel: string }) {
  return (
    <label className="mt-3 block">
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-ink-2">{label}</span>
        <span className="tabular font-medium text-ink">{value.toFixed(2)}</span>
      </div>
      <div className="relative">
        <input type="range" min={0} max={1} step={0.01} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
        <span className="pointer-events-none absolute -top-0.5 h-5 w-0.5 rounded bg-ink/60" style={{ left: `calc(${marker * 100}% - 1px)` }} title={`${markerLabel} = ${marker.toFixed(3)}`} />
      </div>
      <div className="text-[11px] text-muted">{markerLabel} = {pct(marker)} (dark tick)</div>
    </label>
  )
}

/** Crop the vehicle out of the original upload (8% padding, longest side at most 768 px) as a JPEG blob. */
async function cropBlob(src: string, box: number[], size: [number, number]): Promise<Blob> {
  const img = new Image()
  img.src = src
  await img.decode()
  const [W, H] = size
  const sx = img.naturalWidth / W
  const sy = img.naturalHeight / H
  const [x1, y1, x2, y2] = box
  const px = (x2 - x1) * 0.08
  const py = (y2 - y1) * 0.08
  const cx1 = Math.max(0, x1 - px) * sx
  const cy1 = Math.max(0, y1 - py) * sy
  const cw = (Math.min(W, x2 + px) * sx) - cx1
  const ch = (Math.min(H, y2 + py) * sy) - cy1
  const s = Math.min(1, 768 / Math.max(cw, ch))
  const c = document.createElement('canvas')
  c.width = Math.max(1, Math.round(cw * s))
  c.height = Math.max(1, Math.round(ch * s))
  c.getContext('2d')!.drawImage(img, cx1, cy1, cw, ch, 0, 0, c.width, c.height)
  return new Promise((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not crop the image.'))), 'image/jpeg', 0.9))
}

/** Optional: ask a vision LLM about this crop. Useful for cars outside the trained class list; never used for the metrics.
 *  Shares its state with the card (`op`), so an automatic low-confidence call and this button never ask twice. */
function LLMSecondOpinion({ v, op }: { v: Vehicle; op: ReturnType<typeof useLLMOpinion> }) {
  const { res, err, busy, ask } = op
  const a = res?.answer
  const sameMake = a && a.make && a.make.toLowerCase() !== 'unknown'
    ? a.make.toLowerCase().includes(v.make.toLowerCase()) || v.make.toLowerCase().includes(a.make.toLowerCase().split(' ').pop() ?? '')
    : null
  return (
    <div className="mt-4 rounded-xl border border-dashed border-line bg-page p-4 print:hidden">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-sm font-medium text-ink"><IconSparkles className="h-4 w-4" /> Second opinion from a vision LLM</div>
          <p className="mt-0.5 text-xs text-muted">
            {v.status === 'confident' ? 'Optional. ' : 'Useful when the car may not be in the trained list (for example India-only models). '}
            Sends only this crop to an LLM on Nebius. Its answer is not part of the evaluated system and can be wrong.
          </p>
        </div>
        {!res && (
          <button onClick={() => void ask()} disabled={busy}
            className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-ink px-3 py-1.5 text-sm font-medium text-ink hover:bg-surface-2 disabled:opacity-50">
            {busy ? 'Asking…' : 'Ask the LLM'}
          </button>
        )}
      </div>
      {err && <div className="mt-3 rounded-lg bg-critical/8 px-3 py-2 text-xs text-critical-ink">{err}</div>}
      {res && (
        <div className="mt-3 text-sm">
          {a ? (
            <>
              <div className="text-[15px] font-semibold text-ink">{[a.make, a.model].filter((x) => x && x.toLowerCase() !== 'unknown').join(' ') || 'Not sure'}</div>
              <div className="text-ink-2">{[a.generation_or_years, a.body_type].filter((x) => x && x.toLowerCase() !== 'unknown').join(' · ')}</div>
              {a.clues && <p className="mt-1.5 text-[13px] text-ink-2">{a.clues}</p>}
              <div className="mt-2 flex flex-wrap gap-2 text-[11px]">
                <span className="rounded-full border border-line px-2 py-0.5 text-ink-2">LLM confidence: {a.confidence || 'n/a'}</span>
                {sameMake !== null && (
                  <span className={`rounded-full px-2 py-0.5 ${sameMake ? 'bg-good/10 text-good-ink' : 'bg-warning/15 text-warning-ink'}`}>
                    {sameMake ? `agrees with the classifier on the make (${v.make})` : `differs from the classifier (${v.make})`}
                  </span>
                )}
              </div>
            </>
          ) : <p className="whitespace-pre-wrap text-[13px] text-ink-2">{res.text}</p>}
          <div className="mt-2 text-[11px] text-muted">{res.model} · {(res.latency_ms / 1000).toFixed(1)} s</div>
        </div>
      )}
    </div>
  )
}
