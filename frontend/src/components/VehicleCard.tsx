import { useMemo, useState } from 'react'
import type { ClassMeta, Status, Vehicle } from '../lib/api'
import { pct } from '../lib/format'
import { bodyType, indexByName, runnerUpInsight } from '../lib/classes'
import { BarList } from './charts'
import { IconAlert, IconCheck, IconHelp, IconInfo, IconSliders } from './icons'
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

export function VehicleCard({ v, index, total, preview, imageSize, classes, thresholds }: {
  v: Vehicle
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

  return (
    <Card className="print:break-inside-avoid">
      <div className="flex flex-wrap items-start gap-4">
        {preview && imageSize && <CropThumb src={preview} box={v.box} size={imageSize} className="w-28 shrink-0 sm:w-36" />}
        <div className="min-w-0 flex-1">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            {total > 1 && <span className="rounded-md bg-surface-2 px-2 py-0.5 text-xs font-medium text-ink-2">Vehicle {index + 1}</span>}
            <StatusBadge status={v.status} />
            {body !== 'Other' && <span className="rounded-md border border-line px-2 py-0.5 text-xs text-ink-2">{body}</span>}
          </div>
          <h3 className="text-xl font-semibold leading-tight tracking-[-0.015em] text-ink sm:text-2xl">
            {confident ? `${v.make} ${v.model}` : <>Likely a {v.make_guess}</>}
          </h3>
          <div className="mt-1 text-sm text-ink-2">{confident ? v.year : `Best guess: ${v.label}`}</div>
        </div>
        <div className="text-right">
          <div className="text-3xl font-semibold leading-none tracking-[-0.02em] text-ink">{pct(confident ? v.confidence : v.make_confidence)}</div>
          <div className="mt-1 text-xs text-muted">{confident ? 'class confidence' : 'make confidence'}</div>
        </div>
      </div>

      <div className="mt-4">
        <Meter value={v.confidence} color={STATUS_COLOR[v.status]} />
      </div>

      <dl className="mt-5 grid grid-cols-[1fr_1.5fr_0.8fr] gap-2 text-sm sm:gap-3">
        {[['Make', v.make, `${pct(v.make_confidence)} make conf.`], ['Model', v.model, null], ['Year', v.year || '—', null]].map(([k, val, note]) => (
          <div key={k} className="min-w-0 rounded-xl border border-line bg-page px-3.5 py-2.5">
            <dt className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">{k}</dt>
            <dd className={`mt-0.5 break-words font-medium ${confident ? 'text-ink' : 'text-ink-2'}`}>{val}</dd>
            {note && <dd className="text-[11px] text-muted">{note}</dd>}
          </div>
        ))}
      </dl>

      <div className={`mt-4 flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-sm ${STATUS_BOX[v.status]}`}>
        <StatusIcon className="mt-0.5 h-4 w-4 shrink-0" />
        <span>{v.message}</span>
      </div>

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
