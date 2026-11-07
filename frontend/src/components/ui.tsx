import type { ReactNode } from 'react'
import type { DetSource, Status } from '../lib/api'

export function Card({ title, subtitle, children, actions, className = '' }: {
  title?: ReactNode
  subtitle?: ReactNode
  children: ReactNode
  actions?: ReactNode
  className?: string
}) {
  return (
    <section className={`min-w-0 rounded-2xl border border-line bg-surface p-6 sm:p-8 ${className}`}>
      {(title || actions) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            {title && <h2 className="text-[20px] font-semibold leading-[1.4] tracking-[-0.01em] text-ink">{title}</h2>}
            {subtitle && <p className="mt-1 text-[14px] leading-[1.43] text-ink-2">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

export function PageHeader({ title, eyebrow, children, actions }: {
  title: string
  eyebrow?: string
  children?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && <div className="mb-1.5 text-[14px] font-medium text-muted">{eyebrow}</div>}
        <h1 className="text-[36px] font-semibold leading-[1.21] tracking-[-0.02em] text-ink sm:text-[48px] sm:leading-[1.2] sm:tracking-[-0.025em]">{title}</h1>
        {children && <p className="mt-3 max-w-[40rem] text-[16px] leading-[1.5] text-ink-2">{children}</p>}
      </div>
      {actions}
    </div>
  )
}

/** Small section label used inside cards. */
export function Label({ children }: { children: ReactNode }) {
  return <div className="mb-2 text-[12px] font-medium text-muted">{children}</div>
}

/** Empty state for a metrics.json key or plot that the notebook run did not produce. */
export function Missing({ what, hint }: { what: string; hint?: string }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-dashed border-line bg-surface-2/60 px-4 py-4 text-sm text-ink-2">
      <span className="mt-0.5 text-muted">
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" strokeLinecap="round" /></svg>
      </span>
      <div>
        <span className="font-medium text-ink">{what}</span> is not available yet.
        {hint && <span className="mt-0.5 block text-[13px] text-muted">{hint}</span>}
      </div>
    </div>
  )
}

export function StatTile({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-line bg-surface px-4 py-3.5">
      <div className="text-[13px] text-ink-2">{label}</div>
      <div className="mt-1 text-[26px] font-semibold leading-tight tracking-[-0.02em] text-ink">{value}</div>
      {note && <div className="mt-1 text-xs text-muted">{note}</div>}
    </div>
  )
}

const STATUS: Record<Status, { label: string; cls: string; icon: ReactNode }> = {
  confident: {
    label: 'Confident',
    cls: 'bg-good/12 text-good-ink border-good/40',
    icon: <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />,
  },
  uncertain_model: {
    label: 'Uncertain model / year',
    cls: 'bg-warning/15 text-warning-ink border-warning/50',
    icon: <path d="M8 4.5v4.5M8 11.6v.1" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />,
  },
  low_confidence: {
    label: 'Low confidence',
    cls: 'bg-critical/12 text-critical-ink border-critical/40',
    icon: <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />,
  },
}

export const STATUS_COLOR: Record<Status, string> = { confident: 'var(--good)', uncertain_model: 'var(--warning)', low_confidence: 'var(--critical)' }

export function StatusBadge({ status }: { status: Status }) {
  const s = STATUS[status] ?? STATUS.low_confidence
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[12px] font-medium ${s.cls}`}>
      <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" aria-hidden>{s.icon}</svg>
      {s.label}
    </span>
  )
}

export const DET_SOURCE: Record<DetSource, { label: string; note?: string }> = {
  yolo: { label: 'YOLOv8 (fine-tuned)' },
  yolo_low_confidence: {
    label: 'YOLOv8, below threshold',
    note: 'No box passed the detector threshold, so the best low-confidence YOLO box was used.',
  },
  classical_roi_fallback: {
    label: 'Classical contour ROI',
    note: 'YOLO found no vehicle; the region came from the classical edge/contour fallback.',
  },
  full_image_fallback: {
    label: 'Whole image',
    note: 'Neither YOLO nor the classical ROI found a vehicle; the whole image was classified. Results may be unreliable.',
  },
}

/** Horizontal meter: value 0–1. */
export function Meter({ value, color = 'var(--series-1)' }: { value: number; color?: string }) {
  const w = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="h-2 w-full rounded-full bg-surface-2" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(w)}>
      <div className="h-2 rounded-full" style={{ width: `${w}%`, background: color }} />
    </div>
  )
}

export function Toggle({ checked, onChange, label, disabled }: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  disabled?: boolean
}) {
  return (
    <label className={`inline-flex items-center gap-2 text-sm select-none ${disabled ? 'opacity-50' : 'cursor-pointer'}`}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 rounded-full transition-colors ${checked ? 'bg-ink' : 'bg-mist'}`}
      >
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${checked ? 'left-[18px]' : 'left-0.5'}`} />
      </button>
      <span className="text-ink-2">{label}</span>
    </label>
  )
}
