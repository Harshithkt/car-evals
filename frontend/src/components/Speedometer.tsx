import { useEffect, useId, useState } from 'react'
import type { ReactNode } from 'react'
import type { Status } from '../lib/api'

/**
 * Animated SVG automotive speedometer / tachometer dial.
 * Sweeps from -130deg to +130deg (260deg total arc), with tick marks,
 * redline / danger zones, glowing arc, animated needle, and digital readout.
 */
export function Speedometer({
  value,
  max = 1,
  label,
  sublabel,
  unit = '%',
  displayValue,
  color = '#21a4ff',
  size = 180,
  gear,
  redlineStart = 0.85,
  badge,
}: {
  /** Numeric value between 0 and `max` (default 0..1 for accuracy/confidence). */
  value: number | null | undefined
  max?: number
  label: string
  sublabel?: string
  unit?: string
  displayValue?: string
  color?: string
  size?: number
  gear?: string
  redlineStart?: number
  badge?: string
}) {
  const id = useId()
  const raw = typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(max, value)) : 0
  const ratio = max > 0 ? raw / max : 0
  const [animRatio, setAnimRatio] = useState(0)

  useEffect(() => {
    const t = requestAnimationFrame(() => setAnimRatio(ratio))
    return () => cancelAnimationFrame(t)
  }, [ratio])

  // Geometry in a 200x175 viewBox
  const cx = 100
  const cy = 102
  const r = 74
  const startDeg = -220 // 7:30 clock position
  const sweepDeg = 260  // sweeps clockwise to +40 deg (4:30 clock position)

  const polar = (deg: number, radius: number) => {
    const rad = (deg * Math.PI) / 180
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) }
  }

  const arcPath = (fromRatio: number, toRatio: number, radius = r) => {
    const a0 = startDeg + fromRatio * sweepDeg
    const a1 = startDeg + toRatio * sweepDeg
    const p0 = polar(a0, radius)
    const p1 = polar(a1, radius)
    const large = Math.abs(a1 - a0) > 180 ? 1 : 0
    return `M ${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A ${radius} ${radius} 0 ${large} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`
  }

  // Needle angle: -130deg (left) to +130deg (right) relative to straight up (0deg)
  const needleAngle = -130 + animRatio * 260
  const ticks = Array.from({ length: 21 }, (_, i) => i / 20)

  const formatted =
    displayValue ??
    (value == null || !Number.isFinite(value)
      ? '—'
      : unit === '%'
        ? `${(raw * (max === 1 ? 100 : 1)).toFixed(1)}`
        : `${raw.toFixed(1)}`)

  return (
    <div className="cockpit-dial group relative flex flex-col items-center rounded-2xl border border-line bg-gradient-to-b from-[#1e2126] to-[#121417] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] transition-all hover:border-accent/50">
      {/* Top telemetry bar */}
      <div className="mb-1 flex w-full items-center justify-between gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
          {badge ?? 'TELEMETRY'}
        </span>
        {gear && (
          <span className="rounded border border-white/15 bg-black/50 px-1.5 py-0.5 font-mono text-[10px] font-bold text-ink">
            {gear}
          </span>
        )}
      </div>

      <svg
        viewBox="0 0 200 168"
        style={{ width: size, height: Math.round(size * 0.84) }}
        className="overflow-visible select-none"
        aria-label={`${label}: ${formatted}${unit}`}
      >
        <defs>
          <linearGradient id={`grad-${id}`} x1="0%" y1="100%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#21a4ff" />
            <stop offset="65%" stopColor={color} />
            <stop offset="100%" stopColor="#00f283" />
          </linearGradient>
          <radialGradient id={`face-${id}`} cx="50%" cy="54%" r="50%">
            <stop offset="0%" stopColor="#1b222c" stopOpacity="0.9" />
            <stop offset="72%" stopColor="#111317" stopOpacity="0.95" />
            <stop offset="100%" stopColor="#0a0b0d" stopOpacity="1" />
          </radialGradient>
          <filter id={`glow-${id}`} x="-25%" y="-25%" width="150%" height="150%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Outer metallic bezel */}
        <circle cx={cx} cy={cy} r={r + 13} fill={`url(#face-${id})`} stroke="#2b2f36" strokeWidth="1.5" />
        <circle cx={cx} cy={cy} r={r + 10} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="3 5" />

        {/* Background track arc */}
        <path
          d={arcPath(0, 1)}
          fill="none"
          stroke="#25282e"
          strokeWidth="9"
          strokeLinecap="round"
        />

        {/* High-performance redline / peak zone underlay */}
        <path
          d={arcPath(redlineStart, 1)}
          fill="none"
          stroke="rgba(0, 242, 131, 0.16)"
          strokeWidth="9"
          strokeLinecap="round"
        />

        {/* Active glowing value arc */}
        {animRatio > 0.005 && (
          <path
            d={arcPath(0, Math.max(0.01, animRatio))}
            fill="none"
            stroke={`url(#grad-${id})`}
            strokeWidth="9"
            strokeLinecap="round"
            filter={`url(#glow-${id})`}
            style={{ transition: 'd 0.9s cubic-bezier(0.22, 1, 0.36, 1)' }}
          />
        )}

        {/* Tick marks & numbers around dial */}
        {ticks.map((t, i) => {
          const deg = startDeg + t * sweepDeg
          const isMajor = i % 5 === 0
          const isMid = i % 2 === 0
          const pOuter = polar(deg, r - 8)
          const pInner = polar(deg, r - (isMajor ? 18 : isMid ? 14 : 11))
          const pText = polar(deg, r - 27)
          const active = t <= animRatio + 0.001
          return (
            <g key={i}>
              <line
                x1={pOuter.x}
                y1={pOuter.y}
                x2={pInner.x}
                y2={pInner.y}
                stroke={active ? (t >= redlineStart ? '#00f283' : '#e5eef7') : isMajor ? '#68707a' : '#3a3f47'}
                strokeWidth={isMajor ? 2 : 1}
                strokeLinecap="round"
              />
              {isMajor && (
                <text
                  x={pText.x}
                  y={pText.y + 3}
                  textAnchor="middle"
                  className="fill-[#7c8591] font-mono text-[8px] font-semibold"
                >
                  {Math.round(t * (max === 1 ? 100 : max))}
                </text>
              )}
            </g>
          )
        })}

        {/* Inner HUD ring */}
        <circle cx={cx} cy={cy} r={34} fill="#0d0f12" stroke="#262a30" strokeWidth="1.2" />

        {/* Animated speedometer needle */}
        <g
          transform={`translate(${cx}, ${cy}) rotate(${needleAngle})`}
          style={{ transition: 'transform 1s cubic-bezier(0.22, 1, 0.36, 1)' }}
        >
          {/* Needle shadow */}
          <polygon points="-2.5,10 0,-64 2.5,10" fill="rgba(0,0,0,0.6)" transform="translate(1, 2)" />
          {/* Needle body */}
          <polygon points="-2.2,10 0,-64 2.2,10" fill="#ff4d4d" />
          <polygon points="-1,0 0,-62 1,0" fill="#ffd6d6" />
        </g>

        {/* Center cap */}
        <circle cx={cx} cy={cy} r={8} fill="#252930" stroke="#57606a" strokeWidth="1.5" />
        <circle cx={cx} cy={cy} r={3.2} fill={color} />

        {/* Digital Odometer / Speed Readout */}
        <rect x={cx - 42} y={cy + 26} width={84} height={32} rx={7} fill="#090b0d" stroke="#2c313a" strokeWidth="1" />
        <text
          x={cx}
          y={cy + 46}
          textAnchor="middle"
          className="fill-white font-mono text-[17px] font-bold tracking-tight tabular"
        >
          {formatted}
          <tspan className="fill-[#8a94a0] text-[10px] font-semibold">{unit}</tspan>
        </text>
      </svg>

      <div className="mt-1 text-center">
        <div className="text-[13px] font-semibold tracking-[-0.01em] text-ink">{label}</div>
        {sublabel && <div className="mt-0.5 text-[11px] text-muted">{sublabel}</div>}
      </div>
    </div>
  )
}

/**
 * RPM Shift-Light Strip (F1 / GT3 steering-wheel LED bar).
 * Lights up green -> amber -> electric blue/redline as confidence / accuracy increases.
 */
export function ShiftLights({
  value,
  segments = 16,
  label,
  rightText,
}: {
  value: number
  segments?: number
  label?: ReactNode
  rightText?: ReactNode
}) {
  const clamped = Math.max(0, Math.min(1, value))
  const lit = Math.round(clamped * segments)

  const segColor = (idx: number) => {
    const f = (idx + 1) / segments
    if (f < 0.55) return '#00f283' // green RPM range
    if (f < 0.82) return '#fab219' // amber mid-high RPM
    return '#21a4ff'               // shift / peak blue-cyan
  }

  return (
    <div>
      {(label || rightText) && (
        <div className="mb-1.5 flex items-center justify-between text-[11px] font-medium uppercase tracking-[0.1em] text-muted">
          <span>{label}</span>
          <span className="font-mono text-ink-2 tabular">{rightText}</span>
        </div>
      )}
      <div
        className="flex items-center gap-1 rounded-xl border border-line bg-[#0c0e11] p-1.5 shadow-inner"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(clamped * 100)}
      >
        {Array.from({ length: segments }, (_, i) => {
          const on = i < lit
          const c = segColor(i)
          return (
            <span
              key={i}
              className="h-3 flex-1 rounded-[3px] transition-all duration-300"
              style={{
                background: on ? c : '#1e2228',
                boxShadow: on ? `0 0 8px ${c}88` : 'none',
                opacity: on ? 1 : 0.45,
              }}
            />
          )
        })}
      </div>
    </div>
  )
}

/**
 * Digital Gear & Dashboard Indicator Pill (PRND / S+ / CHECK ENGINE style).
 */
export function GearIndicator({ status }: { status: Status }) {
  const gears: { id: string; code: string; active: boolean; color: string; title: string }[] = [
    { id: 'low_confidence', code: 'L', active: status === 'low_confidence', color: '#ff5d5d', title: 'Low-confidence gear (< τ_make)' },
    { id: 'uncertain_model', code: 'M', active: status === 'uncertain_model', color: '#fab219', title: 'Make-level gear (≥ τ_make)' },
    { id: 'confident', code: 'S+', active: status === 'confident', color: '#00f283', title: 'Sport+ full-confidence gear (≥ τ)' },
  ]
  return (
    <div className="inline-flex items-center gap-1 rounded-xl border border-line bg-[#0b0d10] px-2.5 py-1 text-xs" title="Confidence transmission mode">
      <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">GEAR</span>
      {gears.map((g) => (
        <span
          key={g.id}
          title={g.title}
          className={`rounded px-1.5 py-0.5 font-mono text-[11px] font-bold transition ${
            g.active ? 'text-[#081014]' : 'text-muted/50'
          }`}
          style={g.active ? { background: g.color, boxShadow: `0 0 10px ${g.color}80` } : undefined}
        >
          {g.code}
        </span>
      ))}
    </div>
  )
}

/**
 * Automotive VIN / Chassis Spec Plate for detected vehicles.
 */
export function ChassisPlate({
  make,
  model,
  year,
  body,
  classId,
  detSource,
  detConf,
  box,
}: {
  make: string
  model: string
  year: string
  body: string
  classId: number
  detSource: string
  detConf: number
  box: [number, number, number, number]
}) {
  const code = `P5-${make.slice(0, 3).toUpperCase()}-${String(classId + 1).padStart(3, '0')}-${year || '0000'}`
  const w = Math.round(box[2] - box[0])
  const h = Math.round(box[3] - box[1])
  return (
    <div className="relative overflow-hidden rounded-xl border border-white/15 bg-gradient-to-r from-[#14171c] via-[#191d24] to-[#14171c] px-4 py-3 font-mono text-xs shadow-inner">
      {/* Corner rivets */}
      <span className="absolute left-2 top-2 h-1.5 w-1.5 rounded-full bg-white/25" />
      <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-white/25" />
      <span className="absolute bottom-2 left-2 h-1.5 w-1.5 rounded-full bg-white/25" />
      <span className="absolute bottom-2 right-2 h-1.5 w-1.5 rounded-full bg-white/25" />

      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-2">
        <div className="flex items-center gap-2">
          <span className="rounded bg-accent/20 px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-accent">
            CHASSIS ID
          </span>
          <span className="font-bold tracking-wider text-ink">{code}</span>
        </div>
        <span className="text-[11px] text-muted">
          BODY: <strong className="text-ink-2">{body.toUpperCase()}</strong> · ROI: <strong className="text-ink-2">{w}×{h}px</strong>
        </span>
      </div>

      <div className="mt-2.5 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted">MANUFACTURER</div>
          <div className="mt-0.5 truncate font-sans text-sm font-semibold text-ink">{make}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted">MODEL TRIM</div>
          <div className="mt-0.5 truncate font-sans text-sm font-semibold text-ink">{model}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted">MODEL YEAR</div>
          <div className="mt-0.5 font-sans text-sm font-semibold text-ink">{year || '—'}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted">RADAR / SENSOR</div>
          <div className="mt-0.5 truncate font-sans text-sm font-semibold text-good-ink">
            {detSource} · {Math.round(detConf * 100)}%
          </div>
        </div>
      </div>
    </div>
  )
}
