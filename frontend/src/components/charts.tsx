import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { TooltipContentProps } from 'recharts'

/** Fixed categorical order; colour follows the series' position in its (stable) list, never its rank. */
export const SERIES = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)']

export interface Series {
  key: string
  name: string
}

type Fmt = (v: number) => string

function ChartTooltip({ active, payload, label, fmt }: TooltipContentProps<number, string> & { fmt: Fmt }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-medium text-ink">{label}</div>
      {payload.map((p) => (
        <div key={String(p.dataKey)} className="flex items-center gap-2 text-ink-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="flex-1">{p.name}</span>
          <span className="tabular font-medium text-ink">{typeof p.value === 'number' ? fmt(p.value) : '—'}</span>
        </div>
      ))}
    </div>
  )
}

const pctTick = (v: number) => `${Math.round(v * 100)}%`

/** Vertical grouped columns: one group per category, one bar per series (≤ 4 series). */
export function GroupedBars({ data, series, categoryKey = 'category', fmt, height = 280, domain = [0, 1], xInterval = 0 }: {
  data: Record<string, string | number | null>[]
  series: Series[]
  categoryKey?: string
  fmt: Fmt
  height?: number
  domain?: [number, number | 'auto']
  xInterval?: number | 'preserveStartEnd'
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barGap={2} barCategoryGap="22%">
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey={categoryKey} tickLine={false} axisLine={{ stroke: 'var(--axis)' }} interval={xInterval} minTickGap={8} />
          <YAxis domain={domain} tickFormatter={domain[1] === 1 ? pctTick : undefined} tickLine={false} axisLine={false} width={44} />
          <Tooltip content={(p) => <ChartTooltip {...(p as TooltipContentProps<number, string>)} fmt={fmt} />} cursor={{ fill: 'var(--surface-2)' }} />
          {series.length > 1 && <Legend iconType="circle" iconSize={8} itemSorter={null} wrapperStyle={{ paddingTop: 8 }} />}
          {series.map((s, i) => (
            <Bar key={s.key} dataKey={s.key} name={s.name} fill={SERIES[i % SERIES.length]} radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function Lines({ data, xKey, series, fmt, height = 240, domain, xLabel }: {
  data: Record<string, number | string | null | undefined>[]
  xKey: string
  series: Series[]
  fmt: Fmt
  height?: number
  domain?: [number | 'auto', number | 'auto']
  xLabel?: string
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: xLabel ? 14 : 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey={xKey} tickLine={false} axisLine={{ stroke: 'var(--axis)' }}
            label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -8, fill: 'var(--muted)', fontSize: 12 } : undefined} />
          <YAxis domain={domain ?? ['auto', 'auto']} tickFormatter={(v: number) => fmt(v)} tickLine={false} axisLine={false} width={48} />
          <Tooltip content={(p) => <ChartTooltip {...(p as TooltipContentProps<number, string>)} fmt={fmt} />}
            labelFormatter={(l) => `${xLabel ?? xKey} ${l}`} cursor={{ stroke: 'var(--axis)' }} />
          {series.length > 1 && <Legend iconType="plainline" itemSorter={null} wrapperStyle={{ paddingTop: 8 }} />}
          {series.map((s, i) => (
            <Line key={s.key} dataKey={s.key} name={s.name} stroke={SERIES[i % SERIES.length]} strokeWidth={2} dot={false}
              activeDot={{ r: 4, stroke: 'var(--surface)', strokeWidth: 2 }} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Single-series horizontal bars in HTML: long labels wrap instead of being clipped; value sits at the bar tip. */
export function BarList({ items, fmt, max, highlight, color = 'var(--bar)' }: {
  items: { label: string; value: number | null | undefined; hint?: string }[]
  fmt: Fmt
  max?: number
  highlight?: (label: string) => boolean
  color?: string
}) {
  const top = max ?? Math.max(1e-9, ...items.map((i) => i.value ?? 0))
  return (
    <ul className="space-y-2">
      {items.map((it) => {
        const v = it.value ?? 0
        const strong = highlight?.(it.label)
        return (
          <li key={it.label} title={`${it.label}: ${it.value == null ? '—' : fmt(v)}`} className="grid grid-cols-[minmax(0,14rem)_1fr] items-center gap-3 sm:grid-cols-[minmax(0,20rem)_1fr]">
            <span className={`text-[13px] leading-tight ${strong ? 'font-semibold text-ink' : 'text-ink-2'}`}>
              {it.label}
              {it.hint && <span className="block text-[11px] text-muted">{it.hint}</span>}
            </span>
            <span className="flex items-center gap-2 min-w-0">
              <span className="h-3.5 rounded-r-[4px]" style={{ width: `${Math.max(0.5, (v / top) * 100) * 0.82}%`, background: color, opacity: highlight && !strong ? 0.55 : 1 }} />
              <span className="tabular shrink-0 text-xs font-medium text-ink">{it.value == null ? '—' : fmt(v)}</span>
            </span>
          </li>
        )
      })}
    </ul>
  )
}
