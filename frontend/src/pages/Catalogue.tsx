import { useMemo, useState } from 'react'
import type { ClassMeta } from '../lib/api'
import { bodyType } from '../lib/classes'
import { BarList, GroupedBars } from '../components/charts'
import { IconCheck, IconSearch, IconXCircle } from '../components/icons'
import { Card, PageHeader, StatTile } from '../components/ui'

export default function Catalogue({ classes }: { classes: ClassMeta[] }) {
  const [q, setQ] = useState('')
  const [body, setBody] = useState<string | null>(null)

  const enriched = useMemo(() => classes.map((c) => ({ ...c, body: bodyType(c.model) })), [classes])
  const bodies = useMemo(() => Object.entries(enriched.reduce<Record<string, number>>((a, c) => ((a[c.body] = (a[c.body] ?? 0) + 1), a), {})).sort((a, b) => b[1] - a[1]), [enriched])
  const makes = useMemo(() => Object.entries(enriched.reduce<Record<string, number>>((a, c) => ((a[c.make] = (a[c.make] ?? 0) + 1), a), {})).sort((a, b) => b[1] - a[1]), [enriched])
  const years = useMemo(() => {
    const cnt = enriched.reduce<Record<string, number>>((a, c) => ((a[c.year] = (a[c.year] ?? 0) + 1), a), {})
    return Object.keys(cnt).sort().map((y) => ({ category: y, classes: cnt[y] }))
  }, [enriched])

  const terms = q.toLowerCase().split(/\s+/).filter(Boolean)
  const matches = enriched.filter((c) => (!body || c.body === body) && terms.every((t) => c.name.toLowerCase().includes(t)))
  const grouped = Object.entries(matches.reduce<Record<string, typeof matches>>((a, c) => ((a[c.make] ??= []).push(c), a), {})).sort((a, b) => a[0].localeCompare(b[0]))
  const yearSpan = years.length ? `${years[0].category}–${years[years.length - 1].category}` : '—'

  if (!classes.length) return <div className="h-64 animate-pulse rounded-xl bg-surface-2" />

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Reference" title="Coverage catalogue">
        Every vehicle the model can name. Anything outside these 196 make / model / year classes gets mapped to the closest class it knows,
        which is why the confidence handling matters.
      </PageHeader>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Classes" value={classes.length} note="make + model + year" />
        <StatTile label="Makes" value={makes.length} note={`most: ${makes[0]?.[0]} (${makes[0]?.[1]})`} />
        <StatTile label="Body styles" value={bodies.filter(([b]) => b !== 'Other').length} note="sedan, SUV, pickup…" />
        <StatTile label="Model years" value={yearSpan} note={`${years.length} distinct years`} />
      </div>

      <Card>
        <div className="relative">
          <IconSearch className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Can it recognise my car? Try “BMW M3”, “Tesla”, “2012 convertible”…"
            className="w-full rounded-xl border border-line bg-page py-3 pl-10 pr-4 text-[15px] text-ink placeholder:text-muted focus:outline-2 focus:outline-accent" />
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <button onClick={() => setBody(null)} className={`rounded-full border px-3 py-1 text-xs ${!body ? 'border-accent bg-accent-wash/40 text-ink' : 'border-line text-ink-2 hover:text-ink'}`}>All styles</button>
          {bodies.map(([b, n]) => (
            <button key={b} onClick={() => setBody(body === b ? null : b)} className={`rounded-full border px-3 py-1 text-xs ${body === b ? 'border-accent bg-accent-wash/40 text-ink' : 'border-line text-ink-2 hover:text-ink'}`}>
              {b} <span className="tabular text-muted">{n}</span>
            </button>
          ))}
        </div>
        {(q || body) && (
          <div className={`mt-4 flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${matches.length ? 'bg-good/8 text-good-ink' : 'bg-critical/8 text-critical-ink'}`}>
            {matches.length ? <IconCheck className="h-4 w-4" /> : <IconXCircle className="h-4 w-4" />}
            {matches.length ? `${matches.length} matching class${matches.length === 1 ? '' : 'es'} — the model can recognise ${matches.length === 1 ? 'it' : 'these'}.`
              : 'Not in the 196 classes. The model would return the closest look-alike, most likely with low or uncertain confidence.'}
          </div>
        )}
      </Card>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="columns-1 gap-4 space-y-4 sm:columns-2">
          {grouped.map(([make, list]) => (
            <div key={make} className="break-inside-avoid rounded-2xl border border-line bg-surface p-4 shadow-[var(--shadow-card)]">
              <div className="flex items-baseline justify-between">
                <h3 className="font-semibold text-ink">{make}</h3>
                <span className="text-xs text-muted tabular">{list.length}</span>
              </div>
              <ul className="mt-2 space-y-1">
                {list.map((c) => (
                  <li key={c.id} className="flex items-baseline justify-between gap-2 text-[13px]">
                    <span className="text-ink-2">{c.model}</span>
                    <span className="shrink-0 tabular text-muted">{c.year}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="space-y-5">
          <Card title="Classes per make" subtitle={`Top 12 of ${makes.length} makes`}>
            <BarList items={makes.slice(0, 12).map(([k, v]) => ({ label: k, value: v }))} fmt={(v) => String(v)} />
          </Card>
          <Card title="Body styles" subtitle="Derived from the class names">
            <BarList items={bodies.map(([k, v]) => ({ label: k, value: v }))} fmt={(v) => String(v)} />
          </Card>
          <Card title="Model years" subtitle="Number of classes per model year">
            <GroupedBars data={years} series={[{ key: 'classes', name: 'Classes' }]} fmt={(v) => String(v)} domain={[0, 'auto']} height={220} xInterval="preserveStartEnd" />
          </Card>
        </div>
      </div>
    </div>
  )
}
