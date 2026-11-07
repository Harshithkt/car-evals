import { useState } from 'react'
import type { Metrics } from '../lib/api'
import { humanize, pct } from '../lib/format'
import { BarList } from '../components/charts'
import { Plot } from '../components/Plot'
import { Card, Label, Missing, PageHeader, StatTile } from '../components/ui'

const CAUSE_HINT: Record<string, string> = {
  localisation: 'Detector box IoU < 0.5',
  low_resolution: 'Vehicle < 100 px on its short side',
  low_light: 'Crop brightness < 70/255',
  'truncation/occlusion': 'Vehicle cut off by the image border',
  year_confusion: 'Same make and model, wrong year',
  same_make: 'Right make, wrong model',
  cross_make_lookalike: 'Different make, similar silhouette',
}

export default function Failures({ metrics, plots }: { metrics: Metrics | null; plots: string[] | null }) {
  const [filter, setFilter] = useState<string | null>(null)
  if (!metrics) return <div className="h-64 animate-pulse rounded-xl bg-surface-2" />
  const causes = metrics.failure_causes
  const examples = metrics.failure_examples ?? []
  const total = causes ? Object.values(causes).reduce((a, b) => a + b, 0) : 0
  const shown = filter ? examples.filter((e) => e.cause === filter) : examples

  return (
    <div className="space-y-5">
      <PageHeader eyebrow="Evaluation" title="Failure analysis">
        Wrong predictions of the final hybrid system on the test set, automatically tagged with their most likely cause.
      </PageHeader>

      <SimilarClasses metrics={metrics} plots={plots} />

      <Card title="Failure causes" subtitle={causes ? `${total.toLocaleString()} errors` : undefined}>
        {causes ? (
          <BarList
            items={Object.entries(causes)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => ({ label: humanize(k), value: v, hint: CAUSE_HINT[k] }))}
            fmt={(v) => `${v.toLocaleString()} (${pct(v / total, 0)})`}
          />
        ) : <Missing what="failure_causes" hint="Run notebook section 12.1." />}
      </Card>

      <Card
        title="Failure examples"
        subtitle="Ground truth → prediction, with the tagged cause"
        actions={
          examples.length > 0 && (
            <select value={filter ?? ''} onChange={(e) => setFilter(e.target.value || null)} className="rounded-md border border-line bg-surface px-2 py-1 text-sm text-ink">
              <option value="">All causes</option>
              {Array.from(new Set(examples.map((e) => e.cause).filter(Boolean))).map((c) => <option key={c} value={c}>{humanize(c!)}</option>)}
            </select>
          )
        }
      >
        {examples.length ? (
          <ul className="divide-y divide-grid">
            {shown.map((e, i) => (
              <li key={i} className="py-3">
                <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-sm">
                  <span className="font-medium text-ink">{e.true ?? '?'}</span>
                  <span className="text-muted" aria-label="predicted as">→</span>
                  <span className="text-critical-ink">{e.pred ?? '?'}</span>
                  {e.conf != null && <span className="tabular text-xs text-muted">({pct(e.conf)})</span>}
                  {e.cause && <span className="ml-auto rounded-full border border-line px-2 py-0.5 text-xs text-ink-2">{humanize(e.cause)}</span>}
                </div>
                {e.explanation && <p className="mt-1 text-[13px] text-ink-2">{e.explanation}</p>}
                {e.file && <p className="mt-0.5 text-[11px] text-muted">{e.file}</p>}
              </li>
            ))}
          </ul>
        ) : <Missing what="failure_examples" hint="Run notebook section 12.2." />}
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        <Plot name="error_hierarchy" title="Error hierarchy (notebook plot)" available={plots} />
        <Plot name="failure_causes" title="Failure causes (notebook plot)" available={plots} />
      </div>
      <Plot name="confused_pair" title="Most-confused pair" caption="Training examples of the true class (top) and the class it was mistaken for (bottom)" available={plots} />
      <Plot name="failure_gallery" title="Failure gallery with Grad-CAM" available={plots} />
    </div>
  )
}

const RELATION_STYLE: Record<string, string> = {
  year: 'border-warning/50 bg-warning/10 text-warning-ink',
  'same make': 'border-line bg-surface-2 text-ink-2',
  'different make': 'border-critical/40 bg-critical/8 text-critical-ink',
}

/** Requirement 1: distinguishing visually similar makes, models and years. */
function SimilarClasses({ metrics, plots: _plots }: { metrics: Metrics; plots: string[] | null }) {
  const sca = metrics.similar_class_analysis
  if (!sca) return (
    <Card title="Visually similar classes"><Missing what="similar_class_analysis" hint="Run section A of P5_Evaluation_Remaining.ipynb." /></Card>
  )
  const h = sca.error_hierarchy ?? {}
  const errors = Object.entries(h).filter(([k]) => k !== 'correct')
  const errTotal = errors.reduce((a, [, v]) => a + v, 0)
  return (
    <Card title="Visually similar classes" subtitle="Where the errors come from: every mistake falls in exactly one bucket">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Correct (exact class)" value={pct(h.correct)} note="hybrid system, unseen test" />
        {errors.map(([k, v]) => (
          <StatTile key={k} label={k.replace(' (same make+model)', '').replace(/^./, (c) => c.toUpperCase())} value={pct(v, 2)}
            note={errTotal ? `${pct(v / errTotal, 0)} of all errors` : undefined} />
        ))}
      </div>
      {sca.year_acc_given_make_model != null && (
        <p className="mt-3 text-[13px] text-ink-2">
          {sca.multi_year_groups} make + model groups exist in more than one year. When the make and model are right, the system picks the correct year
          <b className="text-ink"> {pct(sca.year_acc_given_make_model)}</b> of the time.
        </p>
      )}
      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <Label>Most-confused pairs</Label>
          {sca.top_confused_pairs?.length ? (
            <div className="-mx-1 overflow-x-auto">
              <table className="w-full min-w-[520px] text-sm">
                <thead><tr className="border-b border-grid text-left text-[12px] text-muted">
                  <th className="px-1 py-2 font-medium">True class</th><th className="px-1 py-2 font-medium">Predicted as</th>
                  <th className="px-1 py-2 font-medium">Relation</th><th className="px-1 py-2 text-right font-medium">Count</th>
                </tr></thead>
                <tbody className="tabular">
                  {sca.top_confused_pairs.slice(0, 10).map((p, i) => (
                    <tr key={i} className="border-b border-grid last:border-0">
                      <td className="px-1 py-1.5 text-ink">{p.true}</td>
                      <td className="px-1 py-1.5 text-ink-2">{p.predicted}</td>
                      <td className="px-1 py-1.5"><span className={`rounded-full border px-2 py-0.5 text-[11px] ${RELATION_STYLE[p.relation ?? ''] ?? 'border-line text-ink-2'}`}>{p.relation}</span></td>
                      <td className="px-1 py-1.5 text-right text-ink-2">{p.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="text-sm text-ink-2">No pairs recorded.</p>}
        </div>
        <div className="min-w-0">
          <Label>Hardest classes (lowest test accuracy)</Label>
          {sca.hardest_classes?.length ? (
            <BarList items={sca.hardest_classes.map((c) => ({ label: c.class ?? '?', value: c.accuracy ?? null }))} fmt={(v) => pct(v, 0)} max={1} />
          ) : <p className="text-sm text-ink-2">Not available.</p>}
        </div>
      </div>
    </Card>
  )
}
