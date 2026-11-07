import type { Metrics, Row, Status } from '../lib/api'
import { humanize, num, pct } from '../lib/format'
import { conditionLabel, signedPts, thresholdRows } from '../lib/metrics'
import { GroupedBars, Lines } from '../components/charts'
import { DataTable } from '../components/DataTable'
import { Plot } from '../components/Plot'
import { Card, Missing, PageHeader, StatTile, StatusBadge } from '../components/ui'

const n = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null)

const ROB_SERIES = [
  { key: 'deep_top1', name: 'Deep top-1' },
  { key: 'hybrid_top1', name: 'Hybrid top-1' },
  { key: 'hybrid_enh_top1', name: 'Hybrid + illum. norm. top-1' },
  { key: 'loc_acc', name: 'Localisation acc.' },
]

const COL_LABELS: Record<string, string> = {
  mean_IoU: 'Mean IoU', loc_acc: 'Loc. acc (IoU≥0.5)', deep_top1: 'Deep top-1', hybrid_top1: 'Hybrid top-1',
  hybrid_enh_top1: 'Hybrid + illum. top-1', hybrid_enh_top5: 'Hybrid + illum. top-5', hybrid_enh_make_acc: 'Hybrid + illum. make',
  drop_vs_clean: 'Change vs clean', group: 'Challenge',
  top1: 'Top-1', top5: 'Top-5', IoU: 'IoU', iou: 'IoU', n: 'Images',
}
const colLabel = (k: string) => COL_LABELS[k] ?? humanize(k)
const columnsOf = (rows: Record<string, Row>) =>
  Array.from(new Set(Object.values(rows).flatMap((r) => Object.keys(r)))).map((key) => ({
    key,
    label: colLabel(key),
    ...(key === 'drop_vs_clean' ? { format: (v: unknown) => <span className={typeof v === 'number' && v < -0.005 ? 'text-critical-ink' : ''}>{signedPts(v)}</span> } : {}),
    ...(key === 'group' ? { align: 'left' as const, format: (v: unknown) => <span className="capitalize">{String(v ?? '')}</span> } : {}),
  }))

export default function Robustness({ metrics, plots }: { metrics: Metrics | null; plots: string[] | null }) {
  if (!metrics) return <div className="h-64 animate-pulse rounded-xl bg-surface-2" />
  const rob = metrics.robustness
  const conf = metrics.confidence
  const breakdown = conf?.decision_breakdown

  return (
    <div className="space-y-5">
      <PageHeader eyebrow="Evaluation" title="Robustness & confidence">
        End-to-end behaviour under illumination, occlusion, resolution and viewpoint changes, plus how calibrated confidence drives
        the confident / uncertain / low-confidence decision.
      </PageHeader>

      {/* ---------- confidence handling ---------- */}
      <Card title="Low-confidence handling" subtitle={`Thresholds chosen on validation for ≥${Math.round((conf?.target_precision ?? 0.95) * 100)}% accuracy on accepted predictions, applied to the test set`}>
        {conf ? (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <StatTile label="τ (class threshold)" value={num(conf.tau, 2)} note="p(top-1) ≥ τ → confident" />
              <StatTile label="τ_make (make threshold)" value={num(conf.tau_make, 2)} note="p(make) ≥ τ_make → uncertain model" />
              {breakdown?.confident && <StatTile label="Confident share" value={pct(breakdown.confident.share)} note={`${pct(breakdown.confident.model_acc)} accurate`} />}
              {metrics.end_to_end_ms_per_image != null && (
                <StatTile label="End-to-end latency" value={`${num(metrics.end_to_end_ms_per_image, 0)} ms`}
                  note={metrics.end_to_end_ms_onnx_int8 != null ? `GPU, incl. stages · ONNX INT8 CPU ${num(metrics.end_to_end_ms_onnx_int8, 0)} ms` : 'per image, notebook GPU'} />
              )}
            </div>
            {conf.tau_make === 0 && (
              <p className="mt-3 text-xs text-muted">
                With τ_make = 0, every prediction below τ is reported as “uncertain model” with its most likely make, so the low-confidence status does not occur.
              </p>
            )}
            {breakdown && (
              <div className="mt-4">
                <DataTable
                  data={Object.fromEntries(Object.entries(breakdown).sort(([a], [b]) => ['confident', 'uncertain_model', 'low_confidence'].indexOf(a) - ['confident', 'uncertain_model', 'low_confidence'].indexOf(b)))}
                  firstCol="Decision"
                  renderName={(s) => (['confident', 'uncertain_model', 'low_confidence'].includes(s) ? <StatusBadge status={s as Status} /> : humanize(s))}
                  columns={[
                    { key: 'share', label: 'Share of test images' },
                    { key: 'model_acc', label: 'Exact-class accuracy' },
                    { key: 'make_acc', label: 'Make accuracy' },
                  ]}
                />
              </div>
            )}
            {thresholdRows(conf.threshold_table).length > 0 && (
              <div className="mt-5">
                <div className="mb-2 text-xs font-medium text-ink-2">Risk–coverage: accuracy of accepted predictions vs threshold</div>
                <Lines
                  data={thresholdRows(conf.threshold_table).map((r) => ({ ...r }))}
                  xKey="threshold"
                  xLabel="Confidence threshold"
                  series={[{ key: 'acc', name: 'Accuracy of accepted' }, { key: 'coverage', name: 'Coverage' }]}
                  fmt={(v) => pct(v, 0)}
                  domain={[0, 1]}
                  height={260}
                />
              </div>
            )}
          </>
        ) : <Missing what="confidence" hint="Run notebook section 10.2." />}
      </Card>

      {/* ---------- calibration ---------- */}
      <Card title="Calibration (expected calibration error)" subtitle="Lower is better; temperature scaling should reduce it">
        {metrics.ece ? (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
            {Object.entries(metrics.ece).map(([k, v]) => <StatTile key={k} label={humanize(k)} value={pct(v, 2)} />)}
          </div>
        ) : <Missing what="ece" hint="Run notebook section 10.1." />}
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Plot name="reliability" title="Reliability diagrams" available={plots} />
        <Plot name="threshold_analysis" title="Threshold analysis" available={plots} />
      </div>

      {/* ---------- robustness ---------- */}
      <Card title="End-to-end robustness" subtitle="Test images perturbed before detection; detector + classifier re-run on each">
        {rob ? (
          <>
            <RobustnessInsights rob={rob} />
            <GroupedBars
              data={Object.entries(rob).map(([cond, r]) => ({ category: conditionLabel(cond), ...Object.fromEntries(ROB_SERIES.map((s) => [s.key, n(r[s.key])])) }))}
              series={ROB_SERIES.filter((s) => Object.values(rob).some((r) => n(r[s.key]) != null))}
              fmt={(v) => pct(v)}
              height={300}
            />
            <div className="mt-5">
              <DataTable data={rob} firstCol="Condition" columns={columnsOf(rob)} renderName={conditionLabel} />
            </div>
          </>
        ) : <Missing what="robustness" hint="Run notebook section 9.1." />}
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Plot name="robustness_bars" title="Robustness (notebook plot)" available={plots} />
        <Plot name="robustness_examples" title="Perturbation examples" available={plots} />
      </div>

      {/* ---------- slices ---------- */}
      <Card title="Slice analysis" subtitle="Clean test set split by viewpoint proxy, resolution and brightness">
        {metrics.slices && Object.keys(metrics.slices).length ? (
          <div className="grid gap-6 lg:grid-cols-3">
            {Object.entries(metrics.slices).map(([name, buckets]) => <SliceBlock key={name} name={name} buckets={buckets} />)}
          </div>
        ) : <Missing what="slices" hint="Run notebook section 9.3." />}
      </Card>
      <Plot name="slices" title="Slices (notebook plot)" available={plots} />
    </div>
  )
}

/** Headline findings computed from the robustness table. */
function RobustnessInsights({ rob }: { rob: Record<string, Row> }) {
  const rows = Object.entries(rob).filter(([c]) => c !== 'clean')
  const final = (r: Row) => n(r.hybrid_enh_top1) ?? n(r.hybrid_top1) ?? n(r.deep_top1)
  const worst = rows.filter(([, r]) => final(r) != null).sort((a, b) => final(a[1])! - final(b[1])!)[0]
  const gains = rows.map(([c, r]) => [c, (n(r.hybrid_enh_top1) ?? NaN) - (n(r.deep_top1) ?? NaN)] as const).filter(([, g]) => Number.isFinite(g))
  const bestGain = [...gains].sort((a, b) => b[1] - a[1])[0]
  const illum = rows.map(([c, r]) => [c, (n(r.hybrid_enh_top1) ?? NaN) - (n(r.hybrid_top1) ?? NaN)] as const).filter(([, g]) => Number.isFinite(g)).sort((a, b) => b[1] - a[1])[0]
  if (!worst && !bestGain) return null
  return (
    <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
      {worst && <StatTile label="Hardest condition" value={pct(final(worst[1]))} note={`${conditionLabel(worst[0])} · final system top-1`} />}
      {bestGain && <StatTile label="Biggest hybrid gain over deep-only" value={signedPts(bestGain[1])} note={conditionLabel(bestGain[0])} />}
      {illum && <StatTile label="Illumination normalisation helps most" value={signedPts(illum[1])} note={`${conditionLabel(illum[0])} · hybrid vs hybrid + norm.`} />}
    </div>
  )
}

function SliceBlock({ name, buckets }: { name: string; buckets: Record<string, Row> }) {
  const cols = Array.from(new Set(Object.values(buckets).flatMap((r) => Object.keys(r))))
  const chartCols = cols.filter((c) => Object.values(buckets).every((r) => { const v = n(r[c]); return v == null || (v >= 0 && v <= 1) })).slice(0, 2)
  return (
    <div className="min-w-0">
      <div className="mb-1 text-sm font-medium text-ink">{humanize(name)}</div>
      {chartCols.length > 0 && (
        <GroupedBars
          data={Object.entries(buckets).map(([b, r]) => ({ category: b, ...Object.fromEntries(chartCols.map((c) => [c, n(r[c])])) }))}
          series={chartCols.map((c) => ({ key: c, name: colLabel(c) }))}
          fmt={(v) => pct(v)}
          height={220}
        />
      )}
      <DataTable data={buckets} firstCol="Bucket" columns={columnsOf(buckets)} />
    </div>
  )
}
