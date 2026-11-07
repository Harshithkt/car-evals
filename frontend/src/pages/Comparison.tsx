import type { Metrics, Row } from '../lib/api'
import { mb, ms, num, pct } from '../lib/format'
import { BarList, GroupedBars, Lines } from '../components/charts'
import { DataTable } from '../components/DataTable'
import { Plot } from '../components/Plot'
import { Card, Missing, PageHeader, StatTile } from '../components/ui'

const n = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null)

const shortApproach = (k: string) =>
  k.startsWith('Classical') ? 'Classical' : k.startsWith('Deep') ? 'Deep learning' : k.startsWith('Hybrid') ? 'Hybrid' : k

const shortDetector: Record<string, string> = {
  classical_contour_roi: 'Classical contour ROI',
  yolov8s_coco_pretrained: 'YOLOv8s (COCO, zero-shot)',
  yolov8s_finetuned: 'YOLOv8s (fine-tuned)',
}

/** Pivot `{entity: {metric: v}}` into chart rows `[{category: metricLabel, s0: v, s1: v…}]`. */
function pivot(obj: Record<string, Row>, metrics: [string, string][]) {
  const keys = Object.keys(obj)
  return {
    series: keys.map((k, i) => ({ key: `s${i}`, entity: k })),
    rows: metrics.map(([m, label]) => {
      const r: Record<string, string | number | null> = { category: label }
      keys.forEach((k, i) => (r[`s${i}`] = n(obj[k]?.[m])))
      return r
    }),
  }
}

export default function Comparison({ metrics, plots }: { metrics: Metrics | null; plots: string[] | null }) {
  if (!metrics) return <div className="h-64 animate-pulse rounded-xl bg-surface-2" />
  const ac = metrics.approach_comparison
  const det = metrics.detection
  const hist = metrics.classifier_training?.history

  return (
    <div className="space-y-5">
      <PageHeader eyebrow="Evaluation" title="Model comparison">
        Classical features vs deep learning vs the hybrid system, evaluated on the unseen Stanford Cars test set (8,041 images).
      </PageHeader>

      {/* ---------- approach comparison ---------- */}
      {ac ? (() => {
        const { series, rows } = pivot(ac, [['top1', 'Top-1'], ['top5', 'Top-5'], ['make_acc', 'Make'], ['make_model_acc', 'Make + model']])
        const names = Object.keys(ac)
        const best = names.reduce((a, b) => ((n(ac[b]?.top1) ?? 0) > (n(ac[a]?.top1) ?? 0) ? b : a), names[0])
        return (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <StatTile label="Best top-1 accuracy" value={pct(ac[best]?.top1)} note={shortApproach(best)} />
              <StatTile label="Best make accuracy" value={pct(ac[best]?.make_acc)} note={shortApproach(best)} />
              <StatTile label="Classical top-1" value={pct(names.map((k) => ac[k]).find((_, i) => names[i].startsWith('Classical'))?.top1)} note="HOG + colour + texture + geometry" />
              <StatTile label="Classifier best val top-1" value={pct(metrics.classifier_training?.best_val_top1)} note="ConvNeXt-Tiny, validation split" />
            </div>
            <Card title="Classical vs deep vs hybrid" subtitle="Accuracy on the unseen test set; size and latency per image">
              <GroupedBars data={rows} series={series.map((s) => ({ key: s.key, name: shortApproach(s.entity) }))} fmt={(v) => pct(v)} />
              <div className="mt-5 grid gap-6 md:grid-cols-2">
                <div>
                  <div className="mb-2 text-xs font-medium text-ink-2">Model size</div>
                  <BarList items={names.map((k) => ({ label: shortApproach(k), value: n(ac[k]?.size_mb) }))} fmt={mb} />
                </div>
                <div>
                  <div className="mb-2 text-xs font-medium text-ink-2">Latency per image (GPU, from the notebook)</div>
                  <BarList items={names.map((k) => ({ label: shortApproach(k), value: n(ac[k]?.ms_per_image) }))} fmt={ms} />
                </div>
              </div>
              <div className="mt-5">
                <DataTable
                  data={ac}
                  firstCol="Approach"
                  highlight={(k) => k === best}
                  columns={[
                    { key: 'top1', label: 'Top-1' }, { key: 'top5', label: 'Top-5' }, { key: 'make_acc', label: 'Make' },
                    { key: 'make_model_acc', label: 'Make + model' }, { key: 'size_mb', label: 'Size' }, { key: 'ms_per_image', label: 'ms / image' },
                  ]}
                />
              </div>
            </Card>
          </>
        )
      })() : <Missing what="approach_comparison" hint="Run notebook section 8.3." />}

      {/* ---------- ablation ---------- */}
      <Card title="Ablation" subtitle="Each row adds or removes one component of the hybrid pipeline. Top-1 on the unseen test set.">
        {metrics.ablation ? (
          <div className="grid gap-6 lg:grid-cols-2">
            <BarList
              items={Object.entries(metrics.ablation).map(([k, r]) => ({ label: k, value: n(r.top1) }))}
              fmt={(v) => pct(v)}
              max={1}
              highlight={(l) => l.startsWith('HYBRID')}
            />
            <DataTable data={metrics.ablation} firstCol="Configuration" highlight={(k) => k.startsWith('HYBRID')}
              columns={[{ key: 'top1', label: 'Top-1' }, { key: 'top5', label: 'Top-5' }, { key: 'make_acc', label: 'Make' }, { key: 'make_model_acc', label: 'Make + model' }]} />
          </div>
        ) : <Missing what="ablation" hint="Run notebook section 8.2." />}
      </Card>

      {/* ---------- detection ---------- */}
      <Card title="Vehicle detection / localisation" subtitle="Three localisers on the unseen test set">
        {det ? (() => {
          const { series, rows } = pivot(det, [['mAP50', 'mAP@0.5'], ['mAP50_95', 'mAP@0.5:0.95'], ['mean_IoU', 'Mean IoU'], ['loc_acc_IoU50', 'Loc. acc (IoU≥0.5)']])
          return (
            <>
              <GroupedBars data={rows} series={series.map((s) => ({ key: s.key, name: shortDetector[s.entity] ?? s.entity }))} fmt={(v) => pct(v)} />
              <div className="mt-5">
                <DataTable
                  data={Object.fromEntries(Object.entries(det).map(([k, v]) => [shortDetector[k] ?? k, v]))}
                  firstCol="Detector"
                  highlight={(k) => k.includes('fine-tuned')}
                  columns={[
                    { key: 'mAP50', label: 'mAP50', format: (v) => num(v, 3) }, { key: 'mAP50_95', label: 'mAP50-95', format: (v) => num(v, 3) },
                    { key: 'mean_IoU', label: 'Mean IoU', format: (v) => num(v, 3) }, { key: 'loc_acc_IoU50', label: 'Loc acc' },
                    { key: 'detection_rate', label: 'Detection rate' }, { key: 'ms_per_image', label: 'ms / image' },
                  ]}
                />
              </div>
            </>
          )
        })() : <Missing what="detection" hint="Run notebook section 5." />}
        {metrics.detection_ultralytics_test && (
          <div className="mt-5">
            <div className="mb-2 text-xs font-medium text-ink-2">Fine-tuned YOLOv8s: Ultralytics validator on the test split</div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {Object.entries(metrics.detection_ultralytics_test).map(([k, v]) => (
                <StatTile key={k} label={k.startsWith('mAP') ? k.replace('_', '-') : k.replace(/^./, (c) => c.toUpperCase())} value={num(v, 3)} />
              ))}
            </div>
          </div>
        )}
      </Card>

      {/* ---------- training ---------- */}
      <Card title="Classifier training" subtitle="ConvNeXt-Tiny fine-tuned for 20 epochs; the best epoch by validation top-1 was kept">
        {hist?.length ? (
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <div className="mb-1 text-xs font-medium text-ink-2">Validation accuracy</div>
              <Lines data={hist} xKey="epoch" xLabel="Epoch" fmt={(v) => pct(v, 0)}
                series={[{ key: 'val_top1', name: 'Top-1' }, { key: 'val_top5', name: 'Top-5' }]} domain={[0, 1]} />
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-ink-2">Training loss (label-smoothed cross-entropy)</div>
              <Lines data={hist} xKey="epoch" xLabel="Epoch" fmt={(v) => num(v, 2)} series={[{ key: 'train_loss', name: 'Train loss' }]} />
            </div>
          </div>
        ) : <Missing what="classifier_training.history" />}
      </Card>

      {/* ---------- optimisation ---------- */}
      <Card title="Model optimisation" subtitle="FP16, ONNX, INT8 dynamic quantisation and pruning: size, batch-1 latency and top-1 on 1,000 test crops">
        {metrics.optimization ? (
          <>
            <OptimisationHighlights opt={metrics.optimization} />
            <DataTable data={metrics.optimization} firstCol="Variant"
              columns={[
                ...(Object.values(metrics.optimization).some((r) => r.technique) ? [{ key: 'technique', label: 'Technique', align: 'left' as const, format: (v: unknown) => String(v ?? '') }] : []),
                { key: 'size_mb', label: 'Size' }, { key: 'ms', label: 'Latency (batch 1)' }, { key: 'top1', label: 'Top-1' }, { key: 'sparsity', label: 'Sparsity' },
              ]} />
          </>
        ) : <Missing what="optimization" hint="Run section D of P5_Evaluation_Remaining.ipynb (it also exports classifier_fp16.pth and the ONNX models)." />}
      </Card>
      <Plot name="optimization" title="Optimisation (notebook plot)" available={plots} />

      {metrics.calibration_params && (
        <Card title="Calibration and fusion parameters" subtitle="Fitted on the validation split">
          <div className="grid grid-cols-3 gap-3">
            <StatTile label="T (deep temperature)" value={num(metrics.calibration_params.T, 3)} />
            <StatTile label="α (classical fusion weight)" value={num(metrics.calibration_params.alpha, 2)} />
            <StatTile label="T₂ (hybrid temperature)" value={num(metrics.calibration_params.T2, 3)} />
          </div>
        </Card>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Plot name="approach_comparison" title="Approach comparison (notebook plot)" available={plots} />
        <Plot name="training_curves" title="Training curves (notebook plot)" available={plots} />
      </div>
      <Plot name="detection_examples" title="Detection examples" caption="Green = ground truth, blue = classical, yellow = COCO YOLO, red = fine-tuned YOLO" available={plots} />
      <Plot name="fusion_alpha" title="Fusion weight search" caption="Validation top-1 vs α (classical weight)" available={plots} />
    </div>
  )
}

/** Size reduction and accuracy cost of INT8 vs the FP32 ONNX baseline. */
function OptimisationHighlights({ opt }: { opt: Record<string, Row> }) {
  const f32 = opt['ONNX Runtime FP32 (CPU)']
  const i8 = opt['ONNX Runtime INT8 (CPU)'] ?? Object.entries(opt).find(([k]) => /INT8/i.test(k))?.[1]
  const fp16 = Object.entries(opt).find(([k]) => /FP16/.test(k) && !/YOLO/i.test(k))?.[1]
  const base = Object.entries(opt).find(([k]) => /FP32 \(GPU\)/.test(k))?.[1] ?? f32
  if (!i8 || !base) return null
  const size = (n(i8.size_mb) ?? 0) / Math.max(1e-9, n(base.size_mb) ?? 1)
  return (
    <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
      <StatTile label="INT8 model size" value={mb(i8.size_mb)} note={`${(1 / size).toFixed(1)}× smaller than FP32 (${mb(base.size_mb)})`} />
      <StatTile label="INT8 accuracy change" value={`${((((n(i8.top1) ?? 0) - (n(base.top1) ?? 0)) * 100) >= 0 ? '+' : '−')}${Math.abs(((n(i8.top1) ?? 0) - (n(base.top1) ?? 0)) * 100).toFixed(2)} pts`} note={`top-1 ${pct(i8.top1)} vs ${pct(base.top1)}`} />
      {fp16 && <StatTile label="FP16 latency" value={ms(fp16.ms)} note={`vs FP32 ${ms(base.ms)} · top-1 ${pct(fp16.top1)}`} />}
    </div>
  )
}
