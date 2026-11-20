import { useState } from 'react'
import type { Metrics, Row } from '../lib/api'
import { mb, ms, num, pct } from '../lib/format'
import { BarList, GroupedBars, Lines } from '../components/charts'
import { DataTable } from '../components/DataTable'
import { Plot } from '../components/Plot'
import { ShiftLights, Speedometer } from '../components/Speedometer'
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
  const [selectedApproach, setSelectedApproach] = useState<string | null>(null)

  return (
    <div className="space-y-5">
      <PageHeader eyebrow="Evaluation · Performance Telemetry" title="Model comparison">
        Classical features vs deep learning vs the hybrid system, evaluated on the unseen Stanford Cars test set (8,041 images).
      </PageHeader>

      {/* ---------- approach comparison ---------- */}
      {ac ? (() => {
        const { series, rows } = pivot(ac, [['top1', 'Top-1'], ['top5', 'Top-5'], ['make_acc', 'Make'], ['make_model_acc', 'Make + model']])
        const names = Object.keys(ac)
        const best = names.reduce((a, b) => ((n(ac[b]?.top1) ?? 0) > (n(ac[a]?.top1) ?? 0) ? b : a), names[0])
        const activeKey = selectedApproach && ac[selectedApproach] ? selectedApproach : best
        const activeRow = ac[activeKey] ?? {}
        const gearMap: Record<string, { gear: string; mode: string; color: string }> = {
          Classical: { gear: '1ST', mode: 'ECO · CLASSICAL CV', color: '#fab219' },
          'Deep learning': { gear: '5TH', mode: 'SPORT · CONVNEXT-TINY', color: '#21a4ff' },
          Hybrid: { gear: 'S+', mode: 'TRACK · HYBRID FUSION', color: '#00f283' },
        }
        const activeMeta = gearMap[shortApproach(activeKey)] ?? { gear: 'D', mode: 'TELEMETRY', color: '#21a4ff' }

        return (
          <>
            {/* Automotive Instrument Cluster for the Accuracy Table */}
            <Card
              title="Digital Instrument Cluster · Accuracy & Dyno Telemetry"
              subtitle="Select a powertrain (approach) below or click any row in the accuracy table to spin the gauges"
              actions={
                <div className="inline-flex flex-wrap items-center gap-1.5 rounded-xl border border-line bg-[#0d0f13] p-1">
                  {names.map((k) => {
                    const label = shortApproach(k)
                    const isSel = k === activeKey
                    const meta = gearMap[label]
                    return (
                      <button
                        key={k}
                        type="button"
                        onClick={() => setSelectedApproach(k)}
                        className={`inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                          isSel
                            ? 'bg-surface-2 text-white shadow-[0_0_16px_rgba(33,164,255,0.25)] ring-1 ring-accent'
                            : 'text-ink-2 hover:text-ink'
                        }`}
                      >
                        <span
                          className="rounded px-1.5 py-0.5 font-mono text-[10px] font-bold"
                          style={{
                            background: isSel ? meta?.color ?? '#21a4ff' : '#1e2229',
                            color: isSel ? '#081014' : '#9a9ca2',
                          }}
                        >
                          {meta?.gear ?? 'D'}
                        </span>
                        {label}
                      </button>
                    )
                  })}
                </div>
              }
            >
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-white/10 bg-[#0d1014] px-4 py-2.5 font-mono text-xs">
                <div className="flex items-center gap-2.5">
                  <span className="h-2 w-2 animate-ping rounded-full" style={{ background: activeMeta.color }} />
                  <span className="font-bold tracking-wider text-ink">{activeMeta.mode}</span>
                  <span className="hidden text-muted sm:inline">· {activeKey}</span>
                </div>
                <div className="flex flex-wrap items-center gap-4 text-[11px] text-ink-2">
                  <span>WEIGHT: <strong className="text-ink">{mb(n(activeRow.size_mb))}</strong></span>
                  <span>0–PREDICTION: <strong className="text-good-ink">{ms(n(activeRow.ms_per_image))}</strong></span>
                  <span>TEST SAMPLES: <strong className="text-ink">8,041</strong></span>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <Speedometer
                  value={n(activeRow.top1)}
                  label="Top-1 Exact Accuracy"
                  sublabel="Make + Model + Year"
                  color={activeMeta.color}
                  gear={activeMeta.gear}
                  badge="RPM · TOP-1"
                />
                <Speedometer
                  value={n(activeRow.make_acc)}
                  label="Make Accuracy"
                  sublabel="Brand identification"
                  color="#00f283"
                  gear={activeMeta.gear}
                  badge="BOOST · MAKE"
                />
                <Speedometer
                  value={n(activeRow.make_model_acc)}
                  label="Make + Model"
                  sublabel="Ignoring facelift year"
                  color="#21a4ff"
                  gear={activeMeta.gear}
                  badge="TORQUE · MODEL"
                />
                <Speedometer
                  value={n(activeRow.top5)}
                  label="Top-5 Recall"
                  sublabel="Correct class in top 5"
                  color="#3df5a0"
                  gear={activeMeta.gear}
                  badge="REDLINE · TOP-5"
                />
              </div>

              <div className="mt-4">
                <ShiftLights
                  value={n(activeRow.top1) ?? 0}
                  segments={20}
                  label={`Tachometer Shift Strip · ${shortApproach(activeKey)}`}
                  rightText={`${pct(activeRow.top1)} TOP-1 · ${pct(activeRow.make_acc)} MAKE · ${ms(activeRow.ms_per_image)}`}
                />
              </div>
            </Card>

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
                  firstCol="Powertrain / Approach"
                  highlight={(k) => k === activeKey}
                  renderName={(k) => (
                    <button
                      type="button"
                      onClick={() => setSelectedApproach(k)}
                      className="inline-flex items-center gap-2 text-left hover:text-accent"
                      title="Click to load into the speedometer cluster above"
                    >
                      <span className="rounded border border-white/15 bg-black/40 px-1.5 py-0.5 font-mono text-[10px] font-bold text-accent">
                        {gearMap[shortApproach(k)]?.gear ?? 'D'}
                      </span>
                      <span>{k}</span>
                    </button>
                  )}
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
      <Card title="Vehicle detection / localisation · Radar & Sensor Lock" subtitle="Three localisers on the unseen test set">
        {det ? (() => {
          const { series, rows } = pivot(det, [['mAP50', 'mAP@0.5'], ['mAP50_95', 'mAP@0.5:0.95'], ['mean_IoU', 'Mean IoU'], ['loc_acc_IoU50', 'Loc. acc (IoU≥0.5)']])
          const ft = det.yolov8s_finetuned
          return (
            <>
              {ft && (
                <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                  <Speedometer value={n(ft.mAP50)} label="mAP@0.5" sublabel="Fine-tuned YOLOv8s" color="#00f283" gear="RADAR" badge="LOCK · mAP50" />
                  <Speedometer value={n(ft.mAP50_95)} label="mAP@0.5:0.95" sublabel="Strict box overlap" color="#21a4ff" gear="LIDAR" badge="PRECISION" />
                  <Speedometer value={n(ft.mean_IoU)} label="Mean IoU" sublabel="Bounding-box overlap" color="#3df5a0" gear="BOX" badge="OVERLAP · IoU" />
                  <Speedometer value={n(ft.loc_acc_IoU50)} label="Localisation Acc." sublabel="Share with IoU ≥ 0.5" color="#00f283" gear="LOCK" badge="TARGET LOCK" />
                </div>
              )}
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
