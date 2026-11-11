import { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { CarsMetrics, ClassMeta, FailureCase, Row } from '../lib/api'
import { mb, ms, num, pct } from '../lib/format'
import { BarList, GroupedBars, Lines } from '../components/charts'
import { DataTable } from '../components/DataTable'
import { IconArrowRight } from '../components/icons'
import { Card, Label, Missing, PageHeader, StatTile } from '../components/ui'

const STATUS_NAME: Record<string, string> = { confident: 'Confident', uncertain_model: 'Uncertain model / year', low_confidence: 'Low confidence' }
const usd = (v: unknown) => (typeof v === 'number' ? `$${v.toFixed(2)}` : '—')

/** Evaluation of the all-cars model (about 1,300 generations, 1948 to 2026), from artifacts_modern/metrics.json. */
export default function AllCars({ metrics, classes, onTry }: { metrics: CarsMetrics | null; classes: ClassMeta[] | null; onTry: () => void }) {
  const [failures, setFailures] = useState<FailureCase[] | null>(null)
  useEffect(() => {
    api.failures().then(setFailures).catch(() => setFailures([]))
  }, [])

  const recent = useMemo(() => {
    const byId = new Map((classes ?? []).map((c) => [c.id, c]))
    return (metrics?.per_class ?? [])
      .map((p) => ({ ...p, c: byId.get(p.id) }))
      .filter((p) => p.c && (p.c.start_year ?? 0) >= 2023 && p.acc != null && p.n_test >= 3)
      .sort((a, b) => (b.c!.start_year ?? 0) - (a.c!.start_year ?? 0) || a.c!.name.localeCompare(b.c!.name))
  }, [metrics, classes])

  if (!metrics) return <div className="h-64 animate-pulse rounded-xl bg-surface-2" />
  if (!metrics.test) return <Missing what="The all-cars model" hint="Train it with modal_train/train_modern.py and copy the results into artifacts_modern/." />

  const t = metrics.test
  const d = metrics.dataset ?? {}
  const tr = metrics.training ?? {}
  const conf = metrics.confidence
  const hist = tr.history ?? []
  const bench = metrics.gpu_benchmark ?? []
  const breakdown: Record<string, Row> = Object.fromEntries(
    Object.entries(conf?.decision_breakdown ?? {}).map(([k, v]) => [STATUS_NAME[k] ?? k, v]))
  const thr: Record<string, Row> = Object.fromEntries((conf?.threshold_table ?? []).map((r) => [`≥ ${Number(r.threshold).toFixed(1)}`, r]))
  const ab = metrics.ablation ?? {}
  const sameModel = (metrics.top_confusions ?? []).filter((c) => c.same_model).reduce((a, c) => a + c.count, 0)
  const allConf = (metrics.top_confusions ?? []).reduce((a, c) => a + c.count, 0)

  return (
    <div className="space-y-5">
      <PageHeader eyebrow="Evaluation" title="All-cars model"
        actions={<button onClick={onTry} className="inline-flex items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-white hover:opacity-90">Try it on a photo <IconArrowRight className="h-4 w-4" /></button>}>
        A second classifier that names {d.classes?.toLocaleString() ?? 'about 1,300'} car generations from {d.years?.first ?? 1948} to 2026,
        including 2023 to 2026 models. It runs inside the same pipeline as the Stanford Cars model: YOLO detection, crop, illumination fix,
        ConvNeXt-Tiny with flip TTA, temperature calibration and the three-level decision rule. All numbers below are on test photos the model never saw.
      </PageHeader>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatTile label="Generation top-1" value={pct(t.top1)} note={`${t.n.toLocaleString()} test photos`} />
        <StatTile label="Top-5" value={pct(t.top5)} note="right answer in the top five" />
        <StatTile label="Make accuracy" value={pct(t.make_acc)} note={`${d.makes ?? '—'} makes`} />
        <StatTile label="Model, any generation" value={pct(t.model_acc_any_generation)} note={`${d.models ?? '—'} models`} />
        <StatTile label="Macro F1" value={num(t.macro_f1, 3)} note="every class counts the same" />
        <StatTile label="Training cost" value={usd(tr.cost_usd_total_estimate)} note={`${tr.gpu ?? ''}, ${tr.total_minutes ?? '—'} min`} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Accuracy by era" subtitle="Grouped by the year each generation was launched">
          {metrics.by_era?.length ? (
            <>
              <GroupedBars data={metrics.by_era.filter((e) => e.era !== '2023 and newer').map((e) => ({ category: e.era, top1: e.top1, make: e.make_acc }))}
                series={[{ key: 'top1', name: 'Generation top-1' }, { key: 'make', name: 'Make' }]} fmt={(v) => pct(v)} height={250} />
              <p className="mt-3 text-[13px] text-ink-2">
                {metrics.by_era.filter((e) => e.era.startsWith('20')).map((e) => `${e.era}: ${pct(e.top1)} on ${e.n.toLocaleString()} photos`).join(' · ')}
              </p>
            </>
          ) : <Missing what="Era breakdown" />}
        </Card>

        <Card title="Training" subtitle={`${tr.epochs_run ?? '—'} epochs on an ${tr.device ?? tr.gpu ?? 'GPU'} in ${tr.train_minutes ?? '—'} minutes`}>
          {hist.length > 1 ? (
            <Lines data={hist.map((h) => ({ epoch: h.epoch, raw: h.val_top1_raw, ema: h.val_top1_ema }))} xKey="epoch" xLabel="epoch"
              series={[{ key: 'raw', name: 'Validation top-1' }, { key: 'ema', name: 'Validation top-1 (EMA weights)' }]} fmt={(v) => pct(v, 0)} height={230} />
          ) : <Missing what="Training curve" />}
          <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-[13px] sm:grid-cols-3">
            <Fact k="Backbone" v="ConvNeXt-Tiny (ImageNet-22k)" />
            <Fact k="Input" v={`${tr.img_size ?? 288} px, ${tr.precision ?? 'bf16'}`} />
            <Fact k="Batch / LR" v={`${tr.batch ?? '—'} / ${tr.lr ?? '—'}`} />
            <Fact k="Kept weights" v={ab.best_weights === 'ema' ? 'EMA average' : 'last good epoch'} />
            <Fact k="Throughput" v={hist.length ? `${Math.round(hist[hist.length - 1].img_per_s).toLocaleString()} img/s` : '—'} />
            <Fact k="Stopped early" v={tr.stopped_early ?? 'no'} />
          </dl>
        </Card>
      </div>

      {bench.length > 0 && (
        <Card title="Choosing the GPU" subtitle="Same model, batch and precision on every GPU, including the 8 CPU cores and 24 GB RAM the run also pays for">
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div>
              <Label>Cost per million training images (lower is better)</Label>
              <BarList items={bench.map((b) => ({ label: b.gpu, value: b.usd_per_1M_train_images, hint: `${Math.round(b.img_per_s).toLocaleString()} img/s · $${b.usd_per_hour_with_8cpu_24gb.toFixed(2)}/h` }))}
                fmt={(v) => `$${v.toFixed(2)}`} highlight={(l) => l === tr.gpu} />
            </div>
            <p className="self-center text-[14px] leading-[1.6] text-ink-2">
              The {bench[0].gpu} costs the most per hour but finishes the job fastest, so it is the cheapest per image:
              the CPU and memory are billed by the hour too, and a fast GPU needs fewer of those hours.
              The whole run, including this benchmark, the data preparation and a smoke test, cost about {usd(tr.cost_usd_total_estimate)} of the $10 budget.
            </p>
          </div>
        </Card>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Confidence and decisions" subtitle={conf ? `τ = ${num(conf.tau)} for a confident answer (99% precise on validation), τ_make = ${num(conf.tau_make)} for a make-only answer (98%)` : undefined}>
          {conf ? (
            <div className="space-y-5">
              <DataTable data={breakdown} firstCol="Decision" columns={[
                { key: 'share', label: 'Share of test photos', format: (v) => pct(v) },
                { key: 'model_acc', label: 'Generation right', format: (v) => pct(v) },
                { key: 'make_acc', label: 'Make right', format: (v) => pct(v) },
              ]} />
              <DataTable data={thr} firstCol="Confidence" columns={[
                { key: 'coverage', label: 'Photos kept', format: (v) => pct(v) },
                { key: 'accuracy', label: 'Accuracy of kept', format: (v) => pct(v) },
              ]} />
            </div>
          ) : <Missing what="Thresholds" />}
        </Card>

        <div className="space-y-5">
          <Card title="Calibration" subtitle="Temperature scaling fitted on the validation split">
            {metrics.calibration ? (
              <div className="grid grid-cols-3 gap-3">
                <StatTile label="Temperature T" value={num(metrics.calibration.T, 3)} />
                <StatTile label="ECE before" value={pct(metrics.calibration.ece_before)} />
                <StatTile label="ECE after" value={pct(metrics.calibration.ece_after)} />
              </div>
            ) : <Missing what="Calibration" />}
          </Card>
          <Card title="What each step adds" subtitle="Test top-1 with one step switched off">
            <BarList items={[
              { label: 'Full pipeline', value: ab.full_pipeline },
              { label: 'Without flip TTA', value: ab.without_flip_tta },
              { label: 'Without the illumination fix', value: ab.without_illumination_fix },
            ]} fmt={(v) => pct(v, 2)} highlight={(l) => l === 'Full pipeline'} max={1} />
          </Card>
          <Card title="Running on a CPU" subtitle="One car crop, 2 CPU threads (like the hosted app)">
            {metrics.optimisation?.length ? (
              <DataTable data={Object.fromEntries(metrics.optimisation.map((o) => [o.variant, o as unknown as Row]))} firstCol="Variant" columns={[
                { key: 'size_mb', label: 'Size', format: (v) => mb(v) },
                { key: 'ms', label: 'Latency', format: (v) => ms(v) },
                { key: 'top1', label: 'Top-1', format: (v) => pct(v) },
                { key: 'agreement_with_fp32', label: '= FP32', format: (v) => (v == null ? '' : pct(v)) },
              ]} />
            ) : <Missing what="Optimisation results" />}
          </Card>
        </div>
      </div>

      {recent.length > 0 && (
        <Card title="Newest generations (launched 2023 or later)" subtitle={`Test accuracy per generation, ${recent.length} generations`}>
          <div className="columns-1 gap-6 sm:columns-2 lg:columns-3">
            {recent.map((p) => (
              <div key={p.id} className="flex break-inside-avoid items-baseline justify-between gap-3 border-b border-grid py-1.5 text-[13px]">
                <span className="text-ink">{p.c!.make} {p.c!.model} <span className="text-muted">{p.c!.year}</span></span>
                <span className="tabular shrink-0 text-ink-2">{pct(p.acc, 0)} <span className="text-muted">of {p.n_test}</span></span>
              </div>
            ))}
          </div>
        </Card>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Most confused pairs" subtitle={allConf ? `${pct(sameModel / allConf, 0)} of these mistakes are between generations of the same model` : undefined}>
          {metrics.top_confusions?.length ? (
            <ul className="space-y-1.5 text-[13px]">
              {metrics.top_confusions.slice(0, 12).map((c, i) => (
                <li key={i} className="grid grid-cols-[1fr_auto] gap-3 border-b border-grid pb-1.5 last:border-0">
                  <span className="text-ink-2"><span className="text-ink">{c.true}</span> taken for {c.pred}{c.same_model && <span className="ml-1.5 rounded bg-surface-2 px-1.5 py-0.5 text-[11px] text-muted">same model</span>}</span>
                  <span className="tabular text-muted">{c.count}×</span>
                </li>
              ))}
            </ul>
          ) : <Missing what="Confusions" />}
        </Card>

        <Card title="Data" subtitle={d.name}>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-[13px]">
            <Fact k="Photos used" v={d.images_used?.toLocaleString() ?? '—'} />
            <Fact k="Classes" v={`${d.classes?.toLocaleString() ?? '—'} generations`} />
            <Fact k="Train / val / test" v={d.split ? `${d.split.train.toLocaleString()} / ${d.split.val.toLocaleString()} / ${d.split.test.toLocaleString()}` : '—'} />
            <Fact k="Minimum per class" v={`${d.min_photos_per_class ?? '—'} clean photos`} />
          </dl>
          <p className="mt-3 text-[13px] leading-[1.6] text-ink-2">
            Photos come from Wikimedia Commons through the <a className="underline" href={`https://huggingface.co/datasets/${d.hf_repo}`} target="_blank" rel="noreferrer">{d.hf_repo}</a> dataset
            (each photo keeps its own Creative Commons licence), plus Stanford Cars training photos for 14 makes the Commons set does not cover
            (Ferrari, Lamborghini, Bentley, Rolls-Royce, Aston Martin, McLaren, Bugatti and others). The Stanford test set was not used, so the benchmark numbers stay fair.
            Split rule: {d.split_rule ?? '80/10/10 per class'}. All photos of the same car by the same person on the same day stay together, so near-copies cannot leak from training into the test set.
          </p>
          <p className="mt-2 text-[13px] leading-[1.6] text-muted">
            Limits: mostly US-market models, so India-only cars (Maruti Suzuki, Tata, Mahindra) and other local models are not in the list. In our check with 8 such
            photos the model marked every one as low confidence or uncertain instead of giving a confident wrong answer; the optional LLM second opinion can then name them.
          </p>
        </Card>
      </div>

      <Card title="Most confident mistakes" subtitle="Test photos the model got wrong while sure of itself. Photo credits under each image.">
        {failures === null ? <div className="h-40 animate-pulse rounded-xl bg-surface-2" /> : failures.length ? (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            {failures.slice(0, 12).map((f) => (
              <figure key={f.file} className="min-w-0">
                <img src={api.failureUrl(f)} alt={`${f.true} predicted as ${f.pred}`} loading="lazy" className="aspect-[4/3] w-full rounded-lg border border-line object-cover" />
                <figcaption className="mt-1.5 space-y-0.5 text-[12px] leading-snug">
                  <div className="text-ink">{f.true}</div>
                  <div className="text-critical-ink">said {f.pred} ({pct(f.confidence, 0)})</div>
                  <div className="truncate text-muted" title={`${f.author ?? ''}, ${f.licence ?? ''}`}>
                    {f.source ? <a href={f.source} target="_blank" rel="noreferrer" className="hover:underline">{f.author}</a> : f.author}, {f.licence}
                  </div>
                </figcaption>
              </figure>
            ))}
          </div>
        ) : <Missing what="Failure examples" />}
      </Card>
    </div>
  )
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3 border-b border-grid py-1 last:border-0 sm:block sm:border-0 sm:py-0.5">
      <dt className="text-muted">{k}</dt>
      <dd className="text-ink">{v}</dd>
    </div>
  )
}
