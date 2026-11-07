import { useRef } from 'react'
import type { ReactNode } from 'react'
import type { Health, Metrics, Row } from '../lib/api'
import { ms, num, pct } from '../lib/format'
import {
  CarMark, IconArrowRight, IconCamera, IconChart, IconCheck, IconCpu, IconEye, IconFlask, IconGrid, IconHelp, IconLayers, IconMerge, IconScan,
  IconSearch, IconShield, IconSliders,
} from '../components/icons'
import { StatusBadge } from '../components/ui'

type Go = (tab: string) => void

/* Real output of the deployed pipeline on an unseen test image (samples/test_01234.jpg, 786×492);
   images in public/landing were saved from the /api/predict response. */
const HERO = {
  label: 'Chevrolet Silverado 1500 Extended Cab 2012',
  make: 'Chevrolet',
  model: 'Silverado 1500 Extended Cab',
  year: '2012',
  confidence: 0.996,
  det: 0.951,
  box: { left: 4.47, top: 21.11, width: 92.48, height: 70.96 }, // % of the image
  top5: [
    ['Chevrolet Silverado 1500 Extended Cab 2012', 0.996],
    ['GMC Canyon Extended Cab 2012', 0.0009],
    ['Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 0.0007],
  ] as [string, number][],
}

const find = (o: Record<string, Row> | undefined, prefix: string) => (o ? Object.entries(o).find(([k]) => k.startsWith(prefix))?.[1] : undefined)

export default function Landing({ metrics, health, go }: { metrics: Metrics | null; health: Health | null; go: Go }) {
  const how = useRef<HTMLElement>(null)
  const results = useRef<HTMLElement>(null)
  const tools = useRef<HTMLElement>(null)
  const ac = metrics?.approach_comparison
  const hybrid = find(ac, 'Hybrid')
  const deep = find(ac, 'Deep')
  const classical = find(ac, 'Classical')
  const det = metrics?.detection
  const tau = metrics?.confidence?.tau
  const tauMake = metrics?.confidence?.tau_make

  return (
    <div className="bg-page">
      {/* ================= HERO ================= */}
      <section className="hero-bg relative overflow-hidden text-white">
        <nav className="relative z-10 mx-auto flex max-w-6xl items-center gap-6 px-4 py-5 sm:px-6">
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/10 ring-1 ring-white/15">
              <CarMark className="h-5 w-5 text-[#6da7ec]" />
            </span>
            <span className="text-[15px] font-semibold tracking-[-0.01em]"><span className="hidden sm:inline">Fine-Grained </span>Vehicle Recognition</span>
            <span className="hidden rounded-md bg-white/10 px-1.5 py-0.5 text-[11px] font-medium text-white/70 sm:inline">P5</span>
          </div>
          <div className="ml-auto hidden items-center gap-6 text-sm text-white/70 md:flex">
            <button onClick={() => how.current?.scrollIntoView()} className="hover:text-white">How it works</button>
            <button onClick={() => tools.current?.scrollIntoView()} className="hover:text-white">Tools</button>
            <button onClick={() => results.current?.scrollIntoView()} className="hover:text-white">Results</button>
            <button onClick={() => go('comparison')} className="hover:text-white">Evaluation</button>
          </div>
          <button onClick={() => go('recognize')} className="ml-auto whitespace-nowrap rounded-lg bg-white px-3.5 py-2 text-sm font-medium text-[#0a0f1a] hover:bg-white/90 md:ml-0">
            Open app
          </button>
        </nav>

        <div className="relative z-10 mx-auto grid max-w-6xl items-center gap-12 px-4 pb-20 pt-10 sm:px-6 md:pb-32 lg:grid-cols-[1.05fr_1fr] lg:pb-36 lg:pt-16">
          <div>
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3 py-1 text-xs text-white/80">
              <span className="h-1.5 w-1.5 rounded-full bg-[#1baf7a]" />
              Stanford Cars · 196 classes · classical CV + deep learning
            </div>
            <h1 className="text-[34px] font-semibold leading-[1.1] tracking-[-0.03em] sm:text-5xl lg:text-[54px]">
              Identify the make, model and year of a car from one photo.
            </h1>
            <p className="mt-5 max-w-xl text-[17px] leading-relaxed text-white/70">
              A hybrid recognition system. Classical image processing, a fine-tuned YOLOv8 detector and a ConvNeXt classifier work together,
              and every prediction comes with a calibrated confidence and a visual explanation.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <button onClick={() => go('recognize')} className="group inline-flex items-center gap-2 rounded-lg bg-[#2a78d6] px-5 py-3 text-[15px] font-medium text-white shadow-lg shadow-[#2a78d6]/25 hover:bg-[#256abf]">
                Try it with a photo
                <IconArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </button>
              <button onClick={() => go('comparison')} className="inline-flex items-center gap-2 rounded-lg border border-white/20 px-5 py-3 text-[15px] font-medium text-white/90 hover:bg-white/5">
                See the evaluation
              </button>
            </div>
            <div className="mt-8 flex flex-wrap gap-x-6 gap-y-2 text-[13px] text-white/55">
              <span className="inline-flex items-center gap-1.5"><IconCheck className="h-4 w-4 text-[#1baf7a]" /> Multiple vehicles per image</span>
              <span className="inline-flex items-center gap-1.5"><IconCheck className="h-4 w-4 text-[#1baf7a]" /> Grad-CAM explanations</span>
              <span className="inline-flex items-center gap-1.5"><IconCheck className="h-4 w-4 text-[#1baf7a]" /> Runs on CPU or GPU</span>
            </div>
          </div>

          <HeroShot />
        </div>
      </section>

      {/* ================= STATS ================= */}
      <section className="relative z-10 mx-auto -mt-10 max-w-6xl px-4 sm:px-6">
        <div className="grid grid-cols-2 overflow-hidden rounded-2xl border border-line bg-surface shadow-[var(--shadow-pop)] md:grid-cols-4">
          <Stat value={pct(hybrid?.top1)} label="Top-1 accuracy" note="exact make + model + year" />
          <Stat value={pct(hybrid?.make_acc)} label="Make accuracy" note="hybrid system, test set" />
          <Stat value={num(det?.yolov8s_finetuned?.mAP50, 3)} label="Detection mAP@0.5" note="fine-tuned YOLOv8s" />
          <Stat value="196" label="Vehicle classes" note="8,041 unseen test images" />
        </div>
      </section>

      {/* ================= HOW IT WORKS ================= */}
      <section ref={how} className="mx-auto max-w-6xl scroll-mt-6 px-4 py-24 sm:px-6">
        <SectionHead eyebrow="How it works" title="From pixels to make, model and year">
          Every image goes through the same pipeline. These are the real intermediate outputs for the pickup truck above.
        </SectionHead>
        <ol className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <Frame n={1} src="/landing/contours.jpg" title="Edges and contours" text="Canny edges, morphological closing and contour analysis give a classical vehicle region." />
          <Frame n={2} src="/landing/roi.jpg" title="Vehicle detection" text="Fine-tuned YOLOv8 localises each car. If it misses, the classical region is used as a fallback." />
          <Frame n={3} src="/landing/lbp.jpg" title="Hand-crafted features" text="HOG shape, LBP/GLCM texture, colour histograms and geometry such as symmetry and Hu moments." />
          <Frame n={4} src="/landing/gradcam.jpg" title="Deep classification" text="ConvNeXt-Tiny with flip TTA scores 196 classes. Grad-CAM shows where it looked." />
          <li className="flex flex-col rounded-2xl border border-line bg-surface p-4 shadow-[var(--shadow-card)]">
            <StepNo n={5} />
            <div className="mt-3 flex flex-1 flex-col justify-center rounded-xl bg-surface-2 p-4">
              <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">Result</div>
              <div className="mt-1 text-[15px] font-semibold leading-snug text-ink">{HERO.make} {HERO.model}</div>
              <div className="text-sm text-ink-2">{HERO.year}</div>
              <div className="mt-3 h-1.5 rounded-full bg-accent-wash/60"><div className="h-1.5 rounded-full bg-good" style={{ width: `${HERO.confidence * 100}%` }} /></div>
              <div className="mt-1.5 text-xs text-ink-2 tabular">{pct(HERO.confidence)} calibrated</div>
            </div>
            <div className="mt-3 text-sm font-medium text-ink">Fusion and decision</div>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-2">Deep and classical scores are fused, calibrated, and checked against confidence thresholds.</p>
          </li>
        </ol>
      </section>

      {/* ================= PILLARS ================= */}
      <section className="border-y border-line bg-surface">
        <div className="mx-auto max-w-6xl px-4 py-24 sm:px-6">
          <SectionHead eyebrow="Technical components" title="Classical vision, deep learning, and a hybrid of both" />
          <div className="mt-12 grid gap-5 lg:grid-cols-3">
            <Pillar icon={<IconScan className="h-5 w-5" />} title="Classical computer vision" items={[
              'Preprocessing: z-score normalisation, adaptive gamma + CLAHE',
              'Canny edge detection and contour analysis',
              'Colour (HSV, LAB) and texture (LBP, GLCM) features',
              'Contour-based region-of-interest extraction',
              'Geometry: aspect ratio, solidity, Hu moments, symmetry',
            ]} />
            <Pillar icon={<IconCpu className="h-5 w-5" />} title="Deep learning" items={[
              'ConvNeXt-Tiny fine-grained classifier',
              'YOLOv8s vehicle detector',
              'Transfer learning from ImageNet-22k and COCO, then fine-tuning',
              'Multi-class classification across 196 make/model/year classes',
            ]} />
            <Pillar icon={<IconMerge className="h-5 w-5" />} title="Hybrid integration" items={[
              'Classical illumination normalisation before the CNN',
              'Late fusion of classical and deep class scores',
              'Detected regions feed the fine-grained classifier',
              'Bounding boxes with labels and calibrated confidence',
              'Low-confidence handling with make-level fallback',
            ]} />
          </div>
        </div>
      </section>

      {/* ================= TOOLS ================= */}
      <section ref={tools} className="mx-auto max-w-6xl scroll-mt-6 px-4 pt-24 sm:px-6">
        <SectionHead eyebrow="Tools" title="More than a single prediction">
          Interactive tools for probing, stress-testing and explaining the model, all running against the same deployed pipeline.
        </SectionHead>
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Tool icon={<IconCamera className="h-5 w-5" />} title="Live camera" tab="recognize" go={go}
            text="Point a webcam at a car (or a photo of one) and watch boxes and labels track it live." />
          <Tool icon={<IconFlask className="h-5 w-5" />} title="Stress lab" tab="stress" go={go}
            text="Add night, fog, blur, CCTV low-res, occlusion or tilt, then see when the prediction breaks and whether the system notices." />
          <Tool icon={<IconGrid className="h-5 w-5" />} title="Batch analysis" tab="batch" go={go}
            text="Drop a whole folder, follow the queue, get a make breakdown and accuracy against ground truth, and export CSV or JSON." />
          <Tool icon={<IconSliders className="h-5 w-5" />} title="Threshold playground" tab="recognize" go={go}
            text="Drag τ and τ_make on any result and watch the confident / uncertain / low-confidence decision change." />
          <Tool icon={<IconHelp className="h-5 w-5" />} title="Why not the runner-up?" tab="recognize" go={go}
            text="Every result explains its closest rival: a year change, a sibling model, or a look-alike from another brand." />
          <Tool icon={<IconSearch className="h-5 w-5" />} title="Coverage catalogue" tab="catalogue" go={go}
            text="Can it recognise my car? Search all 196 classes by make, model, year and body style." />
        </div>
      </section>

      {/* ================= RESULTS ================= */}
      <section ref={results} className="mx-auto max-w-6xl scroll-mt-6 px-4 py-24 sm:px-6">
        <SectionHead eyebrow="Results" title="Evaluated on 8,041 unseen test images">
          Classical features alone fall far short of a fine-tuned network. The hybrid system keeps the deep model's accuracy and adds
          classical preprocessing, fallbacks and interpretability.
        </SectionHead>
        <div className="mt-12 grid gap-5 md:grid-cols-3">
          <Approach name="Classical" detail="HOG + colour + texture + geometry, logistic regression" r={classical} />
          <Approach name="Deep learning" detail="YOLOv8 crop + ConvNeXt-Tiny" r={deep} />
          <Approach name="Hybrid" detail="YOLO + classical preprocessing and fusion + ConvNeXt TTA" r={hybrid} featured />
        </div>
        {det && (
          <div className="mt-5 rounded-2xl border border-line bg-surface p-6 shadow-[var(--shadow-card)]">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <div className="text-[15px] font-semibold text-ink">Vehicle localisation (mAP@0.5:0.95)</div>
              <button onClick={() => go('comparison')} className="inline-flex items-center gap-1 text-sm text-accent hover:underline">
                Full comparison <IconArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
            <div className="mt-4 space-y-3">
              {([['classical_contour_roi', 'Classical contour ROI'], ['yolov8s_coco_pretrained', 'YOLOv8s, COCO pretrained'], ['yolov8s_finetuned', 'YOLOv8s, fine-tuned']] as const).map(([k, label]) => {
                const v = det[k]?.mAP50_95
                return (
                  <div key={k} className="grid grid-cols-[minmax(0,11rem)_1fr_3.5rem] items-center gap-3 text-sm sm:grid-cols-[14rem_1fr_4rem]">
                    <span className={k === 'yolov8s_finetuned' ? 'font-medium text-ink' : 'text-ink-2'}>{label}</span>
                    <span className="h-2 rounded-full bg-surface-2">
                      <span className="block h-2 rounded-full" style={{ width: `${(typeof v === 'number' ? v : 0) * 100}%`, background: 'var(--series-1)', opacity: k === 'yolov8s_finetuned' ? 1 : 0.5 }} />
                    </span>
                    <span className="text-right font-medium tabular text-ink">{num(v, 3)}</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </section>

      {/* ================= CONFIDENCE + EXPLAINABILITY ================= */}
      <section className="border-y border-line bg-surface">
        <div className="mx-auto grid max-w-6xl gap-12 px-4 py-24 sm:px-6 lg:grid-cols-2">
          <div>
            <SectionHead eyebrow="Reliability" title="It knows when it isn't sure" align="left">
              Scores are temperature-calibrated on held-out validation data, and thresholds tuned on the same split decide how much the
              system is willing to claim for each vehicle.
            </SectionHead>
            <div className="mt-8 space-y-3">
              <Decision status="confident" rule={`p(class) ≥ τ${tau != null ? ` = ${num(tau, 2)}` : ''}`} text="Reports the make, model and year with its confidence." />
              <Decision status="uncertain_model" rule={`p(make) ≥ τ_make${tauMake != null ? ` = ${num(tauMake, 2)}` : ''}`} text="Reports the likely make; flags model and year as uncertain and lists candidates." />
              <Decision status="low_confidence" rule="otherwise" text="Asks for a clearer or closer image and shows the top guesses." />
            </div>
          </div>
          <div>
            <SectionHead eyebrow="Explainability" title="See what the model sees" align="left">
              Every prediction exposes its intermediate steps: illumination correction, edges, contours, region of interest, HOG and texture maps, and a Grad-CAM heat-map.
            </SectionHead>
            <div className="mt-8 grid grid-cols-2 gap-3">
              {[['/landing/edges.jpg', 'Canny edges'], ['/landing/hog.jpg', 'HOG gradients'], ['/landing/contours.jpg', 'Contours + ROI'], ['/landing/gradcam.jpg', 'Grad-CAM']].map(([src, cap]) => (
                <figure key={src} className="overflow-hidden rounded-xl border border-line bg-surface-2">
                  <img src={src} alt={cap} loading="lazy" className="aspect-[16/10] w-full object-cover" />
                  <figcaption className="px-3 py-2 text-xs text-ink-2">{cap}</figcaption>
                </figure>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ================= CTA ================= */}
      <section className="mx-auto max-w-6xl px-4 py-24 sm:px-6">
        <div className="hero-bg relative overflow-hidden rounded-3xl px-6 py-14 text-center text-white sm:px-12">
          <h2 className="text-3xl font-semibold tracking-[-0.02em] sm:text-4xl">Try it on your own photo</h2>
          <p className="mx-auto mt-3 max-w-xl text-white/70">
            Upload a JPG, PNG or WEBP image, or start from one of the sample test images.
            {health?.device && <> The model is loaded and running on <b className="text-white">{health.device.toUpperCase()}</b>.</>}
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <button onClick={() => go('recognize')} className="group inline-flex items-center gap-2 rounded-lg bg-white px-5 py-3 text-[15px] font-medium text-[#0a0f1a] hover:bg-white/90">
              Open the recogniser <IconArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </button>
            <button onClick={() => go('pipeline')} className="inline-flex items-center gap-2 rounded-lg border border-white/20 px-5 py-3 text-[15px] font-medium text-white/90 hover:bg-white/5">
              Explore the pipeline
            </button>
          </div>
        </div>
      </section>

      <Footer go={go} />
    </div>
  )
}

/* ---------------- hero product shot ---------------- */
function HeroShot() {
  const b = HERO.box
  return (
    <div className="relative mx-auto w-full max-w-xl lg:max-w-none">
      <div className="rounded-2xl bg-white/5 p-2 ring-1 ring-white/10 backdrop-blur">
        <div className="relative overflow-hidden rounded-xl">
          <img src="/landing/hero.jpg" alt="Pickup truck recognised by the system" className="block w-full" />
          <div className="absolute rounded-[3px] border-2 border-[#22c55e] shadow-[0_0_0_1px_rgba(0,0,0,0.25)]" style={{ left: `${b.left}%`, top: `${b.top}%`, width: `${b.width}%`, height: `${b.height}%` }}>
            <span className="scanline" />
            <span className="absolute -top-[26px] -left-[2px] whitespace-nowrap rounded-t-[3px] bg-[#22c55e] px-2 py-1 text-[11px] font-semibold text-[#06210f] sm:text-xs">
              {HERO.make} Silverado 1500 · {pct(HERO.confidence)}
            </span>
          </div>
        </div>
      </div>

      {/* floating result card: overlaps only the image's bottom edge */}
      <div className="relative mt-4 rounded-xl border border-line bg-surface p-4 text-ink shadow-[var(--shadow-pop)] md:absolute md:-bottom-20 md:-left-8 md:mt-0 md:w-[17.5rem]">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">Prediction</span>
          <StatusBadge status="confident" />
        </div>
        <div className="mt-2.5 text-[15px] font-semibold leading-snug">{HERO.make} {HERO.model}</div>
        <div className="text-sm text-ink-2">{HERO.year} · detector {pct(HERO.det, 0)}</div>
        <div className="mt-3 flex items-center gap-2.5">
          <div className="h-1.5 flex-1 rounded-full bg-surface-2"><div className="h-1.5 rounded-full bg-good" style={{ width: `${HERO.confidence * 100}%` }} /></div>
          <span className="text-xs font-semibold tabular">{pct(HERO.confidence)}</span>
        </div>
      </div>

      {/* floating grad-cam chip */}
      <div className="absolute -right-3 -top-6 hidden w-40 overflow-hidden rounded-xl border border-white/15 bg-[#0a0f1a] shadow-[var(--shadow-pop)] md:block">
        <img src="/landing/gradcam.jpg" alt="Grad-CAM heat-map" className="aspect-[16/10] w-full object-cover" />
        <div className="flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] text-white/80"><IconEye className="h-3.5 w-3.5" /> Grad-CAM</div>
      </div>
    </div>
  )
}

/* ---------------- small building blocks ---------------- */
function Stat({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <div className="border-line px-5 py-6 [&:not(:last-child)]:border-r max-md:[&:nth-child(2)]:border-r-0 max-md:[&:nth-child(-n+2)]:border-b sm:px-7">
      <div className="text-[30px] font-semibold leading-none tracking-[-0.03em] text-ink sm:text-[34px]">{value}</div>
      <div className="mt-2 text-sm font-medium text-ink">{label}</div>
      <div className="text-xs text-muted">{note}</div>
    </div>
  )
}

function SectionHead({ eyebrow, title, children, align = 'center' }: { eyebrow: string; title: string; children?: ReactNode; align?: 'center' | 'left' }) {
  return (
    <div className={align === 'center' ? 'mx-auto max-w-2xl text-center' : ''}>
      <div className="text-xs font-semibold uppercase tracking-[0.1em] text-accent">{eyebrow}</div>
      <h2 className="mt-2 text-[28px] font-semibold leading-tight tracking-[-0.02em] text-ink sm:text-[34px]">{title}</h2>
      {children && <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{children}</p>}
    </div>
  )
}

const StepNo = ({ n }: { n: number }) => (
  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent text-xs font-semibold text-white tabular">{n}</span>
)

function Frame({ n, src, title, text }: { n: number; src: string; title: string; text: string }) {
  return (
    <li className="flex flex-col rounded-2xl border border-line bg-surface p-4 shadow-[var(--shadow-card)]">
      <StepNo n={n} />
      <div className="mt-3 overflow-hidden rounded-xl bg-surface-2">
        <img src={src} alt={title} loading="lazy" className="aspect-[4/3] w-full object-cover" />
      </div>
      <div className="mt-3 text-sm font-medium text-ink">{title}</div>
      <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{text}</p>
    </li>
  )
}

function Pillar({ icon, title, items }: { icon: ReactNode; title: string; items: string[] }) {
  return (
    <div className="rounded-2xl border border-line bg-page p-6">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-wash/50 text-accent">{icon}</div>
      <h3 className="mt-4 text-lg font-semibold tracking-[-0.01em] text-ink">{title}</h3>
      <ul className="mt-4 space-y-2.5">
        {items.map((t) => (
          <li key={t} className="flex gap-2.5 text-sm text-ink-2">
            <IconCheck className="mt-0.5 h-4 w-4 shrink-0 text-good-ink" />
            <span>{t}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Tool({ icon, title, text, tab, go }: { icon: ReactNode; title: string; text: string; tab: string; go: Go }) {
  return (
    <button onClick={() => go(tab)} className="group flex flex-col rounded-2xl border border-line bg-surface p-6 text-left shadow-[var(--shadow-card)] transition hover:-translate-y-0.5 hover:border-accent/50 hover:shadow-[var(--shadow-pop)]">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-wash/50 text-accent transition group-hover:bg-accent group-hover:text-white">{icon}</div>
      <h3 className="mt-4 text-[15px] font-semibold text-ink">{title}</h3>
      <p className="mt-1.5 flex-1 text-[13px] leading-relaxed text-ink-2">{text}</p>
      <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-accent">Open <IconArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" /></span>
    </button>
  )
}

function Approach({ name, detail, r, featured }: { name: string; detail: string; r?: Row; featured?: boolean }) {
  const t1 = typeof r?.top1 === 'number' ? r.top1 : null
  return (
    <div className={`rounded-2xl border bg-surface p-6 shadow-[var(--shadow-card)] ${featured ? 'border-accent ring-1 ring-accent' : 'border-line'}`}>
      <div className="flex items-center justify-between">
        <div className="text-[15px] font-semibold text-ink">{name}</div>
        {featured && <span className="rounded-full bg-accent px-2 py-0.5 text-[11px] font-medium text-white">Deployed</span>}
      </div>
      <p className="mt-1 min-h-10 text-[13px] text-ink-2">{detail}</p>
      <div className="mt-5 text-[40px] font-semibold leading-none tracking-[-0.03em] text-ink">{pct(t1)}</div>
      <div className="mt-1 text-xs text-muted">top-1 accuracy</div>
      <div className="mt-3 h-1.5 rounded-full bg-surface-2">
        <div className="h-1.5 rounded-full" style={{ width: `${(t1 ?? 0) * 100}%`, background: 'var(--series-1)', opacity: featured ? 1 : 0.55 }} />
      </div>
      <dl className="mt-5 grid grid-cols-3 gap-2 border-t border-grid pt-4 text-xs">
        <div><dt className="text-muted">Top-5</dt><dd className="mt-0.5 font-medium tabular text-ink">{pct(r?.top5)}</dd></div>
        <div><dt className="text-muted">Make</dt><dd className="mt-0.5 font-medium tabular text-ink">{pct(r?.make_acc)}</dd></div>
        <div><dt className="text-muted">Latency</dt><dd className="mt-0.5 font-medium tabular text-ink">{ms(r?.ms_per_image)}</dd></div>
      </dl>
    </div>
  )
}

function Decision({ status, rule, text }: { status: 'confident' | 'uncertain_model' | 'low_confidence'; rule: string; text: string }) {
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-line bg-page p-4 sm:flex-row sm:items-start sm:gap-4">
      <div className="shrink-0 sm:w-48"><StatusBadge status={status} /></div>
      <div className="min-w-0">
        <div className="font-mono text-xs text-muted">{rule}</div>
        <div className="mt-0.5 text-sm text-ink-2">{text}</div>
      </div>
    </div>
  )
}

export function Footer({ go, wide }: { go: Go; wide?: boolean }) {
  return (
    <footer className="border-t border-line bg-surface">
      <div className={`mx-auto grid gap-8 px-4 py-10 text-sm sm:px-6 md:grid-cols-[1.4fr_1fr_1fr] ${wide ? 'max-w-7xl' : 'max-w-6xl'}`}>
        <div>
          <div className="flex items-center gap-2 font-semibold text-ink"><CarMark className="h-5 w-5 text-accent" /> Fine-Grained Vehicle Recognition</div>
          <p className="mt-2 max-w-sm text-[13px] leading-relaxed text-ink-2">
            P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System. Trained on Stanford Cars (16,185 images, 196 classes). The dataset has no
            number plates or video, so licence-plate reading and tracking are out of scope.
          </p>
        </div>
        <div>
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">Tools</div>
          <ul className="space-y-1.5 text-ink-2">
            {([['recognize', 'Recognize & live camera', IconScan], ['batch', 'Batch analysis', IconGrid], ['stress', 'Stress lab', IconFlask], ['pipeline', 'Pipeline', IconLayers], ['catalogue', 'Coverage catalogue', IconSearch]] as const).map(([id, l, I]) => (
              <li key={id}><button onClick={() => go(id)} className="inline-flex items-center gap-2 hover:text-ink"><I className="h-4 w-4" />{l}</button></li>
            ))}
          </ul>
        </div>
        <div>
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">Evaluation</div>
          <ul className="space-y-1.5 text-ink-2">
            {([['comparison', 'Model comparison', IconChart], ['robustness', 'Robustness & confidence', IconShield], ['failures', 'Failure analysis', IconEye], ['about', 'About the system', IconCpu]] as const).map(([id, l, I]) => (
              <li key={id}><button onClick={() => go(id)} className="inline-flex items-center gap-2 hover:text-ink"><I className="h-4 w-4" />{l}</button></li>
            ))}
          </ul>
        </div>
      </div>
      <div className="border-t border-line py-4 text-center text-xs text-muted">YOLOv8 · ConvNeXt-Tiny · OpenCV · scikit-image · FastAPI · React</div>
    </footer>
  )
}
