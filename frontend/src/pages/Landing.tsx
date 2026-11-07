import { useRef } from 'react'
import type { ReactNode } from 'react'
import type { Health, Metrics, Row } from '../lib/api'
import { ms, num, pct } from '../lib/format'
import {
  IconArrowRight, IconCamera, IconPlay, IconChart, IconCheck, IconCpu, IconEye, IconFlask, IconGrid, IconHelp, IconLayers, IconMerge, IconScan,
  IconSearch, IconShield, IconSliders,
} from '../components/icons'
import { StatusBadge } from '../components/ui'
import { LogoMark } from '../components/Logo'

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
      {/* ================= ANNOUNCEMENT BAR (dark bookend) ================= */}
      <div className="bg-carbon px-4 py-2 text-center text-[12px] font-medium text-white">
        Live demo: upload a photo, use your camera, or stress-test the model.{' '}
        <button onClick={() => go('recognize')} className="font-semibold text-accent hover:underline">Try it now</button>
      </div>

      {/* ================= NAV ================= */}
      <nav className="border-b border-line bg-white">
        <div className="mx-auto flex max-w-[1200px] items-center gap-8 px-4 py-4 sm:px-6">
          <div className="flex items-center gap-2.5">
            <LogoMark className="h-8 w-8" />
            <span className="text-[17px] font-semibold tracking-[-0.02em]"><span className="hidden sm:inline">Fine-Grained </span>Vehicle Recognition</span>
          </div>
          <div className="hidden items-center gap-6 text-[14px] font-medium text-ink md:flex">
            <button onClick={() => how.current?.scrollIntoView()} className="hover:text-ink-2">How it works</button>
            <button onClick={() => tools.current?.scrollIntoView()} className="hover:text-ink-2">Tools</button>
            <button onClick={() => results.current?.scrollIntoView()} className="hover:text-ink-2">Results</button>
          </div>
          <div className="ml-auto flex items-center gap-5">
            <button onClick={() => go('comparison')} className="hidden text-[14px] font-medium text-ink hover:text-ink-2 sm:inline">See the evaluation</button>
            <button onClick={() => go('recognize')} className="whitespace-nowrap rounded-lg bg-accent px-4 py-2 text-[14px] font-semibold text-white hover:brightness-95">Open the app</button>
          </div>
        </div>
      </nav>

      {/* ================= HERO ================= */}
      <section className="bg-white">
        <div className="mx-auto grid max-w-[1200px] items-center gap-12 px-4 pb-16 pt-14 sm:px-6 md:pb-24 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)] lg:pt-20">
          <div>
            <p className="mb-5 inline-flex items-center gap-2 rounded-md bg-surface-2 px-2.5 py-1 text-[12px] font-medium text-ink-2">
              Stanford Cars, 196 classes, classical CV + deep learning
            </p>
            <h1 className="text-[44px] font-semibold leading-[1.02] tracking-[-0.03em] sm:text-[56px] lg:text-[60px] lg:leading-none xl:text-[64px]">
              Identify the make, model and year of a car from one photo.
            </h1>
            <p className="mt-6 max-w-[36rem] text-[20px] leading-[1.4] text-ink-2">
              Classical image processing, a fine-tuned YOLOv8 detector and a ConvNeXt classifier work together. Every prediction comes with a calibrated confidence and a visual explanation.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-6">
              <button onClick={() => go('recognize')} className="rounded-lg bg-accent px-5 py-3 text-[16px] font-semibold text-white hover:brightness-95">
                Try it with a photo
              </button>
              <button onClick={() => go('stress')} className="inline-flex items-center gap-2 text-[14px] font-medium text-ink hover:text-ink-2">
                <IconPlay className="h-4 w-4 text-accent" /> Watch it handle night and fog
              </button>
            </div>
            <div className="mt-8 flex flex-wrap gap-x-6 gap-y-2 text-[14px] text-muted">
              <span className="inline-flex items-center gap-1.5"><IconCheck className="h-4 w-4 text-ink" /> Multiple vehicles per image</span>
              <span className="inline-flex items-center gap-1.5"><IconCheck className="h-4 w-4 text-ink" /> Grad-CAM explanations</span>
              <span className="inline-flex items-center gap-1.5"><IconCheck className="h-4 w-4 text-ink" /> Runs on CPU or GPU</span>
            </div>
          </div>

          <HeroShot />
        </div>
      </section>

      {/* ================= STATS ================= */}
      <section className="mx-auto max-w-[1200px] px-4 sm:px-6">
        <div className="grid grid-cols-2 overflow-hidden rounded-2xl border border-line bg-white md:grid-cols-4">
          <Stat value={pct(hybrid?.top1)} label="Top-1 accuracy" note="exact make, model and year" />
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
              <div className="mt-3 h-1.5 rounded-full bg-accent-wash/60"><div className="h-1.5 rounded-full bg-ink" style={{ width: `${HERO.confidence * 100}%` }} /></div>
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
              <button onClick={() => go('comparison')} className="inline-flex items-center gap-1 text-sm text-ink hover:underline">
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
                      <span className="block h-2 rounded-full" style={{ width: `${(typeof v === 'number' ? v : 0) * 100}%`, background: 'var(--bar)', opacity: k === 'yolov8s_finetuned' ? 1 : 0.5 }} />
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
      <section className="bg-surface-2">
        <div className="mx-auto max-w-[1200px] px-4 py-20 text-center sm:px-6">
          <h2 className="text-[36px] font-semibold leading-[1.21] tracking-[-0.02em]">Try it on your own photo</h2>
          <p className="mx-auto mt-3 max-w-xl text-[16px] text-ink-2">
            Upload a JPG, PNG or WEBP image, or start from one of the sample test images.
            {health?.device && <> The model is loaded and running on <b className="font-semibold text-ink">{health.device.toUpperCase()}</b>.</>}
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <button onClick={() => go('recognize')} className="rounded-lg bg-accent px-5 py-3 text-[16px] font-semibold text-white hover:brightness-95">
              Open the recogniser
            </button>
            <button onClick={() => go('pipeline')} className="rounded-lg border border-mist bg-white px-5 py-3 text-[16px] font-medium text-ink hover:bg-surface-2">
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
      <div className="rounded-2xl bg-surface-2 p-2">
        <div className="relative overflow-hidden rounded-xl">
          <img src="/landing/hero.jpg" alt="Pickup truck recognised by the system" className="block w-full" />
          <div className="absolute rounded-md border-2 border-white" style={{ left: `${b.left}%`, top: `${b.top}%`, width: `${b.width}%`, height: `${b.height}%` }}>
            <span className="scanline" />
            <span className="absolute left-2 top-2 whitespace-nowrap rounded-md bg-white px-2 py-1 text-[12px] font-semibold text-ink">
              {HERO.make} Silverado 1500, {pct(HERO.confidence)}
            </span>
          </div>
        </div>
      </div>

      {/* floating result card: overlaps only the image's bottom edge */}
      <div className="relative mt-4 rounded-xl border border-line bg-surface p-4 text-ink shadow-[var(--shadow-pop)] md:absolute md:-bottom-20 md:-left-8 md:mt-0 md:w-[17.5rem]">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[12px] font-medium text-muted">Prediction</span>
          <StatusBadge status="confident" />
        </div>
        <div className="mt-2.5 text-[15px] font-semibold leading-snug">{HERO.make} {HERO.model}</div>
        <div className="text-sm text-ink-2">{HERO.year}, detector {pct(HERO.det, 0)}</div>
        <div className="mt-3 flex items-center gap-2.5">
          <div className="h-1.5 flex-1 rounded-full bg-surface-2"><div className="h-1.5 rounded-full bg-ink" style={{ width: `${HERO.confidence * 100}%` }} /></div>
          <span className="text-xs font-semibold tabular">{pct(HERO.confidence)}</span>
        </div>
      </div>

      {/* floating grad-cam chip */}
      <div className="absolute -right-3 -top-6 hidden w-40 overflow-hidden rounded-xl border border-line bg-white shadow-[var(--shadow-pop)] md:block">
        <img src="/landing/gradcam.jpg" alt="Grad-CAM heat-map" className="aspect-[16/10] w-full object-cover" />
        <div className="flex items-center gap-1.5 px-2.5 py-1.5 text-[12px] font-medium text-ink"><IconEye className="h-3.5 w-3.5" /> Grad-CAM</div>
      </div>
    </div>
  )
}

/* ---------------- small building blocks ---------------- */
function Stat({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <div className="border-line px-5 py-6 [&:not(:last-child)]:border-r max-md:[&:nth-child(2)]:border-r-0 max-md:[&:nth-child(-n+2)]:border-b sm:px-7">
      <div className="text-[36px] font-semibold leading-none tracking-[-0.02em] text-ink sm:text-[48px] sm:tracking-[-0.025em]">{value}</div>
      <div className="mt-2 text-sm font-medium text-ink">{label}</div>
      <div className="text-xs text-muted">{note}</div>
    </div>
  )
}

function SectionHead({ eyebrow, title, children, align = 'center' }: { eyebrow: string; title: string; children?: ReactNode; align?: 'center' | 'left' }) {
  return (
    <div className={align === 'center' ? 'mx-auto max-w-2xl text-center' : ''}>
      <div className="text-[14px] font-medium text-muted">{eyebrow}</div>
      <h2 className="mt-2 text-[36px] font-semibold leading-[1.21] tracking-[-0.02em] text-ink sm:text-[48px] sm:leading-[1.2] sm:tracking-[-0.025em]">{title}</h2>
      {children && <p className="mt-4 text-[16px] leading-[1.5] text-ink-2">{children}</p>}
    </div>
  )
}

const StepNo = ({ n }: { n: number }) => (
  <span className="flex h-6 w-6 items-center justify-center rounded-md bg-ink text-xs font-semibold text-white tabular">{n}</span>
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
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-surface-2 text-ink">{icon}</div>
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
    <button onClick={() => go(tab)} className="group flex flex-col rounded-2xl border border-line bg-surface p-6 text-left shadow-[var(--shadow-card)] transition hover:border-mist">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-surface-2 text-ink transition group-hover:bg-ink group-hover:text-white">{icon}</div>
      <h3 className="mt-4 text-[15px] font-semibold text-ink">{title}</h3>
      <p className="mt-1.5 flex-1 text-[13px] leading-relaxed text-ink-2">{text}</p>
      <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-ink">Open <IconArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" /></span>
    </button>
  )
}

function Approach({ name, detail, r, featured }: { name: string; detail: string; r?: Row; featured?: boolean }) {
  const t1 = typeof r?.top1 === 'number' ? r.top1 : null
  return (
    <div className={`rounded-2xl border bg-surface p-6 shadow-[var(--shadow-card)] ${featured ? 'border-ink ring-1 ring-ink' : 'border-line'}`}>
      <div className="flex items-center justify-between">
        <div className="text-[15px] font-semibold text-ink">{name}</div>
        {featured && <span className="rounded-md bg-ink px-2 py-0.5 text-[12px] font-medium text-white">Deployed</span>}
      </div>
      <p className="mt-1 min-h-10 text-[13px] text-ink-2">{detail}</p>
      <div className="mt-5 text-[40px] font-semibold leading-none tracking-[-0.03em] text-ink">{pct(t1)}</div>
      <div className="mt-1 text-xs text-muted">top-1 accuracy</div>
      <div className="mt-3 h-1.5 rounded-full bg-surface-2">
        <div className="h-1.5 rounded-full" style={{ width: `${(t1 ?? 0) * 100}%`, background: 'var(--bar)', opacity: featured ? 1 : 0.55 }} />
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
    <footer className="bg-abyss text-[#b9bbc6]">
      <div className={`mx-auto grid gap-8 px-4 py-10 text-sm sm:px-6 md:grid-cols-[1.4fr_1fr_1fr] ${wide ? 'max-w-7xl' : 'max-w-6xl'}`}>
        <div>
          <div className="flex items-center gap-2 font-semibold text-white"><LogoMark className="h-7 w-7" /> Fine-Grained Vehicle Recognition</div>
          <p className="mt-2 max-w-sm text-[13px] leading-relaxed text-[#b9bbc6]">
            P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System. Trained on Stanford Cars (16,185 images, 196 classes). The dataset has no
            number plates or video, so licence-plate reading and tracking are out of scope.
          </p>
        </div>
        <div>
          <div className="mb-2 text-[14px] font-semibold text-white">Tools</div>
          <ul className="space-y-1.5 text-[#b9bbc6]">
            {([['recognize', 'Recognize & live camera', IconScan], ['batch', 'Batch analysis', IconGrid], ['stress', 'Stress lab', IconFlask], ['pipeline', 'Pipeline', IconLayers], ['catalogue', 'Coverage catalogue', IconSearch]] as const).map(([id, l, I]) => (
              <li key={id}><button onClick={() => go(id)} className="inline-flex items-center gap-2 hover:text-white"><I className="h-4 w-4" />{l}</button></li>
            ))}
          </ul>
        </div>
        <div>
          <div className="mb-2 text-[14px] font-semibold text-white">Evaluation</div>
          <ul className="space-y-1.5 text-[#b9bbc6]">
            {([['comparison', 'Model comparison', IconChart], ['robustness', 'Robustness & confidence', IconShield], ['failures', 'Failure analysis', IconEye], ['about', 'About the system', IconCpu]] as const).map(([id, l, I]) => (
              <li key={id}><button onClick={() => go(id)} className="inline-flex items-center gap-2 hover:text-white"><I className="h-4 w-4" />{l}</button></li>
            ))}
          </ul>
        </div>
      </div>
      <div className="border-t border-white/10 py-5 text-center text-[12px] text-[#8b8d98]">YOLOv8 · ConvNeXt-Tiny · OpenCV · scikit-image · FastAPI · React</div>
    </footer>
  )
}
