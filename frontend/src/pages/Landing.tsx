import { Fragment, useCallback, useEffect, useRef, useState } from 'react'
import type { ComponentType, CSSProperties, ReactNode, SVGProps } from 'react'
import type { CarsMetrics, Health, Metrics, Row } from '../lib/api'
import { ms, num, pct } from '../lib/format'
import {
  IconArrowRight, IconCamera, IconPlay, IconChart, IconCheck, IconCpu, IconEye, IconFlask, IconGrid, IconHelp, IconLayers, IconMerge, IconScan,
  IconSearch, IconShield, IconSliders, IconSparkles,
} from '../components/icons'
import { StatusBadge } from '../components/ui'
import { LogoMark } from '../components/Logo'
import { Speedometer } from '../components/Speedometer'

type Go = (tab: string) => void
type Icon = ComponentType<SVGProps<SVGSVGElement>>

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

const SCENE_MS = 5000

/** Props for an element that fades and lifts in the first time it scrolls into view (carai.me style reveal). */
function useReveal<T extends HTMLElement>(delay = 0) {
  const ref = useRef<T>(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (typeof IntersectionObserver === 'undefined') { setSeen(true); return }
    const io = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setSeen(true); io.disconnect() } },
      { threshold: 0.12, rootMargin: '0px 0px -6% 0px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return { ref, 'data-reveal': '', 'data-in': seen, style: { '--delay': `${delay}ms` } as CSSProperties }
}

function Reveal({ children, className = '', delay = 0, id }: { children: ReactNode; className?: string; delay?: number; id?: string }) {
  const rv = useReveal<HTMLDivElement>(delay)
  return <div {...rv} id={id} className={className}>{children}</div>
}

export default function Landing({ metrics, carsMetrics, health, healthErr, go }: { metrics: Metrics | null; carsMetrics?: CarsMetrics | null; health: Health | null; healthErr?: string | null; go: Go }) {
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
  const map50 = det?.yolov8s_finetuned?.mAP50
  const map5095 = det?.yolov8s_finetuned?.mAP50_95

  const scenes: Scene[] = [
    {
      id: 'recognize', label: 'Recognize', img: '/landing/hero.jpg', alt: 'Pickup truck recognised by the system', box: true,
      card: { icon: IconScan, title: 'Silverado 1500', detail: 'Make · model · year' },
      metric: { icon: IconCheck, label: 'Calibrated confidence', value: pct(HERO.confidence), small: 'top-1' },
    },
    {
      id: 'contours', label: 'Edges & contours', img: '/landing/contours.jpg', alt: 'Canny edges and contour analysis of the vehicle',
      card: { icon: IconLayers, title: 'Classical vision', detail: 'Canny · contours · HOG' },
      metric: { icon: IconChart, label: 'Top-1 accuracy', value: pct(hybrid?.top1), small: 'hybrid' },
    },
    {
      id: 'detect', label: 'Detect', img: '/landing/roi.jpg', alt: 'Vehicle region of interest found by the detector',
      card: { icon: IconSearch, title: 'Find every car', detail: 'YOLOv8s detector' },
      metric: { icon: IconScan, label: 'Detection mAP@0.5', value: num(map50, 3), small: 'fine-tuned' },
    },
    {
      id: 'explain', label: 'Explain', img: '/landing/gradcam.jpg', alt: 'Grad-CAM heat-map showing where the model looked',
      card: { icon: IconEye, title: 'See what it saw', detail: 'Grad-CAM on every result' },
      metric: { icon: IconShield, label: 'Knows when unsure', value: '3 tiers', small: 'confident · uncertain · low' },
    },
  ]

  const connection = healthErr
    ? { tone: '#ff6b6b', text: 'Backend offline. Start the API to run recognition.' }
    : !health
      ? { tone: '#f5c542', text: 'Connecting to the recognition backend…' }
      : health.model_loaded
        ? { tone: '#00f283', text: `Connected to the backend${health.device ? ` · model ready on ${health.device.toUpperCase()}` : ' · model ready'}` }
        : { tone: '#f5c542', text: 'Backend reachable, model not loaded yet.' }

  return (
    <div className="bg-page text-ink">
      {/* ================= HEADER ================= */}
      <header className="site-header">
        <div className="mx-auto flex h-[74px] w-[91.2%] max-w-[1320px] items-center gap-8">
          <a href="#/" className="flex items-center gap-3" aria-label="Fine-Grained Vehicle Recognition, home">
            <LogoMark className="h-9 w-9" />
            <span className="text-[16px] font-bold tracking-[-0.03em]"><span className="hidden sm:inline">Fine-Grained </span>Vehicle Recognition</span>
          </a>
          <nav className="ml-6 hidden items-center gap-9 md:flex" aria-label="Page sections">
            <button onClick={() => how.current?.scrollIntoView()} className="nav-link">How it works</button>
            <button onClick={() => tools.current?.scrollIntoView()} className="nav-link">Tools</button>
            <button onClick={() => results.current?.scrollIntoView()} className="nav-link">Results</button>
            <button onClick={() => go('comparison')} className="nav-link">Evaluation</button>
          </nav>
          <div className="ml-auto flex items-center gap-4">
            <span className="hidden items-center gap-2 text-[12px] text-ink-2 lg:inline-flex" style={{ color: connection.tone }} title={connection.text}>
              <span className="connection-dot" />
              <span className="text-ink-2">{healthErr ? 'Backend offline' : !health ? 'Connecting…' : health.model_loaded ? 'Model ready' : 'Model not loaded'}</span>
            </span>
            <button onClick={() => go('recognize')} className="btn-primary min-h-[45px] whitespace-nowrap px-6 py-3 text-[14px]">
              Open the app <IconArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      {/* ================= HERO ================= */}
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-ground" aria-hidden />
        <div className="mx-auto grid w-[91.2%] max-w-[1320px] items-center gap-14 pb-20 pt-16 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:pb-28 lg:pt-24">
          <div className="hero-copy">
            <p className="eyebrow">Your photo. Your car. Your AI.</p>
            <h1 id="hero-title" className="mt-6 text-[clamp(38px,4.6vw,60px)] font-semibold leading-[1.1] tracking-[-0.045em]">
              Know the make, model and year in one photo.
            </h1>
            <p className="mt-7 max-w-[470px] text-[17px] leading-[1.7] text-ink-2">
              Classical vision, a fine-tuned YOLOv8 detector and a ConvNeXt classifier work together. Every answer comes with a calibrated confidence and a visual explanation.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-6">
              <button onClick={() => go('recognize')} className="btn-primary min-h-[54px] px-7 py-4 text-[14px]">
                Try it with a photo <IconArrowRight className="h-4 w-4" />
              </button>
              <button onClick={() => go('stress')} className="group inline-flex items-center gap-3 text-[14px] font-medium">
                <span className="grid h-[41px] w-[41px] place-items-center rounded-full border border-white transition group-hover:bg-white/15"><IconPlay className="h-[15px] w-[15px]" /></span>
                See it handle night and fog
              </button>
            </div>
            <p className="connection-note mt-7" style={{ color: connection.tone }} role="status">
              <span className="connection-dot" aria-hidden />
              <span className="text-ink-2">{connection.text}</span>
            </p>
            <button onClick={() => go('recognize')} className="orb-question mt-9">
              <span className="orb" aria-hidden />
              <span className="orb-prompt">
                Which car is this?
                <IconScan className="h-5 w-5" />
              </span>
            </button>
          </div>

          <HeroShowcase scenes={scenes} go={go} />
        </div>
      </section>

      {/* ================= STATS (SPEEDOMETER COCKPIT CLUSTER) ================= */}
      <Reveal className="mx-auto mt-16 w-[91.2%] max-w-[1320px]">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 px-1">
          <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-accent">
            ● LIVE DYNO & TELEMETRY CLUSTER · STANFORD CARS BENCHMARK
          </span>
          <span className="font-mono text-[11px] text-muted">
            ENGINE: CONVNEXT-TINY + YOLOV8S + CLASSICAL CV · 196 CLASSES
          </span>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Speedometer
            value={typeof hybrid?.top1 === 'number' ? hybrid.top1 : 0.947}
            label="Top-1 Exact Accuracy"
            sublabel="Exact make, model & year"
            color="#00f283"
            gear="S+"
            badge="RPM · TOP-1"
            size={195}
          />
          <Speedometer
            value={typeof hybrid?.make_acc === 'number' ? hybrid.make_acc : 0.984}
            label="Make Accuracy"
            sublabel="Manufacturer recognition"
            color="#21a4ff"
            gear="MAKE"
            badge="BOOST · BRAND"
            size={195}
          />
          <Speedometer
            value={typeof map50 === 'number' ? map50 : 0.99}
            label="Detection mAP@0.5"
            sublabel="Fine-tuned YOLOv8s radar"
            color="#3df5a0"
            gear="RADAR"
            badge="LOCK · YOLOv8s"
            size={195}
          />
          <Speedometer
            value={typeof hybrid?.top5 === 'number' ? hybrid.top5 : 0.994}
            label="Top-5 Recall"
            sublabel="8,041 unseen test photos"
            color="#fab219"
            gear="196"
            badge="REDLINE · TOP-5"
            size={195}
          />
        </div>
      </Reveal>

      {/* ================= ALL-CARS MODEL BAND (1948–2026) ================= */}
      {carsMetrics?.test && (
        <Reveal className="mx-auto mt-10 w-[91.2%] max-w-[1320px]">
          <div className="grid items-center gap-8 rounded-[22px] border border-accent/40 bg-surface p-6 sm:p-9 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
            <div>
              <div className="eyebrow text-accent">All-cars powertrain · 1948–2026</div>
              <h2 className="mt-3 text-[clamp(26px,3.2vw,40px)] font-semibold leading-[1.12] tracking-[-0.04em] text-ink">
                Not just 2012. Cars up to 2026.
              </h2>
              <p className="mt-3 text-[15px] leading-[1.65] text-ink-2">
                A second classifier trained on ~{Math.round((carsMetrics.dataset?.images_used ?? 180000) / 1000)},000 real-world photos names{' '}
                <strong className="text-ink">{carsMetrics.dataset?.classes?.toLocaleString() ?? '1,291'} generations</strong> across{' '}
                {carsMetrics.dataset?.makes ?? 60} makes from 1948 to 2026 (Mustang S650, Ioniq 6, Cybertruck, latest RAV4).
              </p>
              <button onClick={() => go('allcars')} className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-accent hover:underline">
                Explore the All-cars model telemetry <IconArrowRight className="h-4 w-4" />
              </button>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <Speedometer
                value={carsMetrics.test.top1}
                label="Exact Generation"
                sublabel={`${carsMetrics.test.n.toLocaleString()} test photos`}
                color="#00f283"
                gear="1291"
                badge="GEN · TOP-1"
                size={162}
              />
              <Speedometer
                value={carsMetrics.test.make_acc}
                label="Make Accuracy"
                sublabel={`Top-5 ${pct(carsMetrics.test.top5)}`}
                color="#21a4ff"
                gear="MAKE"
                badge="BRAND · LOCK"
                size={162}
              />
              <Speedometer
                value={carsMetrics.test.model_acc_any_generation}
                label="Model (Any Gen)"
                sublabel={`${carsMetrics.dataset?.models ?? '500+'} models`}
                color="#fab219"
                gear="2026"
                badge="MODEL · LOCK"
                size={162}
              />
            </div>
          </div>
        </Reveal>
      )}

      {/* ================= BENEFIT STORY ================= */}
      <section className="mx-auto w-[91.2%] max-w-[1320px] pb-10 pt-16" id="benefits" aria-label="Detect, classify and trust every prediction">
        <nav className="benefit-nav" aria-label="Benefits">
          <a href="#benefit-detect"><span>01</span>Detect<IconArrowRight className="h-4 w-4" /></a>
          <a href="#benefit-classify"><span>02</span>Classify<IconArrowRight className="h-4 w-4" /></a>
          <a href="#benefit-trust"><span>03</span>Trust<IconArrowRight className="h-4 w-4" /></a>
        </nav>

        <Reveal id="benefit-detect" className="benefit-panel">
          <div className="benefit-art">
            <div className="benefit-board">
              <div className="mb-6 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5 text-[13px] font-medium"><IconScan className="h-5 w-5 text-accent" /> Vehicle detection</div>
                <span className="benefit-tag">TEST SET</span>
              </div>
              <div className="grid grid-cols-2 gap-5">
                <Gauge label="mAP@0.5" value={typeof map50 === 'number' ? map50 : 0} text={num(map50, 3)} />
                <Gauge label="mAP@0.5:0.95" value={typeof map5095 === 'number' ? map5095 : 0} text={num(map5095, 3)} green />
              </div>
              <div className="mt-6 flex items-center gap-2 text-[11px] text-[#a7b1b7]">
                <IconCheck className="h-4 w-4 text-[#8fb6a1]" />
                <span>Fine-tuned YOLOv8s, with contour fallback.</span>
              </div>
            </div>
            <div className="benefit-float">
              <span className="orb" aria-hidden />
              <div>
                <span className="k">THE DETECTOR</span>
                <p>Every car in frame.<br /><strong>Boxed in a blink.</strong></p>
              </div>
            </div>
            <div className="benefit-art-caption"><span>VEHICLE AI / DETECT</span><span>LIVE METRICS</span></div>
          </div>
          <div className="benefit-copy">
            <p className="eyebrow">Find the car</p>
            <h2 className="text-[clamp(32px,4.2vw,50px)] font-semibold leading-[1.12] tracking-[-0.045em]">Pixels become a box.</h2>
            <p className="mt-5 max-w-[460px] text-[16px] leading-[1.7] text-ink-2">
              A fine-tuned detector localises each vehicle. When it misses, the classical contour region and then the whole image keep the pipeline answering.
            </p>
            <p className="mt-6 inline-flex items-center gap-2.5 text-[14px] text-[#d9dfe3]"><IconCheck className="h-[18px] w-[18px] text-good" /> Never a silent failure.</p>
            <div><button onClick={() => go('pipeline')} className="mt-6 inline-flex items-center gap-3.5 text-[15px] font-medium text-accent underline decoration-1 underline-offset-[7px] hover:text-[#77caff]">Explore the pipeline <IconArrowRight className="h-[18px] w-[18px]" /></button></div>
          </div>
        </Reveal>

        <Reveal id="benefit-classify" className="benefit-panel is-flipped">
          <div className="benefit-art">
            <div className="benefit-board">
              <div className="mb-6 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5 text-[13px] font-medium"><IconCpu className="h-5 w-5 text-accent" /> Top candidates</div>
                <span className="benefit-tag">EXAMPLE</span>
              </div>
              <ul className="space-y-4">
                {HERO.top5.map(([label, p], i) => (
                  <li key={label}>
                    <div className="flex items-baseline justify-between gap-3 text-[12px]">
                      <span className={i === 0 ? 'font-semibold text-white' : 'text-ink-2'}>{label}</span>
                      <span className="tabular shrink-0 font-semibold text-white">{pct(p, p < 0.01 ? 2 : 1)}</span>
                    </div>
                    <div className="mt-2 h-1.5 rounded-full bg-[#2c3135]">
                      <div className="grow-bar h-1.5 rounded-full" style={{ '--w': `${Math.max(1.5, p * 100)}%`, background: i === 0 ? 'var(--good)' : '#049cfa' } as CSSProperties} />
                    </div>
                  </li>
                ))}
              </ul>
              <div className="mt-6 flex items-center gap-2 text-[11px] text-[#a7b1b7]">
                <IconCheck className="h-4 w-4 text-[#8fb6a1]" />
                <span>Calibrated, not guessed.</span>
                <span className="ml-auto">{HERO.year} · {HERO.make}</span>
              </div>
            </div>
            <div className="benefit-float">
              <span className="orb" aria-hidden />
              <div>
                <span className="k">FINE-GRAINED</span>
                <p>196 look-alikes.<br /><strong>One confident answer.</strong></p>
              </div>
            </div>
            <div className="benefit-art-caption"><span>VEHICLE AI / CLASSIFY</span><span>ILLUSTRATIVE RESULT</span></div>
          </div>
          <div className="benefit-copy">
            <p className="eyebrow">Make · model · year</p>
            <h2 className="text-[clamp(32px,4.2vw,50px)] font-semibold leading-[1.12] tracking-[-0.045em]">A name. A score. A reason.</h2>
            <p className="mt-5 max-w-[460px] text-[16px] leading-[1.7] text-ink-2">
              ConvNeXt-Tiny fused with hand-crafted features scores every class, and each result lists its closest rival so you can see why it won.
            </p>
            <p className="mt-6 inline-flex items-center gap-2.5 text-[14px] text-[#d9dfe3]"><IconCheck className="h-[18px] w-[18px] text-good" /> Top-5 candidates on every photo.</p>
            <div><button onClick={() => go('recognize')} className="mt-6 inline-flex items-center gap-3.5 text-[15px] font-medium text-accent underline decoration-1 underline-offset-[7px] hover:text-[#77caff]">Try the recogniser <IconArrowRight className="h-[18px] w-[18px]" /></button></div>
          </div>
        </Reveal>

        <Reveal id="benefit-trust" className="benefit-panel">
          <div className="benefit-art">
            <div className="benefit-board">
              <div className="mb-5 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5 text-[13px] font-medium"><IconShield className="h-5 w-5 text-accent" /> Decision rule</div>
                <span className="benefit-tag">LIVE</span>
              </div>
              <div className="space-y-2.5">
                <MiniDecision status="confident" rule={`p(class) ≥ τ = ${num(tau, 2)}`} />
                <MiniDecision status="uncertain_model" rule={`p(make) ≥ τ_make = ${num(tauMake, 2)}`} />
                <MiniDecision status="low_confidence" rule="otherwise" />
              </div>
              <div className="mt-6 rounded-2xl border border-[#2e6f4f] bg-[#10231a] p-4">
                <span className="text-[10px] font-semibold tracking-[0.12em] text-[#3df5a0]">TUNED ON HELD-OUT DATA</span>
                <p className="mt-1.5 text-[13px] leading-snug text-[#d4efe2]">Temperature calibration and thresholds are fitted on a separate validation split.</p>
              </div>
            </div>
            <div className="benefit-float">
              <span className="orb" aria-hidden />
              <div>
                <span className="k">YOUR AI COPILOT</span>
                <p>It knows when<br /><strong>it isn't sure.</strong></p>
              </div>
            </div>
            <div className="benefit-art-caption"><span>VEHICLE AI / TRUST</span><span>THRESHOLDS FROM THE API</span></div>
          </div>
          <div className="benefit-copy">
            <p className="eyebrow">Honest confidence</p>
            <h2 className="text-[clamp(32px,4.2vw,50px)] font-semibold leading-[1.12] tracking-[-0.045em]">Certain, or clearly not.</h2>
            <p className="mt-5 max-w-[460px] text-[16px] leading-[1.7] text-ink-2">
              The system reports a full answer only when it is sure, falls back to the likely make when it is not, and asks for a better photo otherwise.
            </p>
            <p className="mt-6 inline-flex items-center gap-2.5 text-[14px] text-[#d9dfe3]"><IconCheck className="h-[18px] w-[18px] text-good" /> Test it with night, fog and blur.</p>
            <div><button onClick={() => go('stress')} className="mt-6 inline-flex items-center gap-3.5 text-[15px] font-medium text-accent underline decoration-1 underline-offset-[7px] hover:text-[#77caff]">Open the stress lab <IconArrowRight className="h-[18px] w-[18px]" /></button></div>
          </div>
        </Reveal>
      </section>

      {/* ================= HOW IT WORKS ================= */}
      <section ref={how} className="mx-auto w-[91.2%] max-w-[1320px] scroll-mt-20 py-24">
        <Reveal>
          <SectionHead eyebrow="How it works" title="From pixels to make, model and year">
            Every image goes through the same pipeline. These are the real intermediate outputs for the pickup truck above.
          </SectionHead>
        </Reveal>
        <ol className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <Frame n={1} delay={0} src="/landing/contours.jpg" title="Edges and contours" text="Canny edges, morphological closing and contour analysis give a classical vehicle region." />
          <Frame n={2} delay={80} src="/landing/roi.jpg" title="Vehicle detection" text="Fine-tuned YOLOv8 localises each car. If it misses, the classical region is used as a fallback." />
          <Frame n={3} delay={160} src="/landing/lbp.jpg" title="Hand-crafted features" text="HOG shape, LBP/GLCM texture, colour histograms and geometry such as symmetry and Hu moments." />
          <Frame n={4} delay={240} src="/landing/gradcam.jpg" title="Deep classification" text="ConvNeXt-Tiny with flip TTA scores 196 classes. Grad-CAM shows where it looked." />
          <li data-reveal data-in="true" className="lift flex flex-col rounded-[22px] border border-line bg-surface p-4">
            <StepNo n={5} />
            <div className="mt-3 flex flex-1 flex-col justify-center rounded-2xl bg-surface-2 p-4">
              <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-muted">Result</div>
              <div className="mt-1 text-[15px] font-semibold leading-snug text-ink">{HERO.make} {HERO.model}</div>
              <div className="text-sm text-ink-2">{HERO.year}</div>
              <div className="mt-3 h-1.5 rounded-full bg-[#2c3135]"><div className="h-1.5 rounded-full bg-good" style={{ width: `${HERO.confidence * 100}%` }} /></div>
              <div className="mt-1.5 text-xs text-ink-2 tabular">{pct(HERO.confidence)} calibrated</div>
            </div>
            <div className="mt-3 text-sm font-semibold text-ink">Fusion and decision</div>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-2">Deep and classical scores are fused, calibrated, and checked against confidence thresholds.</p>
          </li>
        </ol>
      </section>

      {/* ================= PILLARS ================= */}
      <section className="border-y border-line bg-surface/60">
        <div className="mx-auto w-[91.2%] max-w-[1320px] py-24">
          <Reveal><SectionHead eyebrow="Technical components" title="Classical vision, deep learning, and a hybrid of both" /></Reveal>
          <div className="mt-14 grid gap-5 lg:grid-cols-3">
            <Pillar delay={0} icon={<IconScan className="h-5 w-5" />} title="Classical computer vision" items={[
              'Preprocessing: z-score normalisation, adaptive gamma + CLAHE',
              'Canny edge detection and contour analysis',
              'Colour (HSV, LAB) and texture (LBP, GLCM) features',
              'Contour-based region-of-interest extraction',
              'Geometry: aspect ratio, solidity, Hu moments, symmetry',
            ]} />
            <Pillar delay={100} icon={<IconCpu className="h-5 w-5" />} title="Deep learning" items={[
              'ConvNeXt-Tiny fine-grained classifier',
              'YOLOv8s vehicle detector',
              'Transfer learning from ImageNet-22k and COCO, then fine-tuning',
              'Multi-class classification across 196 make/model/year classes',
            ]} />
            <Pillar delay={200} icon={<IconMerge className="h-5 w-5" />} title="Hybrid integration" items={[
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
      <section ref={tools} className="mx-auto w-[91.2%] max-w-[1320px] scroll-mt-20 pt-24">
        <Reveal>
          <SectionHead eyebrow="Tools" title="More than a single prediction">
            Interactive tools for probing, stress-testing and explaining the model, all running against the same deployed pipeline.
          </SectionHead>
        </Reveal>
        <div className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Tool delay={0} icon={<IconCamera className="h-5 w-5" />} title="Live camera" tab="recognize" go={go}
            text="Point a webcam at a car (or a photo of one) and watch boxes and labels track it live." />
          <Tool delay={60} icon={<IconFlask className="h-5 w-5" />} title="Stress lab" tab="stress" go={go}
            text="Add night, fog, blur, CCTV low-res, occlusion or tilt, then see when the prediction breaks and whether the system notices." />
          <Tool delay={120} icon={<IconGrid className="h-5 w-5" />} title="Batch analysis" tab="batch" go={go}
            text="Drop a whole folder, follow the queue, get a make breakdown and accuracy against ground truth, and export CSV or JSON." />
          <Tool delay={0} icon={<IconSliders className="h-5 w-5" />} title="Threshold playground" tab="recognize" go={go}
            text="Drag τ and τ_make on any result and watch the confident / uncertain / low-confidence decision change." />
          <Tool delay={60} icon={<IconHelp className="h-5 w-5" />} title="Why not the runner-up?" tab="recognize" go={go}
            text="Every result explains its closest rival: a year change, a sibling model, or a look-alike from another brand." />
          <Tool delay={120} icon={<IconSearch className="h-5 w-5" />} title="Coverage catalogue" tab="catalogue" go={go}
            text="Can it recognise my car? Search all 196 classes by make, model, year and body style." />
        </div>
      </section>

      {/* ================= RESULTS ================= */}
      <section ref={results} className="mx-auto w-[91.2%] max-w-[1320px] scroll-mt-20 py-24">
        <Reveal>
          <SectionHead eyebrow="Results" title="Evaluated on 8,041 unseen test images">
            Classical features alone fall far short of a fine-tuned network. The hybrid system keeps the deep model's accuracy and adds
            classical preprocessing, fallbacks and interpretability.
          </SectionHead>
        </Reveal>
        <div className="mt-14 grid gap-5 md:grid-cols-3">
          <Approach delay={0} name="Classical" detail="HOG + colour + texture + geometry, logistic regression" r={classical} />
          <Approach delay={100} name="Deep learning" detail="YOLOv8 crop + ConvNeXt-Tiny" r={deep} />
          <Approach delay={200} name="Hybrid" detail="YOLO + classical preprocessing and fusion + ConvNeXt TTA" r={hybrid} featured />
        </div>
        {det && (
          <Reveal className="mt-5 rounded-[22px] border border-line bg-surface p-6 sm:p-8">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <div className="text-[15px] font-semibold text-ink">Vehicle localisation (mAP@0.5:0.95)</div>
              <button onClick={() => go('comparison')} className="inline-flex items-center gap-1.5 text-sm text-accent hover:underline">
                Full comparison <IconArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
            <div className="mt-5 space-y-4">
              {([['classical_contour_roi', 'Classical contour ROI'], ['yolov8s_coco_pretrained', 'YOLOv8s, COCO pretrained'], ['yolov8s_finetuned', 'YOLOv8s, fine-tuned']] as const).map(([k, label]) => {
                const v = det[k]?.mAP50_95
                return (
                  <div key={k} className="grid grid-cols-[minmax(0,11rem)_1fr_3.5rem] items-center gap-3 text-sm sm:grid-cols-[14rem_1fr_4rem]">
                    <span className={k === 'yolov8s_finetuned' ? 'font-medium text-ink' : 'text-ink-2'}>{label}</span>
                    <span className="h-2 rounded-full bg-[#2c3135]">
                      <span className="grow-bar block h-2 rounded-full" style={{ '--w': `${(typeof v === 'number' ? v : 0) * 100}%`, background: 'var(--bar)', opacity: k === 'yolov8s_finetuned' ? 1 : 0.5 } as CSSProperties} />
                    </span>
                    <span className="text-right font-medium tabular text-ink">{num(v, 3)}</span>
                  </div>
                )
              })}
            </div>
          </Reveal>
        )}
      </section>

      {/* ================= CONFIDENCE + EXPLAINABILITY ================= */}
      <section className="border-y border-line bg-surface/60">
        <div className="mx-auto grid w-[91.2%] max-w-[1320px] gap-14 py-24 lg:grid-cols-2">
          <Reveal>
            <SectionHead eyebrow="Reliability" title="It knows when it isn't sure" align="left">
              Scores are temperature-calibrated on held-out validation data, and thresholds tuned on the same split decide how much the
              system is willing to claim for each vehicle.
            </SectionHead>
            <div className="mt-8 space-y-3">
              <Decision status="confident" rule={`p(class) ≥ τ${tau != null ? ` = ${num(tau, 2)}` : ''}`} text="Reports the make, model and year with its confidence." />
              <Decision status="uncertain_model" rule={`p(make) ≥ τ_make${tauMake != null ? ` = ${num(tauMake, 2)}` : ''}`} text="Reports the likely make; flags model and year as uncertain and lists candidates." />
              <Decision status="low_confidence" rule="otherwise" text="Asks for a clearer or closer image and shows the top guesses." />
            </div>
          </Reveal>
          <Reveal delay={120}>
            <SectionHead eyebrow="Explainability" title="See what the model sees" align="left">
              Every prediction exposes its intermediate steps: illumination correction, edges, contours, region of interest, HOG and texture maps, and a Grad-CAM heat-map.
            </SectionHead>
            <div className="mt-8 grid grid-cols-2 gap-3">
              {[['/landing/edges.jpg', 'Canny edges'], ['/landing/hog.jpg', 'HOG gradients'], ['/landing/contours.jpg', 'Contours + ROI'], ['/landing/gradcam.jpg', 'Grad-CAM']].map(([src, cap]) => (
                <figure key={src} className="lift overflow-hidden rounded-2xl border border-line bg-surface-2">
                  <img src={src} alt={cap} loading="lazy" className="aspect-[16/10] w-full object-cover" />
                  <figcaption className="px-3.5 py-2.5 text-xs text-ink-2">{cap}</figcaption>
                </figure>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* ================= CTA ================= */}
      <section className="relative overflow-hidden">
        <div className="hero-ground" aria-hidden style={{ zIndex: 0, opacity: 0.8 }} />
        <Reveal className="relative mx-auto w-[91.2%] max-w-[1320px] py-24 text-center">
          <h2 className="text-[clamp(32px,4.6vw,54px)] font-semibold leading-[1.12] tracking-[-0.045em]">Try it on your own photo</h2>
          <p className="mx-auto mt-4 max-w-xl text-[16px] text-ink-2">
            Upload a JPG, PNG or WEBP image, or start from one of the sample test images.
            {health?.device && <> The model is loaded and running on <b className="font-semibold text-ink">{health.device.toUpperCase()}</b>.</>}
          </p>
          <div className="mt-9 flex flex-wrap justify-center gap-3">
            <button onClick={() => go('recognize')} className="btn-primary min-h-[54px] px-8 py-4 text-[14px]">
              Open the recogniser <IconArrowRight className="h-4 w-4" />
            </button>
            <button onClick={() => go('pipeline')} className="btn-ghost min-h-[54px] px-8 py-4 text-[14px]">
              Explore the pipeline
            </button>
          </div>
        </Reveal>
      </section>

      <Footer go={go} />
    </div>
  )
}

/* ---------------- hero showcase: rotating scenes with floating cards ---------------- */
interface Scene {
  id: string
  label: string
  img: string
  alt: string
  box?: boolean
  card: { icon: Icon; title: string; detail: string }
  metric: { icon: Icon; label: string; value: string; small: string }
}

function HeroShowcase({ scenes, go }: { scenes: Scene[]; go: Go }) {
  const [view, setView] = useState({ cur: 0, prev: -1 })
  const [playing, setPlaying] = useState(() => !(typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches))
  const n = scenes.length
  const show = useCallback((next: number) => setView((s) => (s.cur === next ? s : { cur: next, prev: s.cur })), [])

  // autoplay
  useEffect(() => {
    if (!playing) return
    const t = window.setTimeout(() => show((view.cur + 1) % n), SCENE_MS)
    return () => window.clearTimeout(t)
  }, [playing, view.cur, n, show])

  // retire the outgoing scene once its exit animation has finished
  useEffect(() => {
    if (view.prev < 0) return
    const t = window.setTimeout(() => setView((s) => ({ ...s, prev: -1 })), 1700)
    return () => window.clearTimeout(t)
  }, [view.cur, view.prev])

  const b = HERO.box
  const stateOf = (i: number) => (i === view.cur ? 'is-current' : i === view.prev ? 'is-outgoing' : '')
  const active = scenes[view.cur]

  return (
    <div className="hero-showcase" role="region" aria-roledescription="carousel" aria-label="Explore the recognition pipeline">
      <div className="hero-device">
        <div className="hero-screen">
          {scenes.map((s, i) => (
            <div key={s.id} className={`hero-scene-layer ${stateOf(i)}`} aria-hidden={i !== view.cur}>
              <img src={s.img} alt={s.alt} loading={i === 0 ? 'eager' : 'lazy'} draggable={false} />
              {s.box && (
                <div className="absolute rounded-md border-2 border-white" style={{ left: `${b.left}%`, top: `${b.top}%`, width: `${b.width}%`, height: `${b.height}%` }}>
                  <span className="scanline" />
                  <span className="absolute left-2 top-2 whitespace-nowrap rounded-md bg-white px-2 py-1 text-[11px] font-bold text-[#111]">
                    {HERO.make} Silverado 1500, {pct(HERO.confidence)}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {scenes.map((s, i) => {
        const Card = s.card.icon
        const Metric = s.metric.icon
        return (
          <Fragment key={s.id}>
            <button type="button" onClick={() => go('recognize')} className={`hero-card ${stateOf(i)}`} aria-hidden={i !== view.cur} tabIndex={i === view.cur ? 0 : -1}
              aria-label={`${s.card.title}: ${s.card.detail}. Open the app`}>
              <span className="flex items-center gap-3">
                <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-white/10"><Card className="h-5 w-5" /></span>
                <span className="min-w-0">
                  <span className="block text-[13px] font-semibold leading-tight">{s.card.title}</span>
                  <span className="mt-1 flex items-center gap-1 text-[11px] text-[#aeb6bb]">{s.card.detail}<IconArrowRight className="h-3 w-3 text-accent" /></span>
                </span>
              </span>
            </button>
            <div className={`hero-metric ${stateOf(i)}`} aria-hidden={i !== view.cur}>
              <Metric className="h-5 w-5 text-accent" />
              <div className="mt-2.5 text-[11px] text-[#afb5ba]">{s.metric.label}</div>
              <div className="mt-1.5 flex items-baseline gap-2.5">
                <strong className="text-[22px] font-semibold leading-none tabular">{s.metric.value}</strong>
                <small className="text-[10px] text-[#a2a9af]">{s.metric.small}</small>
              </div>
            </div>
          </Fragment>
        )
      })}

      <div className="hero-controls">
        <div className="hero-progress" aria-hidden style={{ '--dur': `${SCENE_MS}ms` } as CSSProperties}>
          <span key={`${view.cur}-${playing}`} className={playing ? '' : 'is-paused'} />
        </div>
        <span className="min-w-[8.5rem] text-[12px] font-medium" aria-live="off">
          <span className="mr-2 text-[#82898f]">{String(view.cur + 1).padStart(2, '0')}</span>{active.label}
        </span>
        <div className="hero-picker" role="group" aria-label="Choose a scene">
          {scenes.map((s, i) => (
            <button key={s.id} type="button" aria-label={s.label} aria-pressed={i === view.cur} onClick={() => show(i)}><span /></button>
          ))}
        </div>
        <button type="button" onClick={() => setPlaying((p) => !p)} aria-label={playing ? 'Pause the slideshow' : 'Play the slideshow'}
          className="grid h-8 w-8 place-items-center rounded-full border border-white/25 hover:bg-white/10">
          {playing
            ? <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor" aria-hidden><rect x="5" y="4" width="5" height="16" rx="1" /><rect x="14" y="4" width="5" height="16" rx="1" /></svg>
            : <IconPlay className="h-3.5 w-3.5" />}
        </button>
      </div>
    </div>
  )
}

/* ---------------- small building blocks ---------------- */
function SectionHead({ eyebrow, title, children, align = 'center' }: { eyebrow: string; title: string; children?: ReactNode; align?: 'center' | 'left' }) {
  return (
    <div className={align === 'center' ? 'mx-auto max-w-2xl text-center' : ''}>
      <div className="eyebrow">{eyebrow}</div>
      <h2 className="mt-4 text-[clamp(30px,4.2vw,50px)] font-semibold leading-[1.12] tracking-[-0.045em] text-ink">{title}</h2>
      {children && <p className="mt-5 text-[16px] leading-[1.7] text-ink-2">{children}</p>}
    </div>
  )
}

const StepNo = ({ n }: { n: number }) => (
  <span className="flex h-7 w-7 items-center justify-center rounded-full border border-white/25 text-xs font-semibold text-ink tabular">{n}</span>
)

function Frame({ n, src, title, text, delay }: { n: number; src: string; title: string; text: string; delay: number }) {
  const rv = useReveal<HTMLLIElement>(delay)
  return (
    <li {...rv} className="lift flex flex-col rounded-[22px] border border-line bg-surface p-4">
      <StepNo n={n} />
      <div className="mt-3 overflow-hidden rounded-2xl bg-surface-2">
        <img src={src} alt={title} loading="lazy" className="aspect-[4/3] w-full object-cover transition duration-500 hover:scale-105" />
      </div>
      <div className="mt-3 text-sm font-semibold text-ink">{title}</div>
      <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{text}</p>
    </li>
  )
}

function Pillar({ icon, title, items, delay }: { icon: ReactNode; title: string; items: string[]; delay: number }) {
  return (
    <Reveal delay={delay} className="lift rounded-[22px] border border-line bg-page p-7">
      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-surface-2 text-accent">{icon}</div>
      <h3 className="mt-5 text-lg font-semibold tracking-[-0.03em] text-ink">{title}</h3>
      <ul className="mt-4 space-y-3">
        {items.map((t) => (
          <li key={t} className="flex gap-2.5 text-sm text-ink-2">
            <IconCheck className="mt-0.5 h-4 w-4 shrink-0 text-good" />
            <span>{t}</span>
          </li>
        ))}
      </ul>
    </Reveal>
  )
}

function Tool({ icon, title, text, tab, go, delay }: { icon: ReactNode; title: string; text: string; tab: string; go: Go; delay: number }) {
  const rv = useReveal<HTMLButtonElement>(delay)
  return (
    <button {...rv} onClick={() => go(tab)} className="group lift flex flex-col rounded-[22px] border border-line bg-surface p-7 text-left">
      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-surface-2 text-ink transition group-hover:bg-accent group-hover:text-[#08121a]">{icon}</div>
      <h3 className="mt-5 text-[16px] font-semibold tracking-[-0.02em] text-ink">{title}</h3>
      <p className="mt-2 flex-1 text-[13px] leading-relaxed text-ink-2">{text}</p>
      <span className="mt-5 inline-flex items-center gap-1.5 text-sm font-medium text-accent">Open <IconArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-1" /></span>
    </button>
  )
}

function Approach({ name, detail, r, featured, delay }: { name: string; detail: string; r?: Row; featured?: boolean; delay: number }) {
  const t1 = typeof r?.top1 === 'number' ? r.top1 : null
  const gear = name === 'Hybrid' ? 'S+' : name === 'Deep learning' ? '5TH' : '1ST'
  const dialColor = name === 'Hybrid' ? '#00f283' : name === 'Deep learning' ? '#21a4ff' : '#fab219'
  return (
    <Reveal delay={delay} className={`lift rounded-[22px] border bg-surface p-6 ${featured ? 'border-accent/70 shadow-[0_0_0_1px_rgba(33,164,255,0.35),0_20px_50px_-20px_rgba(33,164,255,0.35)]' : 'border-line'}`}>
      <div className="flex items-center justify-between">
        <div className="text-[15px] font-semibold text-ink">{name}</div>
        {featured ? (
          <span className="rounded-full bg-accent px-2.5 py-0.5 text-[11px] font-bold text-[#08121a]">TRACK · DEPLOYED</span>
        ) : (
          <span className="rounded border border-white/15 bg-black/30 px-2 py-0.5 font-mono text-[10px] text-muted">GEAR {gear}</span>
        )}
      </div>
      <p className="mt-1 min-h-10 text-[13px] text-ink-2">{detail}</p>
      <div className="my-3 flex justify-center">
        <Speedometer
          value={t1}
          label={`${name} Top-1`}
          sublabel="Unseen test split"
          color={dialColor}
          gear={gear}
          badge="DYNO · TOP-1"
          size={172}
        />
      </div>
      <dl className="mt-4 grid grid-cols-3 gap-2 border-t border-grid pt-4 text-xs">
        <div><dt className="text-muted">Top-5</dt><dd className="mt-0.5 font-semibold tabular text-ink">{pct(r?.top5)}</dd></div>
        <div><dt className="text-muted">Make</dt><dd className="mt-0.5 font-semibold tabular text-ink">{pct(r?.make_acc)}</dd></div>
        <div><dt className="text-muted">0–Pred</dt><dd className="mt-0.5 font-semibold tabular text-good-ink">{ms(r?.ms_per_image)}</dd></div>
      </dl>
    </Reveal>
  )
}

function Decision({ status, rule, text }: { status: 'confident' | 'uncertain_model' | 'low_confidence'; rule: string; text: string }) {
  return (
    <div className="flex flex-col gap-2 rounded-2xl border border-line bg-page p-4 sm:flex-row sm:items-start sm:gap-4">
      <div className="shrink-0 sm:w-48"><StatusBadge status={status} /></div>
      <div className="min-w-0">
        <div className="font-mono text-xs text-muted">{rule}</div>
        <div className="mt-0.5 text-sm text-ink-2">{text}</div>
      </div>
    </div>
  )
}

function MiniDecision({ status, rule }: { status: 'confident' | 'uncertain_model' | 'low_confidence'; rule: string }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-white/10 bg-black/20 px-3.5 py-3">
      <StatusBadge status={status} />
      <span className="font-mono text-[11px] text-[#a7b1b7]">{rule}</span>
    </div>
  )
}

/** Segmented meter in the style of the reference site's fuel gauge; fills in when scrolled into view. */
function Gauge({ label, value, text, green }: { label: string; value: number; text: string; green?: boolean }) {
  const SEG = 14
  const filled = Math.round(Math.max(0, Math.min(1, value)) * SEG)
  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
      <div className={`seg-meter ${green ? 'is-green' : ''}`} aria-hidden>
        {Array.from({ length: SEG }, (_, i) => <i key={i} data-on={i < filled} style={{ '--i': i } as CSSProperties} />)}
      </div>
      <div className="mt-3 text-[11px] text-[#a7b1b7]">{label}</div>
      <div className="mt-1 text-[26px] font-semibold leading-none tracking-[-0.04em] tabular">{text}</div>
    </div>
  )
}

export function Footer({ go, wide }: { go: Go; wide?: boolean }) {
  return (
    <footer className="border-t border-line bg-abyss text-[#b9bbc6]">
      <div className={`mx-auto grid gap-8 px-4 py-12 text-sm sm:px-6 md:grid-cols-[1.4fr_1fr_1fr] ${wide ? 'max-w-7xl' : 'max-w-[1320px]'}`}>
        <div>
          <div className="flex items-center gap-2.5 font-semibold text-white"><LogoMark className="h-8 w-8" /> Fine-Grained Vehicle Recognition</div>
          <p className="mt-3 max-w-sm text-[13px] leading-relaxed text-[#b9bbc6]">
            P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System. Trained on Stanford Cars (16,185 images, 196 classes). The dataset has no
            number plates or video, so licence-plate reading and tracking are out of scope.
          </p>
        </div>
        <div>
          <div className="mb-3 text-[14px] font-semibold text-white">Tools</div>
          <ul className="space-y-2 text-[#b9bbc6]">
            {([['recognize', 'Recognize & live camera', IconScan], ['batch', 'Batch analysis', IconGrid], ['stress', 'Stress lab', IconFlask], ['pipeline', 'Pipeline', IconLayers], ['catalogue', 'Coverage catalogue', IconSearch]] as const).map(([id, l, I]) => (
              <li key={id}><button onClick={() => go(id)} className="inline-flex items-center gap-2 transition hover:text-white"><I className="h-4 w-4" />{l}</button></li>
            ))}
          </ul>
        </div>
        <div>
          <div className="mb-3 text-[14px] font-semibold text-white">Evaluation</div>
          <ul className="space-y-2 text-[#b9bbc6]">
            {([['allcars', 'All-cars model', IconSparkles], ['comparison', 'Model comparison', IconChart], ['robustness', 'Robustness & confidence', IconShield], ['failures', 'Failure analysis', IconEye], ['about', 'About the system', IconCpu]] as const).map(([id, l, I]) => (
              <li key={id}><button onClick={() => go(id)} className="inline-flex items-center gap-2 transition hover:text-white"><I className="h-4 w-4" />{l}</button></li>
            ))}
          </ul>
        </div>
      </div>
      <div className="border-t border-white/10 py-5 text-center text-[12px] text-[#8b8d98]">YOLOv8 · ConvNeXt-Tiny · OpenCV · scikit-image · FastAPI · React</div>
    </footer>
  )
}
