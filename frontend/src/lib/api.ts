export const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? ''

/** `allcars` = ~1,300 generations from 1948 to 2026 (default); `stanford` = the 196-class Stanford Cars benchmark model. */
export type ModelId = 'allcars' | 'stanford'
export const MODEL_NAMES: Record<ModelId, string> = { allcars: 'All cars 1948 to 2026', stanford: 'Stanford Cars benchmark' }
let currentModel: ModelId = 'allcars'
/** Every prediction uses the model picked in the header unless a call names one. */
export function setCurrentModel(m: ModelId) { currentModel = m }

export interface ModelInfo {
  id: ModelId
  label: string
  loaded: boolean
  engines: string[]
  classes?: number
  gradcam?: boolean
  error?: string | null
}

export type Status = 'confident' | 'uncertain_model' | 'low_confidence'
export type DetSource = 'yolo' | 'yolo_low_confidence' | 'classical_roi_fallback' | 'full_image_fallback'

export interface Vehicle {
  label: string
  make: string
  model: string
  year: string
  class_id: number
  confidence: number
  status: Status
  message: string
  make_guess: string
  make_confidence: number
  top5: { label: string; confidence: number }[]
  box: [number, number, number, number]
  det_conf: number
  det_source: DetSource
}

export interface Prediction {
  filename: string | null
  image_size: [number, number]
  vehicles: Vehicle[]
  annotated_image: string
  gradcam: string | null
  gradcam_note: string | null
  stages: { name: string; image_b64: string }[]
  classical_info: Record<string, unknown>
  timings_ms: Record<string, number>
  backend: string
  model?: ModelId
  device: string
}

export interface Health {
  status: 'ok' | 'error'
  model_loaded: boolean
  model?: ModelId
  default_model?: ModelId
  models?: ModelInfo[]
  device: string | null
  backend: string | null
  engines?: string[]
  gradcam_available: boolean
  error: string | null
  warnings: string[]
  thresholds: { tau?: number; tau_make?: number; det_conf?: number }
  max_upload_mb: number
  /** a vision LLM second opinion is configured on the server */
  llm_available?: boolean
}

export interface LLMOpinion {
  model: string
  answer: { make: string; model: string; generation_or_years: string; body_type: string; confidence: string; clues: string } | null
  text: string | null
  latency_ms: number
}

export interface ClassMeta {
  id: number
  name: string
  make: string
  model: string
  year: string
  // all-cars model only
  base_model?: string
  generation?: string
  start_year?: number
  source?: string
  n_images?: number
  n_train?: number
  n_test?: number
}

/** One analysed image: what the Recognize page shows and the history strip keeps. */
export interface Analysis {
  id: string
  at: number
  name: string
  preview: string
  result: Prediction
  groundTruth: string | null
  source: 'upload' | 'sample' | 'camera'
  /** a live-camera frame (never triggers automatic LLM calls) */
  live?: boolean
}

export interface Sample {
  name: string
  url: string
  ground_truth: string | null
  /** which model's label set `ground_truth` uses */
  model?: ModelId
  credit?: { author?: string; licence?: string; source?: string } | null
}

/** metrics.json of the all-cars model (written by modal_train/train_modern.py). */
export interface CarsMetrics {
  dataset?: {
    name?: string; hf_repo?: string; images_used?: number; images_by_source?: Record<string, number>; classes?: number; makes?: number
    models?: number; min_photos_per_class?: number; split?: { train: number; val: number; test: number }; split_rule?: string
    years?: { first?: number; last_generation_start?: number }
  }
  training?: {
    gpu?: string; device?: string; precision?: string; epochs_planned?: number; epochs_run?: number; stopped_early?: string | null
    batch?: number; lr?: number; img_size?: number; train_minutes?: number; total_minutes?: number; cost_usd_this_run?: number
    cost_usd_total_estimate?: number
    history?: { epoch: number; train_loss: number; val_top1_raw: number; val_top1_ema: number; train_sec: number; img_per_s: number }[]
  }
  test?: { n: number; top1: number; top5: number; macro_f1: number; balanced_acc: number; make_acc: number; model_acc_any_generation: number }
  by_era?: { era: string; n: number; top1: number; top5: number; make_acc: number; model_acc_any_generation: number }[]
  by_source?: Record<string, { n: number; top1: number; make_acc: number }>
  calibration?: { T: number; ece_before: number; ece_after: number; nll_before: number; nll_after: number }
  confidence?: { tau: number; tau_make: number; target_precision?: number; decision_breakdown?: Record<string, Row>; threshold_table?: Row[] }
  ablation?: { full_pipeline?: number; without_flip_tta?: number; without_illumination_fix?: number; best_weights?: string; val_top1_best?: number }
  optimisation?: { variant: string; size_mb: number; ms: number; top1: number; agreement_with_fp32?: number }[]
  top_confusions?: { true: string; pred: string; count: number; same_model: boolean }[]
  per_class?: { id: number; n_train: number; n_test: number; acc: number | null }[]
  gpu_benchmark?: { gpu: string; device: string; img_per_s: number; usd_per_hour_with_8cpu_24gb: number; usd_per_1M_train_images: number }[]
}

export interface FailureCase { file: string; url: string; true: string; pred: string; confidence: number; author?: string; licence?: string; source?: string }

// metrics.json is produced by the notebook; every key is optional.
export type Row = Record<string, number | string | null | undefined>
export interface Metrics {
  classical?: Record<string, unknown>
  detection?: Record<string, Row>
  detection_ultralytics_test?: Row
  classifier_training?: { history?: Row[]; best_val_top1?: number }
  calibration_params?: { T?: number; alpha?: number; T2?: number }
  ablation?: Record<string, Row>
  approach_comparison?: Record<string, Row>
  robustness?: Record<string, Row>
  slices?: Record<string, Record<string, Row>>
  ece?: Record<string, number>
  confidence?: {
    tau?: number
    tau_make?: number
    target_precision?: number
    decision_breakdown?: Record<string, Row>
    /** dict keyed by threshold (training notebook) or list of records (evaluation notebook) — see lib/metrics.ts */
    threshold_table?: unknown
  }
  optimization?: Record<string, Row>
  failure_causes?: Record<string, number>
  failure_examples?: { file?: string; true?: string; pred?: string; conf?: number; cause?: string; explanation?: string }[]
  end_to_end_ms_per_image?: number
  end_to_end_ms_onnx_int8?: number
  final_system_test?: Row
  similar_class_analysis?: {
    error_hierarchy?: Record<string, number>
    multi_year_groups?: number
    year_acc_given_make_model?: number
    top_confused_pairs?: { true?: string; predicted?: string; count?: number; relation?: string }[]
    hardest_classes?: { class?: string; accuracy?: number }[]
  }
  requirements_summary?: Record<string, string>
}

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(API_BASE + path)
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
  return r.json() as Promise<T>
}

export const api = {
  /** Top-level fields (engines, thresholds, Grad-CAM) describe `model`; `models` lists every model on the server. */
  health: (model: ModelId) => getJSON<Health>(`/api/health?model=${model}`),
  /** The evaluation pages describe the Stanford Cars benchmark model. */
  metrics: () => getJSON<Metrics>('/api/metrics?model=stanford'),
  carsMetrics: () => getJSON<CarsMetrics>('/api/metrics?model=allcars'),
  failures: () => getJSON<FailureCase[]>('/api/failures'),
  plots: () => getJSON<string[]>('/api/plots'),
  samples: () => getJSON<Sample[]>('/api/samples'),
  classes: (model: ModelId) => getJSON<ClassMeta[]>(`/api/classes?model=${model}`),
  failureUrl: (f: FailureCase) => API_BASE + f.url,

  /** Second opinion from a vision LLM on one vehicle crop (not part of the evaluated pipeline). */
  async llmOpinion(crop: Blob): Promise<LLMOpinion> {
    const fd = new FormData()
    fd.append('file', crop, 'vehicle.jpg')
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), 60_000)
    try {
      const r = await fetch(`${API_BASE}/api/llm-opinion`, { method: 'POST', body: fd, signal: ctrl.signal })
      const j = await r.json().catch(() => null)
      if (!r.ok) throw new Error(typeof j?.detail === 'string' ? j.detail : `HTTP ${r.status}`)
      return j as LLMOpinion
    } catch (e) {
      if (ctrl.signal.aborted) throw new Error('The LLM did not answer within 60 s. Please try again.')
      throw e
    } finally {
      clearTimeout(timer)
    }
  },
  /** `w` = display width: the server returns a resized WebP (the full PNG is used for the zoom view). */
  plotUrl: (name: string, w?: number) => `${API_BASE}/api/plots/${encodeURIComponent(name)}${w ? `?w=${w}` : ''}`,
  sampleUrl: (s: Sample) => API_BASE + s.url,

  async predict(file: Blob, filename: string, opts: { gradcam: boolean; stages: boolean; maxVehicles: number; engine?: 'torch' | 'onnx'; timeoutMs?: number; model?: ModelId }) {
    const fd = new FormData()
    fd.append('file', file, filename)
    const q = new URLSearchParams({
      gradcam: String(opts.gradcam),
      stages: String(opts.stages),
      max_vehicles: String(opts.maxVehicles),
      model: opts.model ?? currentModel,
      ...(opts.engine ? { engine: opts.engine } : {}),
    })
    // Never wait forever: a stalled upload/download on a slow long-distance link must surface as a retryable error.
    const timeoutMs = opts.timeoutMs ?? 90_000
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), timeoutMs)
    const timedOut = () => new Error(`No response from the server within ${Math.round(timeoutMs / 1000)} s. The network may be slow; please try again.`)
    try {
      let r: Response
      try {
        r = await fetch(`${API_BASE}/api/predict?${q}`, { method: 'POST', body: fd, signal: ctrl.signal })
      } catch {
        if (ctrl.signal.aborted) throw timedOut()
        throw new Error(API_BASE || location.hostname === 'localhost' || location.hostname === '127.0.0.1'
          ? 'Cannot reach the backend. Is `uvicorn backend.app:app` running on port 8000?'
          : 'Cannot reach the server. Check your connection and try again.')
      }
      if (!r.ok) {
        let detail = `HTTP ${r.status}`
        try {
          const j = await r.json()
          detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
        } catch {
          /* not JSON */
        }
        throw new Error(detail)
      }
      try {
        return (await r.json()) as Prediction
      } catch {
        if (ctrl.signal.aborted) throw timedOut()
        throw new Error('The server response was incomplete. Please try again.')
      }
    } finally {
      clearTimeout(timer)
    }
  },
}
