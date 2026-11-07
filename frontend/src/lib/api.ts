export const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? ''

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
  device: string
}

export interface Health {
  status: 'ok' | 'error'
  model_loaded: boolean
  device: string | null
  backend: string | null
  engines?: string[]
  gradcam_available: boolean
  error: string | null
  warnings: string[]
  thresholds: { tau?: number; tau_make?: number; det_conf?: number }
  max_upload_mb: number
}

export interface ClassMeta {
  id: number
  name: string
  make: string
  model: string
  year: string
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
}

export interface Sample {
  name: string
  url: string
  ground_truth: string | null
}

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
  health: () => getJSON<Health>('/api/health'),
  metrics: () => getJSON<Metrics>('/api/metrics'),
  plots: () => getJSON<string[]>('/api/plots'),
  samples: () => getJSON<Sample[]>('/api/samples'),
  classes: () => getJSON<ClassMeta[]>('/api/classes'),
  /** `w` = display width: the server returns a resized WebP (the full PNG is used for the zoom view). */
  plotUrl: (name: string, w?: number) => `${API_BASE}/api/plots/${encodeURIComponent(name)}${w ? `?w=${w}` : ''}`,
  sampleUrl: (s: Sample) => API_BASE + s.url,

  async predict(file: Blob, filename: string, opts: { gradcam: boolean; stages: boolean; maxVehicles: number; engine?: 'torch' | 'onnx'; timeoutMs?: number }) {
    const fd = new FormData()
    fd.append('file', file, filename)
    const q = new URLSearchParams({
      gradcam: String(opts.gradcam),
      stages: String(opts.stages),
      max_vehicles: String(opts.maxVehicles),
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
