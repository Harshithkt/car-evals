/** Normalise metrics.json shapes written by the two notebooks (training notebook vs P5_Evaluation_Remaining). */

const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null)

export interface ThresholdRow { threshold: string; coverage: number | null; acc: number | null }

/**
 * threshold_table is either `{ "0.1": {coverage, acc} }` (training notebook)
 * or `[{threshold, coverage, accuracy}]` (evaluation notebook).
 */
export function thresholdRows(tt: unknown): ThresholdRow[] {
  if (Array.isArray(tt))
    return tt.map((r) => ({ threshold: Number(r?.threshold ?? 0).toFixed(1), coverage: num(r?.coverage), acc: num(r?.accuracy ?? r?.acc) }))
  if (tt && typeof tt === 'object')
    return Object.entries(tt as Record<string, Record<string, unknown>>).map(([t, r]) => ({ threshold: t, coverage: num(r?.coverage), acc: num(r?.acc ?? r?.accuracy) }))
  return []
}

const CONDITION_LABEL: Record<string, string> = {
  clean: 'Clean',
  dark: 'Dark (×0.3)',
  very_dark: 'Very dark (γ=3)',
  overexposed: 'Over-exposed',
  low_contrast: 'Low contrast',
  occlusion_25: 'Occlusion 25%',
  occlusion_40: 'Occlusion 40%',
  'lowres_0.25': 'Low-res ×0.25',
  'lowres_0.1': 'Low-res ×0.1',
  motion_blur: 'Motion blur',
  gauss_noise: 'Gaussian noise',
  rotate_15: 'Rotated 15°',
}

export const conditionLabel = (c: string) => CONDITION_LABEL[c] ?? c.replace(/_/g, ' ').replace(/^./, (x) => x.toUpperCase())

/** -0.0523 → "−5.2 pts" */
export const signedPts = (v: unknown) => {
  const x = num(v)
  if (x === null) return '—'
  const p = x * 100
  return `${p > 0 ? '+' : p < 0 ? '−' : '±'}${Math.abs(p).toFixed(1)} pts`
}
