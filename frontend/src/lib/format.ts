const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

export const pct = (v: unknown, digits = 1) => (isNum(v) ? `${(v * 100).toFixed(digits)}%` : '—')
export const num = (v: unknown, digits = 2) => (isNum(v) ? v.toFixed(digits) : '—')
export const ms = (v: unknown) => (isNum(v) ? (v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)} ms`) : '—')
export const mb = (v: unknown) => (isNum(v) ? `${v.toFixed(1)} MB` : '—')

/** "mean_IoU" → "Mean IoU", "loc_acc_IoU50" → "Loc acc IoU50" */
export const humanize = (k: string) => {
  const s = k.replace(/_/g, ' ').replace(/\s+/g, ' ').trim()
  return s.charAt(0).toUpperCase() + s.slice(1)
}

/** Format a value by its key name: accuracies/rates → %, latency → ms, size → MB. */
export const auto = (key: string, v: unknown): string => {
  if (v === null || v === undefined) return '—'
  if (!isNum(v)) return String(v)
  const k = key.toLowerCase()
  if (k.includes('ms')) return ms(v)
  if (k.includes('size') || k.endsWith('_mb')) return mb(v)
  if (k === 'n' || k.includes('count') || k.startsWith('num')) return v.toLocaleString()
  if (v >= 0 && v <= 1) return pct(v)
  return num(v)
}
