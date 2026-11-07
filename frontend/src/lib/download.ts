import type { Prediction } from './api'

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export const downloadJSON = (data: unknown, filename: string) =>
  downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), filename)

const csvCell = (v: unknown) => {
  const s = v === null || v === undefined ? '' : String(v)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

export function downloadCSV(rows: Record<string, unknown>[], filename: string) {
  if (!rows.length) return
  const cols = Object.keys(rows[0])
  const text = [cols.join(','), ...rows.map((r) => cols.map((c) => csvCell(r[c])).join(','))].join('\n')
  downloadBlob(new Blob([text], { type: 'text/csv' }), filename)
}

/** Prediction without the (large) base64 images: what goes into JSON exports. */
export const slimPrediction = (p: Prediction) => ({
  filename: p.filename,
  image_size: p.image_size,
  vehicles: p.vehicles,
  classical_info: p.classical_info,
  timings_ms: p.timings_ms,
  backend: p.backend,
  device: p.device,
})

export const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')

/** Unique id that also works on non-secure (LAN http) pages, where crypto.randomUUID is unavailable. */
export const uid = () => (typeof crypto !== 'undefined' && 'randomUUID' in crypto && window.isSecureContext ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`)
