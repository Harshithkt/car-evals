/** In-browser image perturbations for the Stress lab. Deterministic (seeded noise) so runs are repeatable. */

export type Kind = 'dark' | 'night' | 'overexpose' | 'fog' | 'contrast' | 'blur' | 'lowres' | 'noise' | 'occlusion' | 'rotate'

export const KINDS: { id: Kind; label: string; challenge: 'Illumination' | 'Weather' | 'Resolution' | 'Occlusion' | 'Viewpoint' | 'Sensor'; desc: string }[] = [
  { id: 'dark', label: 'Under-exposure', challenge: 'Illumination', desc: 'Scales brightness down. 100% keeps about 12% of the light.' },
  { id: 'night', label: 'Night drive', challenge: 'Illumination', desc: 'Darkens, adds a blue cast and sensor noise.' },
  { id: 'overexpose', label: 'Over-exposure', challenge: 'Illumination', desc: 'Washes highlights out toward white.' },
  { id: 'contrast', label: 'Low contrast', challenge: 'Illumination', desc: 'Compresses the tonal range toward mid-grey.' },
  { id: 'fog', label: 'Fog / haze', challenge: 'Weather', desc: 'Blends the scene with a bright grey haze.' },
  { id: 'blur', label: 'Blur', challenge: 'Resolution', desc: 'Out-of-focus or motion blur.' },
  { id: 'lowres', label: 'CCTV low-res', challenge: 'Resolution', desc: 'Downsamples to as little as 6% of the original size, then upscales blocky.' },
  { id: 'noise', label: 'Sensor noise', challenge: 'Sensor', desc: 'Adds Gaussian noise (σ up to 60).' },
  { id: 'occlusion', label: 'Occlusion', challenge: 'Occlusion', desc: 'A post or sign covers up to 65% of the detected car.' },
  { id: 'rotate', label: 'Camera tilt', challenge: 'Viewpoint', desc: 'Rotates the frame by up to 35°.' },
]

export const PRESETS: { label: string; kind: Kind; severity: number }[] = [
  { label: 'Night drive', kind: 'night', severity: 0.7 },
  { label: 'Dense fog', kind: 'fog', severity: 0.7 },
  { label: 'CCTV', kind: 'lowres', severity: 0.8 },
  { label: 'Behind a pole', kind: 'occlusion', severity: 0.5 },
  { label: 'Tilted dash-cam', kind: 'rotate', severity: 0.5 },
  { label: 'Blown highlights', kind: 'overexpose', severity: 0.7 },
]

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Could not load the image'))
    img.src = src
  })
}

const canvas = (w: number, h: number) => {
  const c = document.createElement('canvas')
  c.width = Math.max(1, Math.round(w))
  c.height = Math.max(1, Math.round(h))
  return c
}

/** Working copy of the source image, at most `maxSide` px, so every stress test runs on the same pixels. */
export function baseCanvas(img: HTMLImageElement, maxSide = 1024): HTMLCanvasElement {
  const s = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight))
  const c = canvas(img.naturalWidth * s, img.naturalHeight * s)
  const ctx = c.getContext('2d')!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(img, 0, 0, c.width, c.height)
  return c
}

function mulberry32(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(rand: () => number) {
  const u = Math.max(1e-9, rand())
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())
}

function pixelOp(src: HTMLCanvasElement, fn: (d: Uint8ClampedArray, rand: () => number) => void) {
  const c = canvas(src.width, src.height)
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(src, 0, 0)
  const data = ctx.getImageData(0, 0, c.width, c.height)
  fn(data.data, mulberry32(42))
  ctx.putImageData(data, 0, 0)
  return c
}

function resample(src: HTMLCanvasElement, scale: number, pixelated: boolean) {
  const small = canvas(src.width * scale, src.height * scale)
  const sctx = small.getContext('2d')!
  sctx.imageSmoothingQuality = 'high'
  sctx.drawImage(src, 0, 0, small.width, small.height)
  const out = canvas(src.width, src.height)
  const octx = out.getContext('2d')!
  octx.imageSmoothingEnabled = !pixelated
  octx.imageSmoothingQuality = 'high'
  octx.drawImage(small, 0, 0, out.width, out.height)
  return out
}

/**
 * Apply one perturbation at severity `s` ∈ [0, 1] (0 = unchanged).
 * `box` (x1,y1,x2,y2 in working-canvas pixels) is the detected car, used to place the occluder.
 */
export function perturb(src: HTMLCanvasElement, kind: Kind, s: number, box?: number[] | null): HTMLCanvasElement {
  s = Math.max(0, Math.min(1, s))
  if (s === 0) return pixelOp(src, () => {})
  switch (kind) {
    case 'dark': {
      const f = 1 - 0.88 * s
      return pixelOp(src, (d) => { for (let i = 0; i < d.length; i += 4) { d[i] *= f; d[i + 1] *= f; d[i + 2] *= f } })
    }
    case 'night': {
      const f = 1 - 0.85 * s
      const sigma = 18 * s
      return pixelOp(src, (d, r) => {
        for (let i = 0; i < d.length; i += 4) {
          const n = gaussian(r) * sigma
          d[i] = d[i] * f * 0.78 + n
          d[i + 1] = d[i + 1] * f * 0.9 + n
          d[i + 2] = d[i + 2] * f * 1.15 + n
        }
      })
    }
    case 'overexpose': {
      const k = 1 - 0.9 * s
      return pixelOp(src, (d) => { for (let i = 0; i < d.length; i += 4) for (let j = 0; j < 3; j++) d[i + j] = 255 - (255 - d[i + j]) * k })
    }
    case 'contrast': {
      const k = 1 - 0.88 * s
      return pixelOp(src, (d) => { for (let i = 0; i < d.length; i += 4) for (let j = 0; j < 3; j++) d[i + j] = (d[i + j] - 128) * k + 128 })
    }
    case 'fog': {
      const a = 0.8 * s
      return pixelOp(src, (d) => { for (let i = 0; i < d.length; i += 4) for (let j = 0; j < 3; j++) d[i + j] = d[i + j] * (1 - a) + 222 * a })
    }
    case 'noise': {
      const sigma = 60 * s
      return pixelOp(src, (d, r) => { for (let i = 0; i < d.length; i += 4) for (let j = 0; j < 3; j++) d[i + j] += gaussian(r) * sigma })
    }
    case 'blur':
      return resample(src, 1 / (1 + 15 * s), false)
    case 'lowres':
      return resample(src, Math.max(0.06, 1 - 0.94 * s), true)
    case 'occlusion': {
      const out = pixelOp(src, () => {})
      const ctx = out.getContext('2d')!
      const [x1, y1, x2, y2] = box ?? [out.width * 0.15, out.height * 0.15, out.width * 0.85, out.height * 0.85]
      const bw = x2 - x1
      const bh = y2 - y1
      const w = bw * 0.65 * s
      const cx = x1 + bw * 0.5
      ctx.fillStyle = '#2a2a2a'
      ctx.fillRect(cx - w / 2, y1 - bh * 0.1, w, bh * 1.2)
      ctx.fillStyle = 'rgba(255,255,255,0.08)'
      for (let x = cx - w / 2; x < cx + w / 2; x += 14) ctx.fillRect(x, y1 - bh * 0.1, 4, bh * 1.2)
      return out
    }
    case 'rotate': {
      const out = canvas(src.width, src.height)
      const ctx = out.getContext('2d')!
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, out.width, out.height)
      ctx.translate(out.width / 2, out.height / 2)
      ctx.rotate((35 * s * Math.PI) / 180)
      ctx.drawImage(src, -src.width / 2, -src.height / 2)
      return out
    }
  }
}

export const toBlob = (c: HTMLCanvasElement, quality = 0.92) =>
  new Promise<Blob>((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error('Encoding failed'))), 'image/jpeg', quality))
