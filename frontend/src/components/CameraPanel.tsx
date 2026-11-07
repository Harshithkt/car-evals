import { useCallback, useEffect, useRef, useState } from 'react'
import type { Prediction } from '../lib/api'
import { pct } from '../lib/format'
import { IconCamera, IconPlay, IconStop } from './icons'
import { Card, STATUS_COLOR, Toggle } from './ui'

/**
 * Webcam capture. Frames are analysed one at a time (CPU inference takes ~1 s), and the latest detections are
 * drawn over the live video. getUserMedia needs a secure context: localhost or HTTPS.
 */
export function CameraPanel({ busy, result, onCapture }: {
  busy: boolean
  result: Prediction | null
  onCapture: (blob: Blob, live: boolean) => Promise<void>
}) {
  const video = useRef<HTMLVideoElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const [live, setLive] = useState(false)
  const [interval, setIntervalSec] = useState(2)
  const [lastAt, setLastAt] = useState<number | null>(null)
  const [, tick] = useState(0)

  const stop = useCallback(() => {
    setLive(false)
    setStream((s) => { s?.getTracks().forEach((t) => t.stop()); return null })
  }, [])

  async function start(id = deviceId) {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access needs a secure page: open the app on http://localhost or over HTTPS.')
      return
    }
    try {
      stream?.getTracks().forEach((t) => t.stop())
      const s = await navigator.mediaDevices.getUserMedia({
        video: id ? { deviceId: { exact: id }, width: { ideal: 1280 }, height: { ideal: 720 } } : { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      setStream(s)
      const all = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput')
      setDevices(all)
      setDeviceId(s.getVideoTracks()[0]?.getSettings().deviceId ?? id)
    } catch (e) {
      const name = e instanceof DOMException ? e.name : ''
      setError(name === 'NotAllowedError' ? 'Camera permission was denied. Allow it in the browser address bar and try again.'
        : name === 'NotFoundError' ? 'No camera was found on this device.'
        : `Could not start the camera (${name || String(e)}).`)
    }
  }

  useEffect(() => {
    if (video.current) video.current.srcObject = stream
  }, [stream])

  useEffect(() => () => stop(), [stop]) // release the camera when leaving the page

  const capture = useCallback(async (isLive: boolean) => {
    const v = video.current
    if (!v || !v.videoWidth) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    const blob = await new Promise<Blob | null>((r) => c.toBlob(r, 'image/jpeg', 0.9))
    if (!blob) return
    setLastAt(Date.now())
    await onCapture(blob, isLive)
  }, [onCapture])

  // live mode: analyse the next frame `interval` seconds after the previous result arrived
  useEffect(() => {
    if (!live || !stream || busy) return
    const t = setTimeout(() => capture(true), interval * 1000)
    return () => clearTimeout(t)
  }, [live, stream, busy, interval, capture, result])

  useEffect(() => {
    if (!lastAt) return
    const t = setInterval(() => tick((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [lastAt])

  const [W, H] = result?.image_size ?? [1, 1]
  const sameFrame = stream && result && video.current && Math.abs(video.current.videoWidth - W) < 2

  return (
    <Card
      title={<span className="inline-flex items-center gap-2"><IconCamera className="h-4 w-4 text-ink" /> Live camera</span>}
      subtitle="Point a webcam at a car, or at a photo of one on another screen."
      actions={stream && (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-critical/10 px-2.5 py-1 text-xs font-medium text-critical-ink">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-critical" /> {live ? 'LIVE' : 'Camera on'}
        </span>
      )}
    >
      <div className="relative overflow-hidden rounded-xl bg-[#0a0f1a]">
        <video ref={video} autoPlay playsInline muted className={`block h-auto w-full ${stream ? '' : 'hidden'}`} />
        {stream && sameFrame && (
          <div className="pointer-events-none absolute inset-0">
            {result!.vehicles.map((v, i) => {
              const [x1, y1, x2, y2] = v.box
              const col = STATUS_COLOR[v.status]
              return (
                <div key={i} className="absolute rounded-[3px] border-2 transition-all duration-500" style={{ left: `${(x1 / W) * 100}%`, top: `${(y1 / H) * 100}%`, width: `${((x2 - x1) / W) * 100}%`, height: `${((y2 - y1) / H) * 100}%`, borderColor: col }}>
                  <span className="absolute -top-6 left-[-2px] whitespace-nowrap rounded-t-[3px] px-1.5 py-0.5 text-[11px] font-semibold text-white" style={{ background: col }}>
                    {v.status === 'confident' ? v.label : `${v.make_guess}?`} · {pct(v.status === 'confident' ? v.confidence : v.make_confidence)}
                  </span>
                </div>
              )
            })}
            {busy && <span className="scanline" />}
          </div>
        )}
        {!stream && (
          <div className="flex aspect-video flex-col items-center justify-center gap-3 px-6 text-center text-white/70">
            <IconCamera className="h-10 w-10 text-white/40" />
            <div className="text-sm">The camera is off</div>
            <button onClick={() => start()} className="inline-flex items-center gap-2 rounded-lg bg-white px-4 py-2 text-sm font-medium text-[#0a0f1a] hover:bg-white/90">
              <IconPlay className="h-4 w-4" /> Start camera
            </button>
          </div>
        )}
      </div>

      {error && <p className="mt-3 rounded-lg border border-critical/40 bg-critical/10 px-3 py-2 text-sm text-critical-ink">{error}</p>}

      {stream && (
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button onClick={() => capture(false)} disabled={busy} className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50">
            <IconCamera className="h-4 w-4" /> {busy ? 'Analysing…' : 'Capture & recognise'}
          </button>
          <Toggle checked={live} onChange={setLive} label="Live mode" />
          {live && (
            <label className="inline-flex items-center gap-2 text-sm text-ink-2">
              every
              <select value={interval} onChange={(e) => setIntervalSec(Number(e.target.value))} className="rounded-lg border border-line bg-surface px-2 py-1 text-ink">
                {[0.5, 1, 2, 3, 5].map((s) => <option key={s} value={s}>{s}s</option>)}
              </select>
            </label>
          )}
          {devices.length > 1 && (
            <select value={deviceId} onChange={(e) => { setDeviceId(e.target.value); start(e.target.value) }} className="max-w-48 rounded-lg border border-line bg-surface px-2 py-1 text-sm text-ink">
              {devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
            </select>
          )}
          <button onClick={stop} className="ml-auto inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-2 text-sm text-ink-2 hover:text-ink">
            <IconStop className="h-3.5 w-3.5" /> Stop
          </button>
        </div>
      )}
      {stream && lastAt && (
        <p className="mt-2 text-xs text-muted">
          Last frame analysed {Math.max(0, Math.round((Date.now() - lastAt) / 1000))} s ago. Boxes show the most recent result, so they lag slightly behind motion.
        </p>
      )}
    </Card>
  )
}
