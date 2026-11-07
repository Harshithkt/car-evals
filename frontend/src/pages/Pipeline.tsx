import { useState } from 'react'
import type { Prediction } from '../lib/api'
import { auto, humanize } from '../lib/format'
import { Plot } from '../components/Plot'
import { Card, PageHeader } from '../components/ui'

const STAGE_NOTES: [RegExp, string][] = [
  [/^Original/, 'Input image, resized for display.'],
  [/^Illumination/, 'Adaptive correction: gamma + CLAHE on the LAB L-channel when the image is too dark or washed out.'],
  [/^Z-score/, 'Per-channel standardisation (zero mean, unit variance), rescaled for display.'],
  [/^Grayscale/, 'Grayscale conversion and Gaussian blur to suppress noise before edge detection.'],
  [/^Canny/, 'Canny edges with automatic (median-based) thresholds.'],
  [/^Morphological/, 'Morphological closing joins broken edge fragments into solid regions.'],
  [/^Contours/, 'Detected contours (green), classical ROI (blue) and the YOLO detection (red).'],
  [/^Region/, 'The vehicle crop that goes to the classifier.'],
  [/^HOG/, 'Histogram of oriented gradients: shape / silhouette descriptor.'],
  [/^LBP/, 'Local binary patterns: texture descriptor (grille, wheels, lights).'],
  [/^Hue/, 'Hue channel at full saturation: the colour cue used in the classical features.'],
]

const INFO_LABELS: Record<string, string> = {
  roi_aspect_ratio: 'ROI aspect ratio',
  lr_symmetry: 'Left–right symmetry',
  mean_rgb: 'Mean RGB',
  num_contours: 'Contours',
  classical_roi_score: 'Classical ROI score',
}

const INFO_NOTES: Record<string, string> = {
  brightness: 'Mean gray level (0–255) of the image.',
  illumination_fix: 'Which illumination correction was applied.',
  roi_aspect_ratio: 'Width / height of the vehicle region.',
  edge_density: 'Share of edge pixels inside the ROI.',
  num_contours: 'Number of external contours found.',
  contour_solidity: 'Area of the largest contour / area of its convex hull.',
  lr_symmetry: 'Left–right mirror similarity of the crop (1 = perfectly symmetric).',
  mean_rgb: 'Average colour of the centre of the crop.',
  classical_roi_score: 'Confidence of the classical contour-based ROI.',
}

function fmtInfo(k: string, v: unknown) {
  if (k === 'mean_rgb' && Array.isArray(v)) {
    const [r, g, b] = v as number[]
    return (
      <span className="inline-flex items-center gap-2">
        <span className="inline-block h-3.5 w-3.5 rounded-sm border border-line" style={{ background: `rgb(${r},${g},${b})` }} />
        <span className="tabular">{v.join(', ')}</span>
      </span>
    )
  }
  if (['brightness', 'roi_aspect_ratio', 'lr_symmetry', 'contour_solidity', 'classical_roi_score', 'edge_density'].includes(k) && typeof v === 'number')
    return <span className="tabular">{v.toLocaleString(undefined, { maximumFractionDigits: 3 })}</span>
  return <span className="tabular">{auto(k, v)}</span>
}

export default function Pipeline({ result, plots, onGoRecognize }: {
  result: Prediction | null
  plots: string[] | null
  onGoRecognize: () => void
}) {
  const [open, setOpen] = useState<number | null>(null)
  const stages = result?.stages ?? []
  return (
    <div>
      <PageHeader eyebrow="Pipeline" title="Intermediate image-processing results">
        Every classical computer-vision stage for the most recent image, as computed by <code className="text-ink">classical_cv.processing_stages()</code>.
        These measurements feed the classical classifier, which is late-fused with ConvNeXt.
      </PageHeader>

      {!result ? (
        <Card>
          <div className="py-10 text-center text-sm text-ink-2">
            No image analysed yet.{' '}
            <button onClick={onGoRecognize} className="text-accent hover:underline">Upload one on the Recognize tab</button>, then come back here.
          </div>
        </Card>
      ) : stages.length === 0 ? (
        <Card><div className="py-6 text-sm text-ink-2">The last prediction was run without stages.</div></Card>
      ) : (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_20rem]">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {stages.map((s, i) => (
              <figure key={s.name} className="overflow-hidden rounded-xl border border-line bg-surface">
                <button type="button" onClick={() => setOpen(i)} className="block w-full cursor-zoom-in bg-surface-2">
                  <img src={s.image_b64} alt={s.name} className="mx-auto aspect-[4/3] w-full object-contain" />
                </button>
                <figcaption className="px-3 py-2">
                  <div className="text-[13px] font-medium text-ink"><span className="tabular text-muted">{i + 1}.</span> {s.name}</div>
                  <div className="text-xs text-ink-2">{STAGE_NOTES.find(([r]) => r.test(s.name))?.[1]}</div>
                </figcaption>
              </figure>
            ))}
          </div>

          <Card title="Classical measurements" subtitle={result.filename ?? undefined} className="self-start xl:sticky xl:top-28">
            <table className="w-full text-sm">
              <tbody>
                {Object.entries(result.classical_info).map(([k, v]) => (
                  <tr key={k} className="border-b border-grid last:border-0 align-top">
                    <td className="py-2 pr-2">
                      <div className="text-ink">{INFO_LABELS[k] ?? humanize(k)}</div>
                      <div className="text-[11px] text-muted">{INFO_NOTES[k]}</div>
                    </td>
                    <td className="py-2 text-right text-ink-2 whitespace-nowrap">{fmtInfo(k, v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {open !== null && stages[open] && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-3 bg-black/80 p-4" onClick={() => setOpen(null)}>
          <img src={stages[open].image_b64} alt={stages[open].name} className="max-h-[85vh] max-w-full rounded-lg" />
          <div className="text-sm text-white">{stages[open].name}</div>
        </div>
      )}

      <div className="mt-6">
        <Plot name="classical_stages" title="Classical stages on training samples (from the notebook)" available={plots} />
      </div>
      {plots?.includes('pipeline_stages') && (
        <div className="mt-6">
          <Plot name="pipeline_stages" title="Stages produced by the deployed pipeline (evaluation notebook)" available={plots} />
        </div>
      )}
    </div>
  )
}
