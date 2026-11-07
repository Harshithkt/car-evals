import { useState } from 'react'
import { api } from '../lib/api'
import { Card, Missing } from './ui'

/** A training/evaluation PNG from ./plots. `available` is the list from /api/plots (null = still loading). */
export function Plot({ name, title, caption, available }: {
  name: string
  title: string
  caption?: string
  available: string[] | null
}) {
  const [zoom, setZoom] = useState(false)
  const [broken, setBroken] = useState(false)
  const exists = available?.includes(name) && !broken
  return (
    <Card title={title} subtitle={caption}>
      {available === null ? (
        <div className="h-40 animate-pulse rounded-lg bg-surface-2" />
      ) : exists ? (
        <>
          <button type="button" onClick={() => setZoom(true)} className="block w-full cursor-zoom-in rounded-lg bg-white p-1">
            <img src={api.plotUrl(name, 1400)} alt={title} loading="lazy" className="mx-auto max-h-[480px] w-auto max-w-full" onError={() => setBroken(true)} />
          </button>
          {zoom && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4" onClick={() => setZoom(false)}>
              <img src={api.plotUrl(name)} alt={title} className="max-h-full max-w-full rounded-lg bg-white" />
            </div>
          )}
        </>
      ) : (
        <Missing what={`plots/${name}.png`} hint="This plot is written by a notebook cell that has not been run yet." />
      )}
    </Card>
  )
}
