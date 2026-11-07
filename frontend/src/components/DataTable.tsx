import type { ReactNode } from 'react'
import { auto, humanize } from '../lib/format'
import type { Row } from '../lib/api'

export interface Column {
  key: string
  label?: string
  format?: (v: unknown, row: Row) => ReactNode
  align?: 'left' | 'right'
}

/** Table for a `{rowName: {col: value}}` object (the shape pandas `to_dict('index')` produces). */
export function DataTable({ data, firstCol = 'Name', columns, highlight, renderName }: {
  data: Record<string, Row>
  firstCol?: string
  columns?: Column[]
  highlight?: (name: string) => boolean
  renderName?: (name: string) => ReactNode
}) {
  const names = Object.keys(data)
  const cols: Column[] = columns ?? Array.from(new Set(names.flatMap((n) => Object.keys(data[n] ?? {})))).map((key) => ({ key }))
  return (
    <div className="overflow-x-auto -mx-1">
      <table className="w-full min-w-max text-sm">
        <thead>
          <tr className="border-b border-grid text-left text-[12px] text-muted">
            <th className="px-2 py-2 font-medium">{firstCol}</th>
            {cols.map((c) => (
              <th key={c.key} className={`px-2 py-2 font-medium ${c.align === 'left' ? 'text-left' : 'text-right'}`}>
                {c.label ?? humanize(c.key)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular">
          {names.map((n) => (
            <tr key={n} className={`border-b border-grid last:border-0 ${highlight?.(n) ? 'bg-accent-wash/25 font-medium' : ''}`}>
              <td className="px-2 py-1.5 text-ink max-w-[28rem] whitespace-normal">{renderName ? renderName(n) : n}</td>
              {cols.map((c) => {
                const v = data[n]?.[c.key]
                return (
                  <td key={c.key} className={`px-2 py-1.5 text-ink-2 ${c.align === 'left' ? 'text-left' : 'text-right'}`}>
                    {c.format ? c.format(v, data[n]) : auto(c.key, v)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
