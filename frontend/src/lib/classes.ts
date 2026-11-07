import type { ClassMeta } from './api'

/** Body style from the Stanford Cars model string ("Silverado 1500 Extended Cab" → "Pickup"). */
export function bodyType(model: string): string {
  if (/\b(Crew|Extended|Regular|Club|Quad|Super)?\s?Cab\b|Supercab/i.test(model)) return 'Pickup'
  for (const b of ['Convertible', 'Coupe', 'Sedan', 'Hatchback', 'Wagon', 'Minivan', 'SUV', 'Van']) {
    if (new RegExp(`\\b${b}\\b`, 'i').test(model)) return b
  }
  return 'Other'
}

export const indexByName = (classes: ClassMeta[]) => new Map(classes.map((c) => [c.name, c]))

export interface RunnerUpInsight {
  runnerUp: string
  margin: number
  strength: 'decisive' | 'clear' | 'narrow'
  kind: 'year' | 'model' | 'lookalike' | 'different'
  text: string
}

/** Explain why the runner-up class was close (or not): the same reasoning used for the notebook's failure tagging. */
export function runnerUpInsight(top5: { label: string; confidence: number }[], byName: Map<string, ClassMeta>): RunnerUpInsight | null {
  if (top5.length < 2) return null
  const [first, second] = top5
  const a = byName.get(first.label)
  const b = byName.get(second.label)
  const margin = first.confidence - second.confidence
  const strength = margin > 0.8 ? 'decisive' : margin > 0.3 ? 'clear' : 'narrow'
  if (!a || !b) return { runnerUp: second.label, margin, strength, kind: 'different', text: '' }
  const ba = bodyType(a.model)
  const bb = bodyType(b.model)
  if (a.make === b.make && a.model === b.model)
    return { runnerUp: second.label, margin, strength, kind: 'year',
      text: `Same ${a.make} ${a.model}; only the model year differs (${a.year} vs ${b.year}). Generations often differ only by a facelift, so the year is the hardest part to get right.` }
  if (a.make === b.make)
    return { runnerUp: second.label, margin, strength, kind: 'model',
      text: `Both are ${a.make} vehicles sharing brand styling (grille, headlights). The decision rests on body shape and proportions: ${ba.toLowerCase()} vs ${bb.toLowerCase()}.` }
  if (ba === bb && ba !== 'Other')
    return { runnerUp: second.label, margin, strength, kind: 'lookalike',
      text: `A different make (${b.make}) but the same body style (${ba.toLowerCase()}), so the silhouettes are similar. Brand cues such as the grille and badge separate them.` }
  return { runnerUp: second.label, margin, strength, kind: 'different',
    text: `A different make and body style (${b.make}, ${bb.toLowerCase()}). The runner-up is only a weak alternative.` }
}
