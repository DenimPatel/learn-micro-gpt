/**
 * Glossary links.
 *
 * `:::term rmsnorm` becomes a link to the definition in `index.json`. The
 * validator fails the build if a term is used but not defined, so a glossary link
 * here is never dangling -- which is worth more than it sounds, since a dangling
 * reference in a teaching site is a dead end at exactly the moment a reader asks
 * a question.
 */

import { useState } from 'react'
import { index } from '../data/sources'

const DEFINITIONS = index.glossary

export function GlossaryLink({ term }: { term: string }) {
  const [open, setOpen] = useState(false)
  const definition = DEFINITIONS[term]
  if (!definition) return <code>{term}</code>

  const id = `glossary-${term}`
  return (
    <>
      <button
        type="button"
        className="term"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((value) => !value)}
      >
        {term}
      </button>
      {open ? (
        <span className="term__definition" id={id} role="note">
          {definition}
        </span>
      ) : null}
    </>
  )
}
