/**
 * Glossary links.
 *
 * `:::term rmsnorm` becomes a link to the definition in `index.json`. The
 * validator fails the build if a term is used but not defined, so a glossary link
 * here is never dangling -- which is worth more than it sounds, since a dangling
 * reference in a teaching site is a dead end at exactly the moment a reader asks
 * a question.
 *
 * The note is a disclosure, not a modal: it opens on click, and it must be
 * dismissable the way a reader expects -- Escape, or a click anywhere else on the
 * page. A reader who opens three terms in a paragraph should never have to hunt
 * each trigger down to put the page back. Escape returns focus to the trigger so
 * a keyboard reader is not dropped at the top of the document.
 */

import { useEffect, useRef, useState } from 'react'
import { index } from '../data/sources'

const DEFINITIONS = index.glossary

export function GlossaryLink({ term }: { term: string }) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLSpanElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const definition = DEFINITIONS[term]

  // Dismissal is only wired up while the note is open, so a page full of closed
  // terms adds no listeners at all.
  useEffect(() => {
    if (!open) return

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      // Only dismiss the note this trigger owns. Pressing Escape with several
      // terms open should close the one the reader is on, not every note on the
      // page -- otherwise a reader loses the others' context too.
      if (!root.current?.contains(document.activeElement)) return
      setOpen(false)
      trigger.current?.focus()
    }
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node
      if (root.current?.contains(target)) return
      // A press on another glossary term is a press on the glossary, not a
      // dismissal of this note: a reader comparing two definitions keeps both.
      if ((target as Element).closest?.('.term-wrap')) return
      setOpen(false)
    }

    window.addEventListener('keydown', onKeyDown)
    // `pointerdown` rather than `click`: a press that starts outside the term is
    // a dismissal, even if it ends somewhere else (or drags a selection across
    // the note).
    window.addEventListener('pointerdown', onPointerDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('pointerdown', onPointerDown)
    }
  }, [open])

  if (!definition) return <code>{term}</code>

  const id = `glossary-${term}`
  return (
    <span className="term-wrap" ref={root}>
      <button
        ref={trigger}
        type="button"
        className="term"
        aria-expanded={open}
        aria-controls={id}
        aria-describedby={open ? id : undefined}
        onClick={() => setOpen((value) => !value)}
      >
        {term}
      </button>
      {open ? (
        <span className="term__definition" id={id} role="note">
          {definition}
        </span>
      ) : null}
    </span>
  )
}
