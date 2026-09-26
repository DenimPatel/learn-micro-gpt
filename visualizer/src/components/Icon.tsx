/**
 * The icon set.
 *
 * Every glyph on this site comes from here. The previous design used text
 * characters for its interface -- `☀`, `☾`, `…`, `·`, `◆` -- which is why the
 * chrome looked like a prototype: those glyphs come from whatever font the
 * reader happens to have, they do not align to a 24-pixel grid, and they change
 * shape between operating systems.
 *
 * Stroke-based, one weight, `currentColor`, no fills except where a shape is the
 * point. `aria-hidden` by default because every icon in the app sits next to text
 * or inside a labelled control; the label is the accessible name, and a
 * duplicated one is noise. `focusable="false"` for the same reason as
 * `aria-hidden`: an inline SVG inside a button is otherwise a tab stop in some
 * engines.
 */

export type IconName =
  | 'sun'
  | 'moon'
  | 'menu'
  | 'close'
  | 'note'
  | 'measure'
  | 'shape'
  | 'warn'
  | 'ok'
  | 'arrow'
  | 'back'
  | 'copy'
  | 'check'
  | 'file'
  | 'reset'

const PATHS: Record<IconName, { d: string; fill?: boolean }[]> = {
  sun: [
    { d: 'M12 7.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9Z' },
    {
      d: 'M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.5 1.5M17.6 17.6l1.5 1.5M19.1 4.9l-1.5 1.5M6.4 17.6l-1.5 1.5',
    },
  ],
  moon: [{ d: 'M20 13.4A8.2 8.2 0 0 1 10.6 4a8.5 8.5 0 1 0 9.4 9.4Z' }],
  menu: [{ d: 'M4 7h16M4 12h16M4 17h10' }],
  close: [{ d: 'M6 6l12 12M18 6 6 18' }],
  note: [
    { d: 'M5 4.8h9.5L19 9.3V19a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1Z' },
    { d: 'M14 4.8V9.5H19' },
    { d: 'M8.5 13.5h7M8.5 16.5h4.5' },
  ],
  measure: [{ d: 'M3 15.5 15.5 3 21 8.5 8.5 21Z' }, { d: 'M7 11.5 9 13.5M10 8.5l2 2M13 5.5l2 2' }],
  shape: [{ d: 'M4 4.5h6.5v6.5H4zM13.5 4.5H20v6.5h-6.5zM4 13.5h6.5V20H4zM13.5 13.5H20V20h-6.5z' }],
  warn: [{ d: 'M12 4.2 21 19.8H3Z' }, { d: 'M12 10v4' }, { d: 'M12 17h.01' }],
  ok: [
    { d: 'M12 3.2a8.8 8.8 0 1 0 0 17.6 8.8 8.8 0 0 0 0-17.6Z' },
    { d: 'M8.2 12.3 11 15l4.9-5.6' },
  ],
  arrow: [{ d: 'M5 12h13' }, { d: 'M12.5 6.5 18 12l-5.5 5.5' }],
  back: [{ d: 'M19 12H6' }, { d: 'M11.5 6.5 6 12l5.5 5.5' }],
  copy: [{ d: 'M9 9V5.5h9.5V15h-3' }, { d: 'M5.5 9H15v9.5H5.5Z' }],
  check: [{ d: 'M5 12.5 10 17.5 19 7' }],
  file: [{ d: 'M6 3.5h7.5L18 8v12.5H6Z' }, { d: 'M13.5 3.5V8H18' }, { d: 'M9 12.5h6M9 16h4' }],
  reset: [{ d: 'M20 12a8 8 0 1 1-2.6-5.9' }, { d: 'M20 3.6V8h-4.4' }],
}

export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {PATHS[name].map((path) => (
        <path key={path.d} d={path.d} fill={path.fill ? 'currentColor' : 'none'} />
      ))}
    </svg>
  )
}
