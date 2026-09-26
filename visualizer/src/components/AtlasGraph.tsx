/**
 * The concept graph, laid out with dagre.
 *
 * Edges come from `prereqs` in the concept frontmatter, resolved into `index.json`
 * by `tools/render_content.py` -- so the graph and the reading order cannot
 * disagree, and the validator rejects a prereq cycle before this ever renders.
 *
 * The graph is progressive enhancement. `dagre` is loaded dynamically, so a reader
 * who navigates straight to a concept never downloads the layout engine; until it
 * arrives (or if it fails) the same information is shown as a nested list, which
 * is also what a screen reader gets, because an SVG of boxes and curves is not
 * something a screen reader can navigate. Both are always in the DOM.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { index } from '../data/sources'

interface Node {
  id: string
  title: string
  chapter: string
  x: number
  y: number
  width: number
  height: number
}

/*
 * Node geometry, in graph units (the SVG scales to fit its container).
 *
 * The width is set by the longest concept title rather than by taste: at 12px the
 * label needs about 6.1 units per character, so 168 fits 24 characters plus
 * padding, and the truncation below is only reached by the handful of genuinely
 * long names. A graph whose nodes are visibly too narrow for their labels is
 * worse than one with a long name shortened.
 */
const NODE_W = 172
const NODE_H = 44
const LABEL_CHARS = 24

export function AtlasGraph() {
  const [nodes, setNodes] = useState<Node[] | null>(null)
  const [failed, setFailed] = useState(false)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    let cancelled = false

    void import('dagre')
      .then((dagre) => {
        if (cancelled || !mounted.current) return
        const graph = new dagre.graphlib.Graph()
        graph.setGraph({ rankdir: 'LR', nodesep: 28, ranksep: 90, marginx: 12, marginy: 12 })
        graph.setDefaultEdgeLabel(() => ({}))

        for (const edge of index.edges) {
          if (edge.kind === 'related') continue
          graph.setNode(edge.from, { width: NODE_W, height: NODE_H })
          graph.setNode(edge.to, { width: NODE_W, height: NODE_H })
          graph.setEdge(edge.from, edge.to)
        }
        dagre.layout(graph)

        const laidOut: Node[] = index.chapters.flatMap((chapter) =>
          chapter.concepts.map((concept) => {
            const laid = graph.node(concept.id) as { x: number; y: number } | undefined
            return {
              id: concept.id,
              title: concept.title,
              chapter: chapter.id,
              x: (laid?.x ?? 0) - NODE_W / 2,
              y: (laid?.y ?? 0) - NODE_H / 2,
              width: NODE_W,
              height: NODE_H,
            }
          }),
        )
        if (!cancelled && mounted.current) setNodes(laidOut)
      })
      .catch(() => {
        // The list below is the fallback, and it is a complete representation.
        if (!cancelled && mounted.current) setFailed(true)
      })

    return () => {
      cancelled = true
      mounted.current = false
    }
  }, [])

  const edges = useMemo(() => index.edges.filter((e) => e.kind !== 'related'), [])

  return (
    <div className="atlas">
      {nodes && !failed ? (
        <div className="atlas__canvas">
          <svg
            className="atlas__svg"
            viewBox={`0 0 ${Math.max(...nodes.map((n) => n.x + n.width)) + 12} ${
              Math.max(...nodes.map((n) => n.y + n.height)) + 12
            }`}
            role="img"
            aria-label={`A graph of the ${nodes.length} concepts and the ${edges.length} prerequisite relations between them. The same information is in the list below.`}
          >
            <defs>
              <marker
                id="arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" className="atlas__arrow" />
              </marker>
            </defs>
            {edges.map((edge) => {
              const from = nodes.find((n) => n.id === edge.from)
              const to = nodes.find((n) => n.id === edge.to)
              if (!from || !to) return null
              return (
                <line
                  key={`${edge.from}->${edge.to}`}
                  x1={from.x + from.width}
                  y1={from.y + from.height / 2}
                  x2={to.x}
                  y2={to.y + to.height / 2}
                  className="atlas__edge"
                  markerEnd="url(#arrow)"
                />
              )
            })}
            {nodes.map((node) => (
              <a key={node.id} href={`#/learn/${node.id}`} aria-label={node.title}>
                <rect
                  x={node.x}
                  y={node.y}
                  width={node.width}
                  height={node.height}
                  rx={8}
                  className={`atlas__node atlas__node--${node.chapter}`}
                />
                <text
                  x={node.x + node.width / 2}
                  y={node.y + node.height / 2 + 4}
                  textAnchor="middle"
                  className="atlas__label"
                >
                  {node.title.length > LABEL_CHARS
                    ? `${node.title.slice(0, LABEL_CHARS - 1)}…`
                    : node.title}
                </text>
                {/* The full title, for the hover and for anyone reading the
                    markup: the visible label is shortened, the title is not. */}
                <title>{node.title}</title>
              </a>
            ))}
          </svg>
        </div>
      ) : (
        <p className="atlas__loading">
          {failed
            ? 'The graph could not be laid out, so here it is as a list — which is the same information.'
            : 'Laying out the graph…'}
        </p>
      )}

      <ol className="atlas__list">
        {index.chapters.map((chapter) => (
          <li key={chapter.id}>
            <h3>{chapter.title}</h3>
            <ul>
              {chapter.concepts.map((concept) => (
                <li key={concept.id}>
                  <a href={`#/learn/${concept.id}`}>{concept.title}</a>
                  {concept.summary ? <span> — {concept.summary}</span> : null}
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </div>
  )
}
