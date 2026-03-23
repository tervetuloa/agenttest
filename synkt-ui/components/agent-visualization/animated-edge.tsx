"use client"

import { memo, useId, useRef, useEffect, useCallback } from "react"
import { cn } from "@/lib/utils"

export type EdgeStatus = "idle" | "active" | "error" | "loop"

export interface AnimatedEdgeProps {
  id: string
  sourceX: number
  sourceY: number
  targetX: number
  targetY: number
  /** Outward normal at target border — used for arrow direction */
  targetNx?: number
  targetNy?: number
  /** Orthogonal waypoints (H/V turn points) computed upstream */
  waypoints?: { x: number; y: number }[]
  status?: EdgeStatus
  label?: string
  animateParticles?: boolean
  glowFilterId?: string
  className?: string
}

const statusStyles = {
  idle: {
    stroke: "rgba(255, 255, 255, 0.2)",
    strokeDasharray: "5,5",
    particleColor: "transparent",
  },
  active: {
    stroke: "#10b981",
    strokeDasharray: "none",
    particleColor: "#10b981",
  },
  error: {
    stroke: "#ef4444",
    strokeDasharray: "none",
    particleColor: "transparent",
  },
  loop: {
    stroke: "#f59e0b",
    strokeDasharray: "none",
    particleColor: "#f59e0b",
  },
}

const PARTICLE_SPEED = 120
const PARTICLE_COUNT = 3

/** Build a path through points with smooth quadratic-bezier rounded corners.
 *  All segments are horizontal or vertical; corners get a smooth 90° arc. */
function buildRoundedPath(points: { x: number; y: number }[], radius: number): string {
  if (points.length < 2) return ""
  if (points.length === 2) {
    return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`
  }

  let d = `M ${points[0].x} ${points[0].y}`

  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1]
    const curr = points[i]
    const next = points[i + 1]

    const d1x = curr.x - prev.x
    const d1y = curr.y - prev.y
    const d1len = Math.sqrt(d1x * d1x + d1y * d1y) || 1
    const d2x = next.x - curr.x
    const d2y = next.y - curr.y
    const d2len = Math.sqrt(d2x * d2x + d2y * d2y) || 1

    const r = Math.min(radius, d1len * 0.4, d2len * 0.4)

    const ax = curr.x - (d1x / d1len) * r
    const ay = curr.y - (d1y / d1len) * r
    const ex = curr.x + (d2x / d2len) * r
    const ey = curr.y + (d2y / d2len) * r

    d += ` L ${ax} ${ay} Q ${curr.x} ${curr.y} ${ex} ${ey}`
  }

  const last = points[points.length - 1]
  d += ` L ${last.x} ${last.y}`
  return d
}

export const AnimatedEdge = memo(function AnimatedEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  targetNx = 0,
  targetNy = -1,
  waypoints = [],
  status = "idle",
  label,
  animateParticles = true,
  glowFilterId,
  className,
}: AnimatedEdgeProps) {
  const uniqueId = useId()
  const pathId = `path-${id}-${uniqueId}`
  const { stroke, strokeDasharray, particleColor } = statusStyles[status]

  const pathRef = useRef<SVGPathElement>(null)
  const particleRefs = useRef<(SVGCircleElement | null)[]>([])
  const rafRef = useRef<number>(0)

  const dx = targetX - sourceX
  const dy = targetY - sourceY
  const distance = Math.sqrt(dx * dx + dy * dy)

  // Arrow scales down for short edges
  const arrowLen = Math.min(10, distance * 0.3)

  // Arrow direction: inward (reverse of outward target normal)
  const arrowAngle = Math.atan2(-targetNy, -targetNx)

  // End point: pull path back from border along the outward normal
  const endX = targetX + targetNx * arrowLen
  const endY = targetY + targetNy * arrowLen

  // Build orthogonal path through waypoints with rounded corners
  const allPoints = [
    { x: sourceX, y: sourceY },
    ...waypoints,
    { x: endX, y: endY },
  ]
  const pathD = buildRoundedPath(allPoints, 20)

  // Arrow polygon
  const ax1 = targetX - arrowLen * Math.cos(arrowAngle - Math.PI / 6)
  const ay1 = targetY - arrowLen * Math.sin(arrowAngle - Math.PI / 6)
  const ax2 = targetX - arrowLen * Math.cos(arrowAngle + Math.PI / 6)
  const ay2 = targetY - arrowLen * Math.sin(arrowAngle + Math.PI / 6)

  // Label at midpoint of the path
  let labelX: number, labelY: number
  if (waypoints.length > 0) {
    const mid = waypoints[Math.floor(waypoints.length / 2)]
    labelX = mid.x
    labelY = mid.y - 16
  } else {
    labelX = (sourceX + endX) / 2
    labelY = (sourceY + endY) / 2 - 16
  }

  const glowFilter = glowFilterId ? `url(#${glowFilterId})` : undefined
  const showParticles = animateParticles && (status === "active" || status === "loop")

  const setParticleRef = useCallback((el: SVGCircleElement | null, i: number) => {
    particleRefs.current[i] = el
  }, [])

  useEffect(() => {
    if (!showParticles) return
    const animate = (timestamp: number) => {
      const pathEl = pathRef.current
      if (!pathEl) { rafRef.current = requestAnimationFrame(animate); return }
      const totalLength = pathEl.getTotalLength()
      const durationMs = (totalLength / PARTICLE_SPEED) * 1000
      if (durationMs <= 0) { rafRef.current = requestAnimationFrame(animate); return }
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        const circle = particleRefs.current[i]
        if (!circle) continue
        const t = ((timestamp / durationMs + i / PARTICLE_COUNT) % 1)
        const point = pathEl.getPointAtLength(t * totalLength)
        circle.setAttribute("cx", String(point.x))
        circle.setAttribute("cy", String(point.y))
        const opacity = t < 0.1 ? t / 0.1 : t > 0.9 ? (1 - t) / 0.1 : 1
        circle.setAttribute("opacity", String(opacity))
      }
      rafRef.current = requestAnimationFrame(animate)
    }
    rafRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(rafRef.current)
  }, [showParticles])

  return (
    <g className={className}>
      <path
        ref={pathRef}
        id={pathId}
        d={pathD}
        fill="none"
        stroke={stroke}
        strokeWidth={2}
        strokeDasharray={strokeDasharray}
        strokeLinecap="round"
        filter={(status === "active" || status === "loop") ? glowFilter : undefined}
        className={cn(status === "loop" && "animate-pulse")}
      />

      <polygon
        points={`${ax1},${ay1} ${targetX},${targetY} ${ax2},${ay2}`}
        fill={stroke}
      />

      {label && distance > 80 && (
        <g>
          <rect
            x={labelX - label.length * 3.5 - 6}
            y={labelY - 10}
            width={label.length * 7 + 12}
            height={20}
            rx={6}
            fill="rgba(26, 26, 26, 0.9)"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth={1}
          />
          <text
            x={labelX}
            y={labelY + 4}
            textAnchor="middle"
            fill="rgba(255, 255, 255, 0.6)"
            fontSize={11}
            fontFamily="monospace"
          >
            {label}
          </text>
        </g>
      )}

      {showParticles && (
        <>
          {Array.from({ length: PARTICLE_COUNT }, (_, i) => (
            <circle
              key={i}
              ref={(el) => setParticleRef(el, i)}
              r={4}
              fill={particleColor}
              filter={glowFilter}
              opacity={0}
            />
          ))}
        </>
      )}
    </g>
  )
})
