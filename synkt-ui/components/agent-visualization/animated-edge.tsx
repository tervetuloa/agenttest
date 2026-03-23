"use client"

import { memo, useId, useRef, useEffect, useCallback } from "react"
import { cn } from "@/lib/utils"

export type EdgeStatus = "idle" | "active" | "error" | "loop"

export interface AnimatedEdgeProps {
  id: string
  sourceX: number
  sourceY: number
  /** Outward normal at source border */
  sourceNx?: number
  sourceNy?: number
  targetX: number
  targetY: number
  /** Outward normal at target border — used for arrow direction */
  targetNx?: number
  targetNy?: number
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

export const AnimatedEdge = memo(function AnimatedEdge({
  id,
  sourceX,
  sourceY,
  sourceNx = 0,
  sourceNy = 1,
  targetX,
  targetY,
  targetNx = 0,
  targetNy = -1,
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
  const progressRef = useRef<number[]>(
    Array.from({ length: PARTICLE_COUNT }, (_, i) => i / PARTICLE_COUNT)
  )
  const lastTimeRef = useRef<number>(0)

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

  // Cubic bezier control point offset — proportional to distance, clamped
  const controlOffset = Math.max(30, Math.min(200, distance * 0.4))

  // Control points extend outward from each border along the normal
  const cx1 = sourceX + sourceNx * controlOffset
  const cy1 = sourceY + sourceNy * controlOffset
  const cx2 = endX + targetNx * controlOffset
  const cy2 = endY + targetNy * controlOffset

  const pathD = `M ${sourceX} ${sourceY} C ${cx1} ${cy1} ${cx2} ${cy2} ${endX} ${endY}`

  // Arrow polygon
  const ax1 = targetX - arrowLen * Math.cos(arrowAngle - Math.PI / 6)
  const ay1 = targetY - arrowLen * Math.sin(arrowAngle - Math.PI / 6)
  const ax2 = targetX - arrowLen * Math.cos(arrowAngle + Math.PI / 6)
  const ay2 = targetY - arrowLen * Math.sin(arrowAngle + Math.PI / 6)

  // Label at bezier midpoint (t=0.5): 0.125*P0 + 0.375*C1 + 0.375*C2 + 0.125*P3
  const labelX = 0.125 * sourceX + 0.375 * cx1 + 0.375 * cx2 + 0.125 * endX
  const labelY = 0.125 * sourceY + 0.375 * cy1 + 0.375 * cy2 + 0.125 * endY - 16

  const glowFilter = glowFilterId ? `url(#${glowFilterId})` : undefined
  const showParticles = animateParticles && (status === "active" || status === "loop")

  const setParticleRef = useCallback((el: SVGCircleElement | null, i: number) => {
    particleRefs.current[i] = el
  }, [])

  // Stable particle animation: tracks progress (0–1) per particle,
  // increments by constant pixel speed each frame. When the path changes
  // (e.g. node dragging), progress stays stable — no jumps.
  useEffect(() => {
    if (!showParticles) return

    progressRef.current = Array.from({ length: PARTICLE_COUNT }, (_, i) => i / PARTICLE_COUNT)
    lastTimeRef.current = 0

    const animate = (timestamp: number) => {
      const pathEl = pathRef.current
      if (!pathEl) { rafRef.current = requestAnimationFrame(animate); return }

      const totalLength = pathEl.getTotalLength()
      if (totalLength <= 0) { rafRef.current = requestAnimationFrame(animate); return }

      const deltaMs = lastTimeRef.current ? Math.min(timestamp - lastTimeRef.current, 50) : 16
      lastTimeRef.current = timestamp

      const progressIncrement = (PARTICLE_SPEED * deltaMs / 1000) / totalLength

      for (let i = 0; i < PARTICLE_COUNT; i++) {
        const circle = particleRefs.current[i]
        if (!circle) continue

        progressRef.current[i] = (progressRef.current[i] + progressIncrement) % 1
        const t = progressRef.current[i]
        const point = pathEl.getPointAtLength(t * totalLength)
        circle.setAttribute("cx", String(point.x))
        circle.setAttribute("cy", String(point.y))

        const opacity = t < 0.1 ? t / 0.1 : t > 0.9 ? (1 - t) / 0.1 : 1
        circle.setAttribute("opacity", String(opacity))
      }

      rafRef.current = requestAnimationFrame(animate)
    }

    rafRef.current = requestAnimationFrame(animate)
    return () => { cancelAnimationFrame(rafRef.current); lastTimeRef.current = 0 }
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
