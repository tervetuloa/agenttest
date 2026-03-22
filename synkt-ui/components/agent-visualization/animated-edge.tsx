"use client"

import { memo, useId } from "react"
import { cn } from "@/lib/utils"

export type EdgeStatus = "idle" | "active" | "error" | "loop"

export interface AnimatedEdgeProps {
  id: string
  sourceX: number
  sourceY: number
  targetX: number
  targetY: number
  status?: EdgeStatus
  label?: string
  animateParticles?: boolean
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

export const AnimatedEdge = memo(function AnimatedEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  status = "idle",
  label,
  animateParticles = true,
  className,
}: AnimatedEdgeProps) {
  const uniqueId = useId()
  const pathId = `path-${id}-${uniqueId}`
  const { stroke, strokeDasharray, particleColor } = statusStyles[status]

  const dx = targetX - sourceX
  const dy = targetY - sourceY
  const distance = Math.sqrt(dx * dx + dy * dy)

  // Scale curvature proportionally — avoid oversized curves for short edges
  const curvature = Math.min(distance * 0.3, Math.max(20, distance * 0.25))

  // Determine curve direction based on relative positions
  // For mostly horizontal edges, curve vertically; for vertical, curve horizontally
  const absDx = Math.abs(dx)
  const absDy = Math.abs(dy)

  let controlX1: number, controlY1: number, controlX2: number, controlY2: number

  if (distance < 30) {
    // Very short edges: straight line
    controlX1 = sourceX + dx * 0.33
    controlY1 = sourceY + dy * 0.33
    controlX2 = sourceX + dx * 0.66
    controlY2 = sourceY + dy * 0.66
  } else if (absDy > absDx) {
    // More vertical: curve outward horizontally
    const sign = dx >= 0 ? 1 : -1
    controlX1 = sourceX + sign * curvature
    controlY1 = sourceY + dy * 0.33
    controlX2 = targetX + sign * curvature
    controlY2 = targetY - dy * 0.33
  } else {
    // More horizontal: curve outward vertically
    const sign = dy >= 0 ? 1 : -1
    controlX1 = sourceX + dx * 0.33
    controlY1 = sourceY + sign * curvature
    controlX2 = targetX - dx * 0.33
    controlY2 = targetY + sign * curvature
  }

  const pathD = `M ${sourceX} ${sourceY} C ${controlX1} ${controlY1}, ${controlX2} ${controlY2}, ${targetX} ${targetY}`

  // Arrow direction: angle at target
  const arrowAngle = Math.atan2(targetY - controlY2, targetX - controlX2)
  const arrowLen = 10
  const ax1 = targetX - arrowLen * Math.cos(arrowAngle - Math.PI / 6)
  const ay1 = targetY - arrowLen * Math.sin(arrowAngle - Math.PI / 6)
  const ax2 = targetX - arrowLen * Math.cos(arrowAngle + Math.PI / 6)
  const ay2 = targetY - arrowLen * Math.sin(arrowAngle + Math.PI / 6)

  // Label position: midpoint of the curve
  const midT = 0.5
  const labelX = Math.pow(1-midT, 3) * sourceX + 3*Math.pow(1-midT, 2)*midT * controlX1 + 3*(1-midT)*midT*midT * controlX2 + midT*midT*midT * targetX
  const labelY = Math.pow(1-midT, 3) * sourceY + 3*Math.pow(1-midT, 2)*midT * controlY1 + 3*(1-midT)*midT*midT * controlY2 + midT*midT*midT * targetY

  return (
    <g className={className}>
      {/* Glow filter for active/loop states */}
      <defs>
        <filter id={`glow-${uniqueId}`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Main path */}
      <path
        id={pathId}
        d={pathD}
        fill="none"
        stroke={stroke}
        strokeWidth={2}
        strokeDasharray={strokeDasharray}
        strokeLinecap="round"
        filter={(status === "active" || status === "loop") ? `url(#glow-${uniqueId})` : undefined}
        className={cn(status === "loop" && "animate-pulse")}
      />

      {/* Arrow marker - rotated to match edge direction */}
      <polygon
        points={`${ax1},${ay1} ${targetX},${targetY} ${ax2},${ay2}`}
        fill={stroke}
      />

      {/* Edge label */}
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

      {/* Animated particles for active/loop states */}
      {animateParticles && (status === "active" || status === "loop") && (
        <>
          {[0, 0.33, 0.66].map((offset, i) => (
            <circle
              key={i}
              r={4}
              fill={particleColor}
              filter={`url(#glow-${uniqueId})`}
              className="opacity-0"
              style={{
                offsetPath: `path("${pathD}")`,
                animation: `particle-flow 1.5s ease-in-out ${offset * 1.5}s infinite`,
              }}
            />
          ))}
        </>
      )}
    </g>
  )
})
