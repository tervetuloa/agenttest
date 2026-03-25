"use client"

import { memo } from "react"
import { Bot, Search, FileText, Zap, AlertCircle, AlertTriangle, GripVertical } from "lucide-react"
import { cn } from "@/lib/utils"

export type AgentStatus = "idle" | "active" | "error" | "completed"

export type StructuralWarning = "dead_end" | "unreachable" | "unbounded_cycle" | "no_end_path"

export interface AgentNodeProps {
  id: string
  name: string
  type: "research" | "writer" | "analyzer" | "executor"
  status: AgentStatus
  cost: number
  isSelected?: boolean
  onClick?: () => void
  draggable?: boolean
  structuralWarnings?: StructuralWarning[]
}

const typeIcons = {
  research: Search,
  writer: FileText,
  analyzer: Bot,
  executor: Zap,
}

const statusColors = {
  idle: "border-white/10",
  active: "border-accent-green",
  error: "border-accent-red",
  completed: "border-accent-blue",
}

const statusGlows = {
  idle: "",
  active: "shadow-[0_0_20px_rgba(16,185,129,0.3)]",
  error: "shadow-[0_0_20px_rgba(239,68,68,0.3)] animate-[shake_0.3s_ease-in-out]",
  completed: "shadow-[0_0_20px_rgba(59,130,246,0.3)]",
}

const warningLabels: Record<StructuralWarning, string> = {
  dead_end: "Dead end",
  unreachable: "Unreachable",
  unbounded_cycle: "Infinite loop",
  no_end_path: "No path to END",
}

const warningColors: Record<StructuralWarning, string> = {
  dead_end: "bg-accent-amber text-amber-900",
  unreachable: "bg-purple-400 text-purple-900",
  unbounded_cycle: "bg-accent-red text-white",
  no_end_path: "bg-orange-400 text-orange-900",
}

export const AgentNode = memo(function AgentNode({
  name,
  type,
  status,
  cost,
  isSelected,
  onClick,
  draggable,
  structuralWarnings,
}: AgentNodeProps) {
  const Icon = typeIcons[type]
  const hasWarnings = structuralWarnings && structuralWarnings.length > 0

  return (
    <div
      onClick={onClick}
      className={cn(
        "group relative w-[280px] cursor-pointer",
        "rounded-2xl border bg-[rgba(36,36,36,0.85)] backdrop-blur-[20px]",
        "p-5 transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)]",
        hasWarnings ? "border-accent-amber/60" : statusColors[status],
        hasWarnings ? "shadow-[0_0_20px_rgba(245,158,11,0.2)]" : statusGlows[status],
        isSelected && "border-accent-blue shadow-[0_0_20px_rgba(59,130,246,0.3)]",
        "hover:scale-[1.02] hover:brightness-110",
        "active:scale-[0.98]"
      )}
    >
      {/* Drag handle */}
      {draggable && (
        <div
          data-node-handle
          className="absolute top-2 right-2 p-1 rounded-md opacity-0 group-hover:opacity-100 hover:bg-white/10 cursor-grab active:cursor-grabbing transition-opacity duration-150"
          onClick={(e) => e.stopPropagation()}
        >
          <GripVertical className="h-4 w-4 text-white/40" />
        </div>
      )}

      {/* Header */}
      <div className="flex items-center gap-3">
        <div className={cn(
          "flex h-10 w-10 items-center justify-center rounded-xl",
          "bg-white/5 text-white/70 transition-colors duration-200",
          status === "active" && "bg-accent-green/20 text-accent-green",
          status === "error" && "bg-accent-red/20 text-accent-red",
          status === "completed" && "bg-accent-blue/20 text-accent-blue",
          hasWarnings && "bg-accent-amber/20 text-accent-amber"
        )}>
          <Icon className="h-5 w-5" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-[15px] font-medium text-white truncate">{name}</h3>
        </div>
      </div>

      {/* Structural warnings */}
      {hasWarnings && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {structuralWarnings!.map((warning, index) => (
            <span
              key={`${warning}-${index}`}
              className={cn(
                "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-semibold",
                warningColors[warning]
              )}
            >
              <AlertTriangle className="h-3 w-3" />
              {warningLabels[warning]}
            </span>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              status === "idle" && "bg-white/30",
              status === "active" && "bg-accent-green animate-[pulse-glow_2s_ease-in-out_infinite]",
              status === "error" && "bg-accent-red",
              status === "completed" && "bg-accent-blue"
            )}
          />
          <span className="text-[13px] text-white/60 capitalize">{status}</span>
        </div>
        <span className="text-[13px] font-mono text-white/50">
          ${cost.toFixed(3)}
        </span>
      </div>

      {/* Error indicator */}
      {status === "error" && (
        <div className="absolute -top-2 -right-2 flex h-6 w-6 items-center justify-center rounded-full bg-accent-red text-white">
          <AlertCircle className="h-3.5 w-3.5" />
        </div>
      )}

      {/* Structural warning indicator (top-left) */}
      {hasWarnings && status !== "error" && (
        <div className="absolute -top-2 -left-2 flex h-6 w-6 items-center justify-center rounded-full bg-accent-amber text-amber-900">
          <AlertTriangle className="h-3.5 w-3.5" />
        </div>
      )}
    </div>
  )
})
