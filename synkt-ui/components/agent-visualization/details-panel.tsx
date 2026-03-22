"use client"

import { memo, useRef, useEffect, useState } from "react"
import { X, Bot, Search, FileText, Zap, Clock, DollarSign, Hash, AlertCircle, CheckCircle2, ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { type AgentNodeProps, type AgentStatus } from "./agent-node"
import { GlassButton } from "./glass-button"

export interface ActivityLogEntry {
  time: string
  event: string
  type?: "info" | "error" | "success" | "handoff"
}

export interface DetailsPanelProps {
  agent: AgentNodeProps | null
  isOpen: boolean
  onClose: () => void
  activityLog?: ActivityLogEntry[]
  messages?: Array<{ from_agent: string; to_agent: string; content: string; timestamp: number }>
  className?: string
}

const typeIcons = {
  research: Search,
  writer: FileText,
  analyzer: Bot,
  executor: Zap,
}

const statusConfig: Record<AgentStatus, { icon: typeof CheckCircle2; color: string; label: string }> = {
  idle: { icon: Clock, color: "text-white/50", label: "Idle" },
  active: { icon: Zap, color: "text-accent-green", label: "Active" },
  error: { icon: AlertCircle, color: "text-accent-red", label: "Error" },
  completed: { icon: CheckCircle2, color: "text-accent-blue", label: "Completed" },
}

const logTypeColors = {
  info: "text-white/40",
  error: "text-accent-red",
  success: "text-accent-green",
  handoff: "text-accent-blue",
}

export const DetailsPanel = memo(function DetailsPanel({
  agent,
  isOpen,
  onClose,
  activityLog,
  messages,
  className,
}: DetailsPanelProps) {
  const lastAgentRef = useRef(agent)
  const [displayAgent, setDisplayAgent] = useState(agent)

  useEffect(() => {
    if (agent) {
      lastAgentRef.current = agent
      setDisplayAgent(agent)
    }
  }, [agent])

  const activeAgent = displayAgent || lastAgentRef.current
  if (!activeAgent) return null

  const TypeIcon = typeIcons[activeAgent.type]
  const { icon: StatusIcon, color: statusColor, label: statusLabel } = statusConfig[activeAgent.status]

  // Filter messages relevant to this agent
  const agentMessages = messages?.filter(
    (m) => m.from_agent === activeAgent.id || m.to_agent === activeAgent.id
  ) ?? []

  // Build activity log from real data or fall back to provided log or defaults
  const displayLog: ActivityLogEntry[] = activityLog && activityLog.length > 0
    ? activityLog
    : agentMessages.length > 0
      ? agentMessages.map((m) => ({
          time: new Date(m.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
          event: m.from_agent === activeAgent.id
            ? `Handed off to ${m.to_agent}`
            : `Received from ${m.from_agent}`,
          type: (m.from_agent === activeAgent.id ? "handoff" : "info") as "handoff" | "info",
        }))
      : [
          { time: "-", event: "No activity recorded", type: "info" as const },
        ]

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          "fixed inset-0 z-40 transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]",
          isOpen
            ? "bg-black/20 backdrop-blur-sm opacity-100 pointer-events-auto"
            : "bg-transparent backdrop-blur-none opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={cn(
          "fixed top-0 right-0 h-full w-[400px] z-50",
          "border-l border-white/8 bg-[rgba(26,26,26,0.95)] backdrop-blur-[20px]",
          "shadow-[-20px_0_60px_rgba(0,0,0,0.5)]",
          "transform transition-transform duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]",
          isOpen ? "translate-x-0" : "translate-x-full",
          className
        )}
        style={{ willChange: "transform" }}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/8 p-6">
          <div className="flex items-center gap-3">
            <div className={cn(
              "flex h-10 w-10 items-center justify-center rounded-xl",
              "bg-white/5 text-white/70"
            )}>
              <TypeIcon className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-[17px] font-semibold text-white">{activeAgent.name}</h2>
              <p className="text-[13px] text-white/50 capitalize">{activeAgent.type} Agent</p>
            </div>
          </div>
          <GlassButton size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </GlassButton>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 overflow-y-auto h-[calc(100%-80px)]">
          {/* Status */}
          <div className="rounded-xl border border-white/8 bg-white/5 p-4">
            <div className="flex items-center gap-3">
              <StatusIcon className={cn("h-5 w-5", statusColor)} />
              <div>
                <p className="text-[11px] uppercase tracking-wider text-white/40">Status</p>
                <p className={cn("text-[15px] font-medium", statusColor)}>{statusLabel}</p>
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-white/8 bg-white/5 p-4">
              <div className="flex items-center gap-2 text-white/40 text-[11px] uppercase tracking-wider">
                <DollarSign className="h-3.5 w-3.5" />
                Cost
              </div>
              <p className="mt-1 text-[20px] font-semibold text-white font-mono">
                ${activeAgent.cost.toFixed(3)}
              </p>
            </div>

            <div className="rounded-xl border border-white/8 bg-white/5 p-4">
              <div className="flex items-center gap-2 text-white/40 text-[11px] uppercase tracking-wider">
                <Hash className="h-3.5 w-3.5" />
                ID
              </div>
              <p className="mt-1 text-[15px] font-mono text-white truncate">
                {activeAgent.id}
              </p>
            </div>
          </div>

          {/* Connections - show who this agent talks to */}
          {agentMessages.length > 0 && (() => {
            // Precompute peer counts in a single pass
            const peerCounts = new Map<string, { outgoing: number; incoming: number }>()
            for (const m of agentMessages) {
              const peer = m.from_agent === activeAgent.id ? m.to_agent : m.from_agent
              const counts = peerCounts.get(peer) ?? { outgoing: 0, incoming: 0 }
              if (m.from_agent === activeAgent.id) counts.outgoing++
              else counts.incoming++
              peerCounts.set(peer, counts)
            }

            return (
              <div>
                <h3 className="text-[13px] font-medium text-white/60 mb-3">Connections</h3>
                <div className="space-y-2">
                  {Array.from(peerCounts.entries()).map(([peer, { outgoing, incoming }]) => (
                    <div
                      key={peer}
                      className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.03] px-3 py-2"
                    >
                      <div className="flex items-center gap-2 text-[13px] text-white/70">
                        <ArrowRight className="h-3.5 w-3.5 text-white/30" />
                        <span className="font-mono">{peer}</span>
                      </div>
                      <div className="flex items-center gap-2 text-[11px] text-white/40">
                        {outgoing > 0 && <span>{outgoing} sent</span>}
                        {incoming > 0 && <span>{incoming} recv</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )
          })()}

          {/* Activity Log */}
          <div>
            <h3 className="text-[13px] font-medium text-white/60 mb-3">Activity Log</h3>
            <div className="space-y-1 max-h-[300px] overflow-y-auto">
              {displayLog.map((log, i) => (
                <div
                  key={i}
                  className="flex items-start gap-3 text-[13px] py-2 border-b border-white/5 last:border-0"
                >
                  <span className="font-mono text-white/40 w-20 shrink-0">{log.time}</span>
                  <span className={cn("text-white/70", log.type && logTypeColors[log.type])}>
                    {log.event}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  )
})
