"use client"

import { useState, useCallback, useEffect, useRef, useMemo } from "react"
import { Header } from "@/components/agent-visualization/header"
import { GraphCanvas, type GraphNode, type GraphEdge } from "@/components/agent-visualization/graph-canvas"
import { CostPanel } from "@/components/agent-visualization/cost-panel"
import { Timeline } from "@/components/agent-visualization/timeline"
import { DetailsPanel } from "@/components/agent-visualization/details-panel"
import { SettingsPanel, type Settings } from "@/components/agent-visualization/settings-panel"
import { type AgentStatus } from "@/components/agent-visualization/agent-node"
import { type EdgeStatus } from "@/components/agent-visualization/animated-edge"
import { useTraceStream, type TraceData } from "@/hooks/use-trace-stream"

// ─── Demo data (fallback when server is not running) ───

const initialNodes: GraphNode[] = [
  { id: "orchestrator", name: "Orchestrator", type: "executor", status: "completed", cost: 0.012, x: 400, y: 50 },
  { id: "researcher", name: "Research Agent", type: "research", status: "active", cost: 0.042, x: 100, y: 200 },
  { id: "analyzer", name: "Data Analyzer", type: "analyzer", status: "active", cost: 0.028, x: 400, y: 200 },
  { id: "writer", name: "Content Writer", type: "writer", status: "idle", cost: 0.000, x: 700, y: 200 },
  { id: "reviewer", name: "Quality Reviewer", type: "analyzer", status: "idle", cost: 0.000, x: 400, y: 380 },
  { id: "publisher", name: "Publisher", type: "executor", status: "idle", cost: 0.000, x: 700, y: 380 },
]

const initialEdges: GraphEdge[] = [
  { id: "e1", source: "orchestrator", target: "researcher", status: "active" },
  { id: "e2", source: "orchestrator", target: "analyzer", status: "active" },
  { id: "e3", source: "orchestrator", target: "writer", status: "idle" },
  { id: "e4", source: "researcher", target: "analyzer", status: "active" },
  { id: "e5", source: "analyzer", target: "writer", status: "idle" },
  { id: "e6", source: "writer", target: "reviewer", status: "idle" },
  { id: "e7", source: "reviewer", target: "publisher", status: "idle" },
  { id: "e8", source: "reviewer", target: "writer", status: "idle" },
]

const demoTimeline: { time: number; nodeUpdates: Record<string, AgentStatus>; edgeUpdates: Record<string, EdgeStatus> }[] = [
  { time: 0, nodeUpdates: { orchestrator: "active" }, edgeUpdates: {} },
  { time: 0.5, nodeUpdates: { orchestrator: "completed", researcher: "active", analyzer: "active" }, edgeUpdates: { e1: "active", e2: "active" } },
  { time: 1.5, nodeUpdates: {}, edgeUpdates: { e4: "active" } },
  { time: 2.5, nodeUpdates: { researcher: "completed" }, edgeUpdates: {} },
  { time: 3.0, nodeUpdates: { analyzer: "completed", writer: "active" }, edgeUpdates: { e3: "active", e5: "active" } },
  { time: 4.0, nodeUpdates: { writer: "completed", reviewer: "active" }, edgeUpdates: { e6: "active" } },
  { time: 4.5, nodeUpdates: { reviewer: "error" }, edgeUpdates: { e8: "error" } },
  { time: 5.0, nodeUpdates: { reviewer: "completed", writer: "active" }, edgeUpdates: { e8: "loop" } },
  { time: 5.5, nodeUpdates: { writer: "completed", publisher: "active" }, edgeUpdates: { e7: "active" } },
  { time: 6.0, nodeUpdates: { publisher: "completed" }, edgeUpdates: {} },
]

// ─── Helpers ───

const agentTypeMap: Record<string, "research" | "writer" | "analyzer" | "executor"> = {
  research: "research",
  writer: "writer",
  analyzer: "analyzer",
  executor: "executor",
}

function mapAgentStatus(s: string): AgentStatus {
  if (s === "complete") return "completed"
  if (s === "active" || s === "idle" || s === "error") return s
  return "idle"
}

function traceToGraph(trace: TraceData, showEdgeLabels: boolean): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const agents = trace.agents
  const cols = Math.max(Math.ceil(Math.sqrt(agents.length)), 2)
  const nodes: GraphNode[] = agents.map((a, i) => ({
    id: a.name,
    name: a.name,
    type: agentTypeMap[a.type] || "executor",
    status: mapAgentStatus(a.status),
    cost: a.cost,
    x: 100 + (i % cols) * 340,
    y: 80 + Math.floor(i / cols) * 200,
  }))

  const edgeSet = new Set<string>()
  const edges: GraphEdge[] = []
  for (const msg of trace.messages) {
    const key = `${msg.from_agent}->${msg.to_agent}`
    if (edgeSet.has(key)) continue
    edgeSet.add(key)

    let status: EdgeStatus = "active"
    if (trace.loop_detected && trace.loop_agents.includes(msg.from_agent) && trace.loop_agents.includes(msg.to_agent)) {
      status = "loop"
    }

    edges.push({
      id: `e-${edges.length}`,
      source: msg.from_agent,
      target: msg.to_agent,
      status,
      label: showEdgeLabels && msg.content ? msg.content.slice(0, 20) : undefined,
    })
  }

  return { nodes, edges }
}

// Layout presets
type LayoutMode = "freeform" | "hierarchical" | "radial"

function applyLayout(nodes: GraphNode[], mode: LayoutMode): GraphNode[] {
  if (mode === "hierarchical") {
    // Arrange top-to-bottom with staggered rows
    const cols = Math.max(Math.ceil(Math.sqrt(nodes.length)), 2)
    return nodes.map((n, i) => ({
      ...n,
      x: (i % cols) * 340 + 50,
      y: Math.floor(i / cols) * 200 + 50,
    }))
  }
  if (mode === "radial") {
    const cx = 400
    const cy = 300
    const radius = Math.max(200, nodes.length * 40)
    return nodes.map((n, i) => {
      const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2
      return {
        ...n,
        x: cx + Math.cos(angle) * radius - 140,
        y: cy + Math.sin(angle) * radius - 50,
      }
    })
  }
  return nodes
}

// ─── Toast notification ───

function Toast({ message, type, onDismiss }: { message: string; type: "error" | "warning" | "info"; onDismiss: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 8000)
    return () => clearTimeout(timer)
  }, [onDismiss])

  const colors = {
    error: "border-accent-red bg-red-900/80 text-red-200",
    warning: "border-accent-amber bg-amber-900/80 text-amber-200",
    info: "border-accent-blue bg-blue-900/80 text-blue-200",
  }

  return (
    <div className={`rounded-xl border px-4 py-3 text-sm shadow-lg backdrop-blur-xl animate-[fade-in-up_0.3s_ease-out] ${colors[type]}`}>
      <div className="flex items-center gap-3">
        <span>{message}</span>
        <button onClick={onDismiss} className="text-white/50 hover:text-white/80 text-xs">
          dismiss
        </button>
      </div>
    </div>
  )
}

// ─── Dashboard ───

export default function AgentVisualizationDashboard() {
  const [settings, setSettings] = useState<Settings>({
    serverUrl: "http://localhost:8000/stream",
    showEdgeLabels: true,
    showMinimap: true,
    animateParticles: true,
  })
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)

  const { traceData, connected, error: streamError, traceHistory } = useTraceStream(settings.serverUrl)
  const isLive = connected && traceData !== null

  // Layout mode
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("freeform")

  // Toast notifications
  const [toasts, setToasts] = useState<Array<{ id: number; message: string; type: "error" | "warning" | "info" }>>([])
  const toastIdRef = useRef(0)
  const lastLoopRef = useRef(false)

  const addToast = useCallback((message: string, type: "error" | "warning" | "info" = "info") => {
    const id = ++toastIdRef.current
    setToasts((prev) => [...prev.slice(-4), { id, message, type }])
  }, [])

  const removeToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  // Loop detection toast
  useEffect(() => {
    if (traceData?.loop_detected && !lastLoopRef.current) {
      addToast(
        `Loop detected: ${traceData.loop_agents.join(" -> ")}`,
        "warning"
      )
    }
    lastLoopRef.current = traceData?.loop_detected ?? false
  }, [traceData?.loop_detected, traceData?.loop_agents, addToast])

  // Demo mode state
  const [demoNodes, setDemoNodes] = useState<GraphNode[]>(initialNodes)
  const [demoEdges, setDemoEdges] = useState<GraphEdge[]>(initialEdges)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const demoDuration = 6.0

  // Live timeline replay
  const [isLiveReplay, setIsLiveReplay] = useState(false)
  const [liveReplayIndex, setLiveReplayIndex] = useState(0)

  const panelMountedRef = useRef(false)
  const liveReplayIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Compute live graph from trace data (or replay snapshot)
  const activeTraceData = isLiveReplay && traceHistory.length > 0
    ? traceHistory[Math.min(liveReplayIndex, traceHistory.length - 1)]
    : traceData

  const liveGraph = useMemo(() => {
    if (!activeTraceData) return null
    return traceToGraph(activeTraceData, settings.showEdgeLabels)
  }, [activeTraceData, settings.showEdgeLabels])

  // Active nodes/edges: live or demo
  const nodes = isLive && liveGraph ? liveGraph.nodes : demoNodes
  const edges = isLive && liveGraph ? liveGraph.edges : demoEdges

  const selectedNode = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) || null : null

  const totalCost = isLive && activeTraceData ? activeTraceData.total_cost : nodes.reduce((sum, n) => sum + n.cost, 0)
  const totalTokens = isLive && activeTraceData ? activeTraceData.total_tokens : Math.round(totalCost * 1000)
  const latency = isLive && activeTraceData ? activeTraceData.latency_ms / 1000 : currentTime

  // Messages for details panel
  const currentMessages = activeTraceData?.messages ?? []

  // Timeline duration for live replay
  const liveDuration = traceHistory.length > 0 ? traceHistory.length - 1 : 0

  // Demo timeline updates
  useEffect(() => {
    if (isLive) return
    const applicableUpdates = demoTimeline.filter((t) => t.time <= currentTime)

    setDemoNodes((prev) => {
      const updated = [...prev]
      applicableUpdates.forEach((update) => {
        Object.entries(update.nodeUpdates).forEach(([nodeId, status]) => {
          const idx = updated.findIndex((n) => n.id === nodeId)
          if (idx !== -1) {
            updated[idx] = { ...updated[idx], status }
            if (status === "active" || status === "completed") {
              const baseCost = initialNodes.find((n) => n.id === nodeId)?.cost || 0
              updated[idx].cost = baseCost > 0 ? baseCost : Math.random() * 0.05
            }
          }
        })
      })
      return updated
    })

    setDemoEdges((prev) => {
      const updated = [...prev]
      applicableUpdates.forEach((update) => {
        Object.entries(update.edgeUpdates).forEach(([edgeId, status]) => {
          const idx = updated.findIndex((e) => e.id === edgeId)
          if (idx !== -1) {
            updated[idx] = { ...updated[idx], status }
          }
        })
      })
      return updated
    })
  }, [currentTime, isLive])

  // Playback loop (demo mode)
  useEffect(() => {
    if (!isPlaying || isLive) return

    const interval = setInterval(() => {
      setCurrentTime((prev) => {
        const next = prev + 0.1 * speed
        if (next >= demoDuration) {
          setIsPlaying(false)
          return demoDuration
        }
        return next
      })
    }, 100)

    return () => clearInterval(interval)
  }, [isPlaying, speed, isLive])

  useEffect(() => {
    panelMountedRef.current = true
    return () => {
      if (liveReplayIntervalRef.current) {
        clearInterval(liveReplayIntervalRef.current)
      }
    }
  }, [])

  const handleNodeSelect = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setIsPanelOpen(true)
      })
    })
  }, [])

  const handlePanelClose = useCallback(() => {
    setIsPanelOpen(false)
    setTimeout(() => setSelectedNodeId(null), 300)
  }, [])

  const handlePlayPause = useCallback(() => {
    if (isLive && traceHistory.length > 0) {
      // Clear any existing replay interval
      if (liveReplayIntervalRef.current) {
        clearInterval(liveReplayIntervalRef.current)
        liveReplayIntervalRef.current = null
      }

      // Live replay mode
      if (liveReplayIndex >= traceHistory.length - 1) {
        setLiveReplayIndex(0)
      }
      setIsLiveReplay(true)
      // Auto-advance replay
      liveReplayIntervalRef.current = setInterval(() => {
        setLiveReplayIndex((prev) => {
          if (prev >= traceHistory.length - 1) {
            if (liveReplayIntervalRef.current) {
              clearInterval(liveReplayIntervalRef.current)
              liveReplayIntervalRef.current = null
            }
            setIsLiveReplay(false)
            return traceHistory.length - 1
          }
          return prev + 1
        })
      }, 200)
      return
    }

    if (currentTime >= demoDuration) {
      setCurrentTime(0)
    }
    setIsPlaying((prev) => !prev)
  }, [currentTime, isLive, traceHistory, liveReplayIndex])

  const handleTimeChange = useCallback((time: number) => {
    if (isLive && traceHistory.length > 0) {
      const idx = liveDuration > 0
        ? Math.round((time / liveDuration) * (traceHistory.length - 1))
        : 0
      setLiveReplayIndex(Math.max(0, Math.min(idx, traceHistory.length - 1)))
      setIsLiveReplay(true)
      return
    }
    setCurrentTime(time)
    setIsPlaying(false)
  }, [isLive, traceHistory, liveDuration])

  const handleRefresh = useCallback(() => {
    setCurrentTime(0)
    setIsPlaying(false)
    setIsLiveReplay(false)
    setLiveReplayIndex(0)
    setDemoNodes(initialNodes)
    setDemoEdges(initialEdges)
  }, [])

  const handleNodesChange = useCallback((updatedNodes: GraphNode[]) => {
    if (!isLive) {
      setDemoNodes(updatedNodes)
    }
  }, [isLive])

  const handleLayoutToggle = useCallback(() => {
    if (isLive) {
      addToast("Layout changes disabled during live mode", "warning")
      return
    }

    const modes: LayoutMode[] = ["freeform", "hierarchical", "radial"]
    const nextIdx = (modes.indexOf(layoutMode) + 1) % modes.length
    const nextMode = modes[nextIdx]
    setLayoutMode(nextMode)
    addToast(`Layout: ${nextMode}`, "info")

    if (nextMode !== "freeform") {
      const arranged = applyLayout(demoNodes, nextMode)
      setDemoNodes(arranged)
    }
  }, [layoutMode, isLive, demoNodes, addToast])

  const handleExport = useCallback(() => {
    const exportData = {
      timestamp: new Date().toISOString(),
      nodes: nodes.map(({ id, name, type, status, cost }) => ({ id, name, type, status, cost })),
      edges: edges.map(({ id, source, target, status, label }) => ({ id, source, target, status, label })),
      metrics: { totalCost, totalTokens, latency },
      traceData: activeTraceData ?? null,
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `synkt-trace-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    addToast("Trace exported as JSON", "info")
  }, [nodes, edges, totalCost, totalTokens, latency, activeTraceData, addToast])

  return (
    <div className="flex h-screen flex-col bg-[#121212]">
      {/* Header */}
      <Header
        title={isLive ? "Agent Workflow (Live)" : "Agent Workflow (Demo)"}
        isConnected={connected}
        onRefresh={handleRefresh}
        onSettings={() => setIsSettingsOpen(true)}
        onLayoutToggle={handleLayoutToggle}
        onExport={handleExport}
      />

      {/* Loop detection banner */}
      {traceData?.loop_detected && (
        <div className="bg-red-900/80 border-b border-red-700 px-4 py-2 text-center text-sm text-red-200">
          Loop detected between agents: {traceData.loop_agents.join(" -> ")}
        </div>
      )}

      {/* Main content */}
      <div className="relative flex-1 flex flex-col">
        {/* Canvas */}
        <GraphCanvas
          nodes={nodes}
          edges={edges}
          selectedNodeId={selectedNodeId}
          onNodeSelect={handleNodeSelect}
          onNodesChange={handleNodesChange}
          showMinimap={settings.showMinimap}
          animateParticles={settings.animateParticles}
        />

        {/* Floating cost panel */}
        <div className="absolute top-6 right-6">
          <CostPanel
            totalCost={totalCost}
            tokens={totalTokens}
            latency={latency}
          />
        </div>

        {/* Toast notifications */}
        <div className="absolute top-6 left-1/2 -translate-x-1/2 z-30 flex flex-col gap-2">
          {toasts.map((toast) => (
            <Toast
              key={toast.id}
              message={toast.message}
              type={toast.type}
              onDismiss={() => removeToast(toast.id)}
            />
          ))}
        </div>

        {/* Details panel */}
        <DetailsPanel
          agent={selectedNode}
          isOpen={isPanelOpen}
          onClose={handlePanelClose}
          messages={currentMessages}
        />
      </div>

      {/* Timeline — always visible */}
      <Timeline
        currentTime={isLive ? liveReplayIndex : currentTime}
        duration={isLive ? liveDuration : demoDuration}
        isPlaying={isLive ? isLiveReplay : isPlaying}
        speed={speed}
        onTimeChange={handleTimeChange}
        onPlayPause={handlePlayPause}
        onSpeedChange={setSpeed}
      />

      {/* Settings panel */}
      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSettingsChange={setSettings}
      />
    </div>
  )
}
