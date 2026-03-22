"use client"

import { useState, useCallback, useEffect, useRef, useMemo } from "react"
import { Header } from "@/components/agent-visualization/header"
import { GraphCanvas, type GraphNode, type GraphEdge } from "@/components/agent-visualization/graph-canvas"
import { CostPanel } from "@/components/agent-visualization/cost-panel"
import { Timeline } from "@/components/agent-visualization/timeline"
import { DetailsPanel } from "@/components/agent-visualization/details-panel"
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

const timeline: { time: number; nodeUpdates: Record<string, AgentStatus>; edgeUpdates: Record<string, EdgeStatus> }[] = [
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

// ─── Helpers to convert live trace data to graph nodes/edges ───

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

function traceToGraph(trace: TraceData): { nodes: GraphNode[]; edges: GraphEdge[] } {
  // Build nodes from agents, arrange in a grid
  const agents = trace.agents
  const cols = Math.max(Math.ceil(Math.sqrt(agents.length)), 2)
  const nodes: GraphNode[] = agents.map((a, i) => ({
    id: a.name,
    name: a.name,
    type: agentTypeMap[a.type] || "executor",
    status: mapAgentStatus(a.status),
    cost: a.cost,
    x: 100 + (i % cols) * 300,
    y: 80 + Math.floor(i / cols) * 200,
  }))

  // Build edges from messages (deduplicate)
  const edgeSet = new Set<string>()
  const edges: GraphEdge[] = []
  for (const msg of trace.messages) {
    const key = `${msg.from_agent}->${msg.to_agent}`
    if (edgeSet.has(key)) continue
    edgeSet.add(key)

    // Determine edge status
    let status: EdgeStatus = "active"
    if (trace.loop_detected && trace.loop_agents.includes(msg.from_agent) && trace.loop_agents.includes(msg.to_agent)) {
      status = "loop"
    }

    edges.push({
      id: `e-${edges.length}`,
      source: msg.from_agent,
      target: msg.to_agent,
      status,
    })
  }

  return { nodes, edges }
}

// ─── Dashboard ───

export default function AgentVisualizationDashboard() {
  const { traceData, connected, error: streamError } = useTraceStream()
  const isLive = connected && traceData !== null

  // Demo mode state
  const [demoNodes, setDemoNodes] = useState<GraphNode[]>(initialNodes)
  const [demoEdges, setDemoEdges] = useState<GraphEdge[]>(initialEdges)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const duration = 6.0

  const panelMountedRef = useRef(false)

  // Compute live graph from trace data
  const liveGraph = useMemo(() => {
    if (!traceData) return null
    return traceToGraph(traceData)
  }, [traceData])

  // Active nodes/edges: live or demo
  const nodes = isLive && liveGraph ? liveGraph.nodes : demoNodes
  const edges = isLive && liveGraph ? liveGraph.edges : demoEdges

  const selectedNode = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) || null : null

  const totalCost = isLive && traceData ? traceData.total_cost : nodes.reduce((sum, n) => sum + n.cost, 0)
  const totalTokens = isLive && traceData ? traceData.total_tokens : Math.round(totalCost * 1000)
  const latency = isLive && traceData ? traceData.latency_ms / 1000 : currentTime

  // Demo timeline updates
  useEffect(() => {
    if (isLive) return
    const applicableUpdates = timeline.filter((t) => t.time <= currentTime)

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
        if (next >= duration) {
          setIsPlaying(false)
          return duration
        }
        return next
      })
    }, 100)

    return () => clearInterval(interval)
  }, [isPlaying, speed, isLive])

  useEffect(() => {
    panelMountedRef.current = true
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
    if (currentTime >= duration) {
      setCurrentTime(0)
    }
    setIsPlaying((prev) => !prev)
  }, [currentTime])

  const handleTimeChange = useCallback((time: number) => {
    setCurrentTime(time)
    setIsPlaying(false)
  }, [])

  const handleRefresh = useCallback(() => {
    setCurrentTime(0)
    setIsPlaying(false)
    setDemoNodes(initialNodes)
    setDemoEdges(initialEdges)
  }, [])

  const handleNodesChange = useCallback((updatedNodes: GraphNode[]) => {
    if (!isLive) {
      setDemoNodes(updatedNodes)
    }
  }, [isLive])

  return (
    <div className="flex h-screen flex-col bg-[#121212]">
      {/* Header */}
      <Header
        title={isLive ? "Agent Workflow (Live)" : "Agent Workflow (Demo)"}
        isConnected={connected}
        onRefresh={handleRefresh}
      />

      {/* Loop detection banner */}
      {traceData?.loop_detected && (
        <div className="bg-red-900/80 border-b border-red-700 px-4 py-2 text-center text-sm text-red-200">
          Loop detected between agents: {traceData.loop_agents.join(" → ")}
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
        />

        {/* Floating cost panel */}
        <div className="absolute top-6 right-6">
          <CostPanel
            totalCost={totalCost}
            tokens={totalTokens}
            latency={latency}
          />
        </div>

        {/* Details panel */}
        <DetailsPanel
          agent={selectedNode}
          isOpen={isPanelOpen}
          onClose={handlePanelClose}
        />
      </div>

      {/* Timeline (demo mode only, hidden when live) */}
      {!isLive && (
        <Timeline
          currentTime={currentTime}
          duration={duration}
          isPlaying={isPlaying}
          speed={speed}
          onTimeChange={handleTimeChange}
          onPlayPause={handlePlayPause}
          onSpeedChange={setSpeed}
        />
      )}
    </div>
  )
}
