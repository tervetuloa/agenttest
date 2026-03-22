"use client"

import { memo, useState, useCallback, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { RotateCcw, Grid3X3, Maximize2, Search, X } from "lucide-react"
import { AgentNode, type AgentNodeProps } from "./agent-node"
import { AnimatedEdge, type EdgeStatus } from "./animated-edge"
import { GlassButton } from "./glass-button"

const NODE_WIDTH = 280
const NODE_HEIGHT = 100 // Approximate rendered height

export interface GraphNode extends AgentNodeProps {
  x: number
  y: number
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  status: EdgeStatus
  label?: string
}

export interface GraphCanvasProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  selectedNodeId: string | null
  onNodeSelect: (nodeId: string) => void
  onNodesChange?: (nodes: GraphNode[]) => void
  showMinimap?: boolean
  animateParticles?: boolean
  className?: string
}

/**
 * Calculate the point on the border of a rectangle closest to a target point,
 * plus the outward normal direction at that point.
 * This ensures edges connect from the edge of nodes, not the center.
 */
function getBorderPoint(
  nodeX: number, nodeY: number,
  targetX: number, targetY: number,
): { x: number; y: number; nx: number; ny: number } {
  const cx = nodeX + NODE_WIDTH / 2
  const cy = nodeY + NODE_HEIGHT / 2
  const dx = targetX - cx
  const dy = targetY - cy

  if (dx === 0 && dy === 0) return { x: cx, y: cy, nx: 0, ny: -1 }

  const halfW = NODE_WIDTH / 2
  const halfH = NODE_HEIGHT / 2

  // Check intersection with each edge of the rectangle
  const scaleX = dx !== 0 ? halfW / Math.abs(dx) : Infinity
  const scaleY = dy !== 0 ? halfH / Math.abs(dy) : Infinity
  const scale = Math.min(scaleX, scaleY)

  const bx = cx + dx * scale
  const by = cy + dy * scale

  // Determine outward normal based on which side was hit
  let nx = 0, ny = 0
  if (scaleX < scaleY) {
    nx = dx > 0 ? 1 : -1
  } else {
    ny = dy > 0 ? 1 : -1
  }

  return { x: bx, y: by, nx, ny }
}

export const GraphCanvas = memo(function GraphCanvas({
  nodes,
  edges,
  selectedNodeId,
  onNodeSelect,
  onNodesChange,
  showMinimap = true,
  animateParticles = true,
  className,
}: GraphCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [scale, setScale] = useState(1)
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null)
  const [nodeDragStart, setNodeDragStart] = useState({ x: 0, y: 0 })
  const [initialNodePos, setInitialNodePos] = useState({ x: 0, y: 0 })
  const [searchQuery, setSearchQuery] = useState("")
  const [isSearchOpen, setIsSearchOpen] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)

  // Filtered nodes based on search
  const matchingNodeIds = searchQuery
    ? new Set(
        nodes
          .filter((n) =>
            n.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            n.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
            n.type.toLowerCase().includes(searchQuery.toLowerCase())
          )
          .map((n) => n.id)
      )
    : null

  // Get edge connection points using border intersection
  const getEdgePoints = useCallback(
    (sourceId: string, targetId: string) => {
      const sourceNode = nodes.find((n) => n.id === sourceId)
      const targetNode = nodes.find((n) => n.id === targetId)
      if (!sourceNode || !targetNode) {
        return { sourceX: 0, sourceY: 0, targetX: 0, targetY: 0, sourceNx: 0, sourceNy: 0, targetNx: 0, targetNy: 0 }
      }

      const sourceCx = sourceNode.x + NODE_WIDTH / 2
      const sourceCy = sourceNode.y + NODE_HEIGHT / 2
      const targetCx = targetNode.x + NODE_WIDTH / 2
      const targetCy = targetNode.y + NODE_HEIGHT / 2

      const sourcePoint = getBorderPoint(sourceNode.x, sourceNode.y, targetCx, targetCy)
      const targetPoint = getBorderPoint(targetNode.x, targetNode.y, sourceCx, sourceCy)

      return {
        sourceX: sourcePoint.x,
        sourceY: sourcePoint.y,
        targetX: targetPoint.x,
        targetY: targetPoint.y,
        sourceNx: sourcePoint.nx,
        sourceNy: sourcePoint.ny,
        targetNx: targetPoint.nx,
        targetNy: targetPoint.ny,
      }
    },
    [nodes]
  )

  // Handle canvas panning
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement
    if (target.closest('[data-node-handle]')) {
      const nodeElement = target.closest('[data-node]')
      if (nodeElement) {
        const nodeId = nodeElement.getAttribute('data-node-id')
        if (nodeId) {
          const node = nodes.find(n => n.id === nodeId)
          if (node) {
            e.preventDefault()
            e.stopPropagation()
            setDraggingNodeId(nodeId)
            setNodeDragStart({ x: e.clientX, y: e.clientY })
            setInitialNodePos({ x: node.x, y: node.y })
          }
        }
      }
      return
    }

    if (target.closest('[data-node]')) return
    if (target.closest('[data-search]')) return
    e.preventDefault()
    setIsDragging(true)
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y })
  }, [offset, nodes])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (draggingNodeId && onNodesChange) {
      const deltaX = (e.clientX - nodeDragStart.x) / scale
      const deltaY = (e.clientY - nodeDragStart.y) / scale
      const updatedNodes = nodes.map(n =>
        n.id === draggingNodeId
          ? { ...n, x: initialNodePos.x + deltaX, y: initialNodePos.y + deltaY }
          : n
      )
      onNodesChange(updatedNodes)
      return
    }

    if (!isDragging) return
    setOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    })
  }, [isDragging, dragStart, draggingNodeId, nodeDragStart, initialNodePos, scale, nodes, onNodesChange])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
    setDraggingNodeId(null)
  }, [])

  // Handle zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setScale((prev) => Math.max(0.3, Math.min(3, prev * delta)))
  }, [])

  // Center graph on mount
  useEffect(() => {
    if (canvasRef.current && nodes.length > 0) {
      const rect = canvasRef.current.getBoundingClientRect()
      const avgX = nodes.reduce((sum, n) => sum + n.x, 0) / nodes.length
      const avgY = nodes.reduce((sum, n) => sum + n.y, 0) / nodes.length
      setOffset({
        x: rect.width / 2 - avgX - NODE_WIDTH / 2,
        y: rect.height / 2 - avgY - NODE_HEIGHT / 2,
      })
    }
  }, [])

  // Focus search input when opened
  useEffect(() => {
    if (isSearchOpen && searchInputRef.current) {
      searchInputRef.current.focus()
    }
  }, [isSearchOpen])

  // Keyboard shortcut: Ctrl/Cmd+F to open search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "f") {
        e.preventDefault()
        setIsSearchOpen(true)
      }
      if (e.key === "Escape") {
        setIsSearchOpen(false)
        setSearchQuery("")
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [])

  // Layout functions
  const resetView = useCallback(() => {
    if (canvasRef.current && nodes.length > 0) {
      const rect = canvasRef.current.getBoundingClientRect()
      const avgX = nodes.reduce((sum, n) => sum + n.x, 0) / nodes.length
      const avgY = nodes.reduce((sum, n) => sum + n.y, 0) / nodes.length
      setOffset({
        x: rect.width / 2 - avgX - NODE_WIDTH / 2,
        y: rect.height / 2 - avgY - NODE_HEIGHT / 2,
      })
      setScale(1)
    }
  }, [nodes])

  const arrangeGrid = useCallback(() => {
    if (!onNodesChange) return
    const cols = Math.ceil(Math.sqrt(nodes.length))
    const spacing = { x: 340, y: 160 }
    const updatedNodes = nodes.map((node, i) => ({
      ...node,
      x: (i % cols) * spacing.x,
      y: Math.floor(i / cols) * spacing.y,
    }))
    onNodesChange(updatedNodes)
    setTimeout(resetView, 50)
  }, [nodes, onNodesChange, resetView])

  const fitToView = useCallback(() => {
    if (!canvasRef.current || nodes.length === 0) return
    const rect = canvasRef.current.getBoundingClientRect()
    const minX = Math.min(...nodes.map(n => n.x))
    const maxX = Math.max(...nodes.map(n => n.x + NODE_WIDTH))
    const minY = Math.min(...nodes.map(n => n.y))
    const maxY = Math.max(...nodes.map(n => n.y + NODE_HEIGHT))

    const contentWidth = maxX - minX + 100
    const contentHeight = maxY - minY + 100
    const newScale = Math.min(
      rect.width / contentWidth,
      rect.height / contentHeight,
      1.5
    )

    setScale(Math.max(0.3, newScale))
    setOffset({
      x: (rect.width - contentWidth * newScale) / 2 - minX * newScale,
      y: (rect.height - contentHeight * newScale) / 2 - minY * newScale,
    })
  }, [nodes])

  // Minimap calculations
  const minimapSize = { w: 160, h: 100 }
  const allX = nodes.map(n => n.x)
  const allY = nodes.map(n => n.y)
  const graphBounds = nodes.length > 0
    ? {
        minX: Math.min(...allX) - 50,
        maxX: Math.max(...allX) + NODE_WIDTH + 50,
        minY: Math.min(...allY) - 50,
        maxY: Math.max(...allY) + NODE_HEIGHT + 50,
      }
    : { minX: 0, maxX: 800, minY: 0, maxY: 600 }
  const graphW = graphBounds.maxX - graphBounds.minX
  const graphH = graphBounds.maxY - graphBounds.minY
  const minimapScale = Math.min(minimapSize.w / graphW, minimapSize.h / graphH)

  return (
    <div
      ref={canvasRef}
      className={cn(
        "relative flex-1 overflow-hidden bg-[#121212] cursor-grab",
        isDragging && "cursor-grabbing select-none",
        draggingNodeId && "cursor-grabbing select-none",
        className
      )}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    >
      {/* Grid background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)
          `,
          backgroundSize: `${40 * scale}px ${40 * scale}px`,
          backgroundPosition: `${offset.x}px ${offset.y}px`,
        }}
      />

      {/* Graph content layer */}
      <div
        className="absolute"
        style={{
          transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
          transformOrigin: "0 0",
        }}
      >
        {/* Edges SVG */}
        <svg
          className="absolute pointer-events-none overflow-visible"
          style={{ left: 0, top: 0, width: 1, height: 1, overflow: "visible" }}
        >
          {edges.map((edge) => {
            const pts = getEdgePoints(edge.source, edge.target)
            return (
              <AnimatedEdge
                key={edge.id}
                id={edge.id}
                sourceX={pts.sourceX}
                sourceY={pts.sourceY}
                targetX={pts.targetX}
                targetY={pts.targetY}
                sourceNx={pts.sourceNx}
                sourceNy={pts.sourceNy}
                targetNx={pts.targetNx}
                targetNy={pts.targetNy}
                status={edge.status}
                label={edge.label}
                animateParticles={animateParticles}
              />
            )
          })}
        </svg>

        {/* Nodes */}
        {nodes.map((node) => {
          const dimmed = matchingNodeIds !== null && !matchingNodeIds.has(node.id)
          return (
            <div
              key={node.id}
              data-node
              data-node-id={node.id}
              className={cn(
                "absolute",
                draggingNodeId === node.id ? "z-50" : "transition-shadow duration-200",
                dimmed && "opacity-20 pointer-events-none"
              )}
              style={{
                left: node.x,
                top: node.y,
                willChange: draggingNodeId === node.id ? "transform" : "auto"
              }}
            >
              <AgentNode
                {...node}
                isSelected={selectedNodeId === node.id}
                onClick={() => onNodeSelect(node.id)}
                draggable={!!onNodesChange}
              />
            </div>
          )
        })}
      </div>

      {/* Search bar */}
      {isSearchOpen && (
        <div data-search className="absolute top-4 left-1/2 -translate-x-1/2 z-30">
          <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-[rgba(36,36,36,0.95)] backdrop-blur-xl px-3 py-2 shadow-lg">
            <Search className="h-4 w-4 text-white/40" />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search agents..."
              className="bg-transparent text-sm text-white outline-none placeholder:text-white/30 w-[200px]"
            />
            {searchQuery && (
              <span className="text-xs text-white/40">
                {matchingNodeIds?.size ?? 0} found
              </span>
            )}
            <button
              onClick={() => { setIsSearchOpen(false); setSearchQuery("") }}
              className="p-0.5 rounded hover:bg-white/10"
            >
              <X className="h-3.5 w-3.5 text-white/40" />
            </button>
          </div>
        </div>
      )}

      {/* Minimap */}
      {showMinimap && nodes.length > 2 && (
        <div className="absolute bottom-4 right-4 rounded-xl border border-white/10 bg-[rgba(26,26,26,0.9)] backdrop-blur-xl p-2 shadow-lg">
          <svg width={minimapSize.w} height={minimapSize.h} className="block">
            {/* Edges */}
            {edges.map((edge) => {
              const s = nodes.find(n => n.id === edge.source)
              const t = nodes.find(n => n.id === edge.target)
              if (!s || !t) return null
              return (
                <line
                  key={edge.id}
                  x1={(s.x + NODE_WIDTH / 2 - graphBounds.minX) * minimapScale}
                  y1={(s.y + NODE_HEIGHT / 2 - graphBounds.minY) * minimapScale}
                  x2={(t.x + NODE_WIDTH / 2 - graphBounds.minX) * minimapScale}
                  y2={(t.y + NODE_HEIGHT / 2 - graphBounds.minY) * minimapScale}
                  stroke="rgba(255,255,255,0.15)"
                  strokeWidth={1}
                />
              )
            })}
            {/* Nodes */}
            {nodes.map((node) => {
              const color = node.status === "active" ? "#10b981"
                : node.status === "error" ? "#ef4444"
                : node.status === "completed" ? "#3b82f6"
                : "rgba(255,255,255,0.3)"
              return (
                <rect
                  key={node.id}
                  x={(node.x - graphBounds.minX) * minimapScale}
                  y={(node.y - graphBounds.minY) * minimapScale}
                  width={NODE_WIDTH * minimapScale}
                  height={NODE_HEIGHT * minimapScale}
                  rx={2}
                  fill={color}
                  opacity={0.7}
                />
              )
            })}
            {/* Viewport indicator */}
            {canvasRef.current && (
              <rect
                x={(-offset.x / scale - graphBounds.minX) * minimapScale}
                y={(-offset.y / scale - graphBounds.minY) * minimapScale}
                width={(canvasRef.current.getBoundingClientRect().width / scale) * minimapScale}
                height={(canvasRef.current.getBoundingClientRect().height / scale) * minimapScale}
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth={1}
                rx={2}
              />
            )}
          </svg>
        </div>
      )}

      {/* Layout controls - Bottom left */}
      <div className="absolute bottom-4 left-4 flex items-center gap-2">
        <div className="flex items-center gap-1 rounded-xl border border-white/10 bg-[rgba(36,36,36,0.9)] backdrop-blur-xl p-1">
          <GlassButton size="sm" onClick={resetView} title="Reset View">
            <RotateCcw className="h-4 w-4" />
          </GlassButton>
          <GlassButton size="sm" onClick={arrangeGrid} title="Arrange Grid">
            <Grid3X3 className="h-4 w-4" />
          </GlassButton>
          <GlassButton size="sm" onClick={fitToView} title="Fit to View">
            <Maximize2 className="h-4 w-4" />
          </GlassButton>
          <GlassButton size="sm" onClick={() => setIsSearchOpen(true)} title="Search (Ctrl+F)">
            <Search className="h-4 w-4" />
          </GlassButton>
        </div>
        <div className="text-[13px] font-mono text-white/40 px-2">
          {Math.round(scale * 100)}%
        </div>
      </div>
    </div>
  )
})
