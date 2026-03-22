"use client"

import { memo } from "react"
import { Activity, Settings, RefreshCw, Layout, Download } from "lucide-react"
import { cn } from "@/lib/utils"
import { GlassButton } from "./glass-button"

export interface HeaderProps {
  title?: string
  onRefresh?: () => void
  onSettings?: () => void
  onLayoutToggle?: () => void
  onExport?: () => void
  isConnected?: boolean
  className?: string
}

export const Header = memo(function Header({
  title = "Agent Graph",
  onRefresh,
  onSettings,
  onLayoutToggle,
  onExport,
  isConnected = true,
  className,
}: HeaderProps) {
  return (
    <header
      className={cn(
        "flex h-[60px] items-center justify-between border-b border-white/8",
        "bg-[rgba(26,26,26,0.9)] backdrop-blur-[20px] px-6",
        className
      )}
    >
      {/* Left: Logo + Title */}
      <div className="flex items-center gap-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-accent-blue to-accent-purple">
          <Activity className="h-5 w-5 text-white" />
        </div>
        <div className="flex items-center gap-3">
          <h1 className="text-[17px] font-semibold text-white">{title}</h1>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "h-2 w-2 rounded-full",
                isConnected ? "bg-accent-green animate-pulse" : "bg-accent-red"
              )}
            />
            <span className="text-[13px] text-white/50">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>
      </div>

      {/* Right: Controls */}
      <div className="flex items-center gap-2">
        {onLayoutToggle && (
          <GlassButton size="sm" onClick={onLayoutToggle}>
            <Layout className="h-4 w-4" />
            <span className="hidden sm:inline">Layout</span>
          </GlassButton>
        )}

        {onExport && (
          <GlassButton size="sm" onClick={onExport}>
            <Download className="h-4 w-4" />
            <span className="hidden sm:inline">Export</span>
          </GlassButton>
        )}

        <GlassButton size="sm" onClick={onRefresh}>
          <RefreshCw className="h-4 w-4" />
          <span className="hidden sm:inline">Refresh</span>
        </GlassButton>

        {onSettings && (
          <GlassButton size="sm" onClick={onSettings}>
            <Settings className="h-4 w-4" />
          </GlassButton>
        )}
      </div>
    </header>
  )
})
