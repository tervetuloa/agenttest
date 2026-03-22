"use client"

import { memo } from "react"
import { X } from "lucide-react"
import { cn } from "@/lib/utils"
import { GlassButton } from "./glass-button"

export interface Settings {
  serverUrl: string
  showEdgeLabels: boolean
  showMinimap: boolean
  animateParticles: boolean
}

export interface SettingsPanelProps {
  isOpen: boolean
  onClose: () => void
  settings: Settings
  onSettingsChange: (settings: Settings) => void
  className?: string
}

export const SettingsPanel = memo(function SettingsPanel({
  isOpen,
  onClose,
  settings,
  onSettingsChange,
  className,
}: SettingsPanelProps) {
  if (!isOpen) return null

  const toggle = (key: keyof Settings) => {
    if (typeof settings[key] === "boolean") {
      onSettingsChange({ ...settings, [key]: !settings[key] })
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={cn(
          "fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50",
          "w-[380px] rounded-2xl border border-white/8",
          "bg-[rgba(26,26,26,0.95)] backdrop-blur-[20px]",
          "shadow-[0_24px_80px_rgba(0,0,0,0.6)]",
          className
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/8 p-5">
          <h2 className="text-[17px] font-semibold text-white">Settings</h2>
          <GlassButton size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </GlassButton>
        </div>

        {/* Settings */}
        <div className="p-5 space-y-4">
          {/* Server URL */}
          <div>
            <label className="text-[11px] uppercase tracking-wider text-white/40 block mb-2">
              Server URL
            </label>
            <input
              type="text"
              value={settings.serverUrl}
              onChange={(e) =>
                onSettingsChange({ ...settings, serverUrl: e.target.value })
              }
              className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white font-mono outline-none focus:border-accent-blue"
            />
          </div>

          {/* Toggles */}
          <div className="space-y-3">
            <ToggleRow
              label="Edge labels"
              description="Show handoff content on edges"
              checked={settings.showEdgeLabels}
              onChange={() => toggle("showEdgeLabels")}
            />
            <ToggleRow
              label="Minimap"
              description="Show navigation minimap"
              checked={settings.showMinimap}
              onChange={() => toggle("showMinimap")}
            />
            <ToggleRow
              label="Particle animations"
              description="Animate flow on active edges"
              checked={settings.animateParticles}
              onChange={() => toggle("animateParticles")}
            />
          </div>
        </div>
      </div>
    </>
  )
})

function ToggleRow({
  label,
  description,
  checked,
  onChange,
}: {
  label: string
  description: string
  checked: boolean
  onChange: () => void
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <p className="text-[14px] text-white">{label}</p>
        <p className="text-[12px] text-white/40">{description}</p>
      </div>
      <button
        role="switch"
        aria-checked={checked}
        onClick={onChange}
        className={cn(
          "relative h-6 w-11 rounded-full transition-colors",
          checked ? "bg-accent-blue" : "bg-white/10"
        )}
      >
        <span
          className={cn(
            "absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition-transform",
            checked && "translate-x-5"
          )}
        />
      </button>
    </div>
  )
}
