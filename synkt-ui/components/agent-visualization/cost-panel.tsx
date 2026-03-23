"use client"

import { memo, useEffect, useRef, useState } from "react"
import { DollarSign, Coins, Clock } from "lucide-react"
import { cn } from "@/lib/utils"

export interface CostPanelProps {
  totalCost: number
  tokens: number
  latency: number
  className?: string
}

function AnimatedNumber({ value, format }: { value: number; format: (n: number) => string }) {
  const [displayValue, setDisplayValue] = useState(value)
  const rafRef = useRef(0)

  useEffect(() => {
    const start = displayValue
    const end = value
    const duration = 500
    const startTime = performance.now()

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      const easeOut = 1 - Math.pow(1 - progress, 3)
      setDisplayValue(start + (end - start) * easeOut)

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate)
      }
    }

    rafRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(rafRef.current)
  }, [value])

  return <span className="animate-[count-up_0.3s_ease-out]">{format(displayValue)}</span>
}

export const CostPanel = memo(function CostPanel({
  totalCost,
  tokens,
  latency,
  className,
}: CostPanelProps) {
  return (
    <div
      className={cn(
        "w-[200px] rounded-2xl border border-white/8",
        "bg-[rgba(36,36,36,0.9)] backdrop-blur-[20px]",
        "p-5 shadow-[0_12px_48px_rgba(0,0,0,0.5)]",
        "animate-[fade-in-up_0.3s_ease-out]",
        className
      )}
    >
      <div className="flex items-center gap-2 text-white/60 text-[13px]">
        <DollarSign className="h-4 w-4" />
        <span>Total Cost</span>
      </div>
      <div className="mt-2 text-[24px] font-semibold text-white font-mono">
        $<AnimatedNumber value={totalCost} format={(n) => n.toFixed(3)} />
      </div>

      <div className="mt-4 h-px bg-white/10" />

      <div className="mt-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-white/60 text-[13px]">
            <Coins className="h-3.5 w-3.5" />
            <span>Tokens</span>
          </div>
          <span className="text-[13px] font-mono text-white">
            <AnimatedNumber value={tokens} format={(n) => Math.round(n).toLocaleString()} />
          </span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-white/60 text-[13px]">
            <Clock className="h-3.5 w-3.5" />
            <span>Latency</span>
          </div>
          <span className="text-[13px] font-mono text-white">
            <AnimatedNumber value={latency} format={(n) => n.toFixed(1)} />s
          </span>
        </div>
      </div>
    </div>
  )
})
