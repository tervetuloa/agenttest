import { useEffect, useState, useCallback } from "react"

export interface AgentData {
  name: string
  status: "idle" | "active" | "complete" | "error"
  cost: number
  tokens: number
  type: string
}

export interface MessageData {
  from_agent: string
  to_agent: string
  timestamp: number
  content: string
}

export interface TraceData {
  agents: AgentData[]
  messages: MessageData[]
  total_cost: number
  total_tokens: number
  latency_ms: number
  timestamp: number
  loop_detected: boolean
  loop_agents: string[]
}

export function useTraceStream(url: string = "http://localhost:8000/stream") {
  const [traceData, setTraceData] = useState<TraceData | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const connect = useCallback(() => {
    let eventSource: EventSource | null = null

    try {
      eventSource = new EventSource(url)

      eventSource.onopen = () => {
        setConnected(true)
        setError(null)
      }

      eventSource.addEventListener("trace", (event) => {
        try {
          const data = JSON.parse(event.data) as TraceData
          setTraceData(data)
        } catch {
          // Ignore parse errors
        }
      })

      eventSource.onerror = () => {
        setConnected(false)
        setError(
          "Connection lost. Make sure server is running: python -m synkt.server"
        )
      }
    } catch {
      setError("Failed to connect to server")
    }

    return () => {
      eventSource?.close()
    }
  }, [url])

  useEffect(() => {
    const cleanup = connect()
    return cleanup
  }, [connect])

  return { traceData, connected, error }
}
