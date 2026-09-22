import type { HealthResponse, MeetingIntelligenceResponse, MeetingRequest } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) {
    throw new Error(`Health check failed (${res.status})`)
  }
  return res.json()
}

export async function analyzeMeeting(
  payload: MeetingRequest,
): Promise<MeetingIntelligenceResponse> {
  const res = await fetch(`${API_BASE}/meetings/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body?.detail === 'string') detail = body.detail
      else if (Array.isArray(body?.detail)) {
        detail = body.detail
          .map((d: { msg?: string }) => d.msg)
          .filter(Boolean)
          .join('; ')
      }
    } catch {
      /* keep default */
    }
    throw new Error(detail)
  }

  return res.json()
}
