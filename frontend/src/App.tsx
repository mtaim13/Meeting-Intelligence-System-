import { useCallback, useEffect, useState } from 'react'
import { analyzeMeeting, fetchHealth } from './api'
import { DropZone } from './components/DropZone'
import { HealthBadge } from './components/HealthBadge'
import { ResultsBriefing } from './components/ResultsBriefing'
import { TranscriptPreview } from './components/TranscriptPreview'
import type { HealthResponse, MeetingIntelligenceResponse, MeetingRequest } from './types'

export default function App() {
  const [request, setRequest] = useState<MeetingRequest | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [meetingId, setMeetingId] = useState('')
  const [result, setResult] = useState<MeetingIntelligenceResponse | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const h = await fetchHealth()
        if (!cancelled) {
          setHealth(h)
          setHealthError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setHealth(null)
          setHealthError(err instanceof Error ? err.message : 'API unreachable')
        }
      }
    }
    void poll()
    const id = window.setInterval(poll, 8000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  const onLoaded = useCallback((payload: MeetingRequest, name: string) => {
    setRequest(payload)
    setFileName(name)
    setMeetingId(payload.meeting_id ?? '')
    setResult(null)
    setError(null)
  }, [])

  const onAnalyze = async () => {
    if (!request) return
    setAnalyzing(true)
    setError(null)
    try {
      const payload: MeetingRequest = {
        ...request,
        meeting_id: meetingId.trim() || request.meeting_id || null,
      }
      const data = await analyzeMeeting(payload)
      setResult(data)
    } catch (err) {
      setResult(null)
      setError(err instanceof Error ? err.message : 'Analysis failed.')
    } finally {
      setAnalyzing(false)
    }
  }

  const apiReady = health?.status === 'ok' && !healthError

  return (
    <div className="page">
      <div className="atmosphere" aria-hidden />

      <header className="topbar">
        <p className="brand">Meeting Intelligence</p>
        <HealthBadge health={health} error={healthError} />
      </header>

      <main className="workspace">
        <section className="panel panel--input">
          <div className="hero">
            <h1 className="hero__brand">Meeting Intelligence</h1>
            <p className="hero__lead">
              Turn a transcript into a narrative summary, key decisions, and action items.
            </p>
          </div>

          <DropZone onLoaded={onLoaded} disabled={analyzing} fileName={fileName} />

          {request && (
            <div className="input-controls animate-in">
              <label className="field">
                <span>Meeting ID</span>
                <input
                  type="text"
                  value={meetingId}
                  onChange={(e) => setMeetingId(e.target.value)}
                  placeholder="unnamed-meeting"
                  disabled={analyzing}
                />
              </label>

              <TranscriptPreview request={request} />

              <button
                type="button"
                className="btn-analyze"
                onClick={() => void onAnalyze()}
                disabled={analyzing || !apiReady}
              >
                {analyzing ? 'Analyzing…' : 'Analyze meeting'}
              </button>

              {!apiReady && (
                <p className="hint">
                  {healthError
                    ? 'Start the backend on port 8000 to analyze.'
                    : 'Waiting for models to finish loading…'}
                </p>
              )}
            </div>
          )}

          {error && (
            <p className="field-error animate-in" role="alert">
              {error}
            </p>
          )}
        </section>

        <section className="panel panel--results">
          {analyzing && (
            <div className="analyzing-state" aria-live="polite">
              <div className="analyzing-state__pulse" />
              <p>Running classifiers and generating the briefing…</p>
            </div>
          )}

          {!analyzing && result && <ResultsBriefing result={result} />}

          {!analyzing && !result && (
            <div className="results-placeholder">
              <p>Your briefing will appear here after analysis.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
