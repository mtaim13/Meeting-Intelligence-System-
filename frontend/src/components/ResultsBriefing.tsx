import { useState } from 'react'
import type {
  ActionBullet,
  DecisionBullet,
  EvidenceChunk,
  MeetingIntelligenceResponse,
} from '../types'

interface ResultsBriefingProps {
  result: MeetingIntelligenceResponse
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100)
  return (
    <div className="confidence" aria-label={`Confidence ${pct}%`}>
      <div className="confidence__track">
        <div className="confidence__fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="confidence__label">{pct}%</span>
    </div>
  )
}

function EvidenceList({ items }: { items: string[] }) {
  if (!items.length) return null
  return (
    <ul className="evidence-quotes">
      {items.map((text, i) => (
        <li key={i}>“{text}”</li>
      ))}
    </ul>
  )
}

function DecisionItem({ item }: { item: DecisionBullet }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="result-item">
      <button
        type="button"
        className="result-item__toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="result-item__body">
          <span className="result-item__title">{item.bullet}</span>
          <span className="result-item__meta">
            {item.speakers}
            <ConfidenceBar value={item.confidence} />
          </span>
        </span>
        <span className="result-item__chevron" aria-hidden>
          {open ? '−' : '+'}
        </span>
      </button>
      {open && <EvidenceList items={item.evidence} />}
    </li>
  )
}

function ActionItem({ item }: { item: ActionBullet }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="result-item result-item--action">
      <button
        type="button"
        className="result-item__toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="result-item__body">
          <span className="result-item__title">{item.action}</span>
          <span className="result-item__meta result-item__meta--tags">
            {item.assignee && (
              <span className="tag">
                <em>Assignee</em> {item.assignee}
              </span>
            )}
            {item.deadline && (
              <span className="tag">
                <em>Due</em> {item.deadline}
              </span>
            )}
            <span className="tag tag--muted">{item.speakers}</span>
          </span>
        </span>
        <span className="result-item__chevron" aria-hidden>
          {open ? '−' : '+'}
        </span>
      </button>
      {open && <EvidenceList items={item.evidence} />}
    </li>
  )
}

function EvidenceRow({ chunk }: { chunk: EvidenceChunk }) {
  return (
    <li className="evidence-row">
      <div className="evidence-row__head">
        <span className="preview__speaker">{chunk.speaker}</span>
        <span className="evidence-row__scores">
          {chunk.is_decision && <span className="flag flag--decision">decision</span>}
          {chunk.is_action && <span className="flag flag--action">action</span>}
          <span>imp {(chunk.importance * 100).toFixed(0)}%</span>
          <span>dec {(chunk.decision_score * 100).toFixed(0)}%</span>
        </span>
      </div>
      <p>{chunk.text}</p>
    </li>
  )
}

export function ResultsBriefing({ result }: ResultsBriefingProps) {
  return (
    <article className="briefing" aria-live="polite">
      <header className="briefing__header">
        <p className="briefing__kicker">Meeting briefing</p>
        <h2 className="briefing__id">{result.meeting_id}</h2>
        {!result.bullet_fusion_used && (
          <p className="briefing__note">
            Bullet fusion model unavailable — decisions and actions use joined source sentences.
          </p>
        )}
      </header>

      <section className="briefing__section briefing__section--summary">
        <h3>Summary</h3>
        <p className="summary-text">{result.summary}</p>
      </section>

      <section className="briefing__section">
        <h3>
          Key decisions
          <span className="count">{result.key_decisions.length}</span>
        </h3>
        {result.key_decisions.length === 0 ? (
          <p className="empty">No key decisions detected.</p>
        ) : (
          <ul className="result-list">
            {result.key_decisions.map((item, i) => (
              <DecisionItem key={i} item={item} />
            ))}
          </ul>
        )}
      </section>

      <section className="briefing__section">
        <h3>
          Action items
          <span className="count">{result.action_items.length}</span>
        </h3>
        {result.action_items.length === 0 ? (
          <p className="empty">No action items detected.</p>
        ) : (
          <ul className="result-list">
            {result.action_items.map((item, i) => (
              <ActionItem key={i} item={item} />
            ))}
          </ul>
        )}
      </section>

      <section className="briefing__section">
        <h3>
          Selected evidence
          <span className="count">{result.selected_evidence.length}</span>
        </h3>
        <p className="section-lead">
          Chunks scored and passed to the summarizer for explainability.
        </p>
        {result.selected_evidence.length === 0 ? (
          <p className="empty">No evidence selected.</p>
        ) : (
          <ul className="evidence-list">
            {result.selected_evidence.map((chunk, i) => (
              <EvidenceRow key={i} chunk={chunk} />
            ))}
          </ul>
        )}
      </section>
    </article>
  )
}
