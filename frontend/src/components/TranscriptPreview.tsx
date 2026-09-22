import type { MeetingRequest } from '../types'

interface TranscriptPreviewProps {
  request: MeetingRequest
}

export function TranscriptPreview({ request }: TranscriptPreviewProps) {
  const utterances = request.utterances

  if (utterances?.length) {
    return (
      <div className="preview">
        <div className="preview__meta">
          <span>{utterances.length} turns</span>
          {request.meeting_id && <span>{request.meeting_id}</span>}
        </div>
        <ul className="preview__list">
          {utterances.map((u, i) => (
            <li key={`${u.speaker}-${i}`} className="preview__turn">
              <span className="preview__speaker">{u.speaker}</span>
              <p className="preview__text">{u.text}</p>
            </li>
          ))}
        </ul>
      </div>
    )
  }

  if (request.transcript) {
    return (
      <div className="preview">
        <div className="preview__meta">
          <span>Raw transcript</span>
          {request.meeting_id && <span>{request.meeting_id}</span>}
        </div>
        <p className="preview__transcript">{request.transcript}</p>
      </div>
    )
  }

  return null
}
