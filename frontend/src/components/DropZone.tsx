import { useCallback, useId, useRef, useState } from 'react'
import type { MeetingRequest } from '../types'
import { readMeetingFile } from '../lib/parseMeetingJson'

interface DropZoneProps {
  onLoaded: (request: MeetingRequest, fileName: string) => void
  disabled?: boolean
  fileName: string | null
}

export function DropZone({ onLoaded, disabled, fileName }: DropZoneProps) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file) return
      setError(null)
      try {
        const request = await readMeetingFile(file)
        onLoaded(request, file.name)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Could not read file.')
      }
    },
    [onLoaded],
  )

  return (
    <div className="drop-zone-wrap">
      <label
        htmlFor={inputId}
        className={`drop-zone${dragging ? ' is-dragging' : ''}${fileName ? ' has-file' : ''}`}
        onDragEnter={(e) => {
          e.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (disabled) return
          void handleFile(e.dataTransfer.files[0])
        }}
      >
        <span className="drop-zone__glyph" aria-hidden>
          {fileName ? '✓' : '↓'}
        </span>
        <span className="drop-zone__title">
          {fileName ? fileName : 'Drop a meeting JSON file'}
        </span>
        <span className="drop-zone__hint">
          {fileName
            ? 'Drop another file to replace, or browse'
            : 'or click to browse · utterances or transcript'}
        </span>
        <input
          id={inputId}
          ref={inputRef}
          type="file"
          accept=".json,application/json"
          disabled={disabled}
          className="visually-hidden"
          onChange={(e) => {
            void handleFile(e.target.files?.[0])
            e.target.value = ''
          }}
        />
      </label>
      {error && <p className="field-error" role="alert">{error}</p>}
    </div>
  )
}
