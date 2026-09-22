import type { MeetingRequest, Utterance } from '../types'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseUtterances(value: unknown): Utterance[] | null {
  if (!Array.isArray(value) || value.length === 0) return null

  const utterances: Utterance[] = []
  for (const item of value) {
    if (!isRecord(item) || typeof item.text !== 'string' || !item.text.trim()) {
      throw new Error('Each utterance needs a non-empty "text" field.')
    }
    utterances.push({
      speaker: typeof item.speaker === 'string' && item.speaker.trim()
        ? item.speaker.trim()
        : 'Speaker',
      text: item.text.trim(),
    })
  }
  return utterances
}

/** Validate and normalize a parsed JSON file into a MeetingRequest. */
export function parseMeetingJson(raw: unknown): MeetingRequest {
  if (!isRecord(raw)) {
    throw new Error('JSON root must be an object.')
  }

  const meeting_id =
    typeof raw.meeting_id === 'string' && raw.meeting_id.trim()
      ? raw.meeting_id.trim()
      : null

  const utterances = parseUtterances(raw.utterances)
  const transcript =
    typeof raw.transcript === 'string' && raw.transcript.trim()
      ? raw.transcript.trim()
      : null

  if (!utterances && !transcript) {
    throw new Error(
      'Provide either a non-empty "utterances" array or a non-empty "transcript" string.',
    )
  }

  return { meeting_id, utterances, transcript }
}

export async function readMeetingFile(file: File): Promise<MeetingRequest> {
  if (!file.name.toLowerCase().endsWith('.json') && file.type !== 'application/json') {
    throw new Error('Please select a .json file.')
  }

  const text = await file.text()
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    throw new Error('File is not valid JSON.')
  }

  return parseMeetingJson(raw)
}
