export interface Utterance {
  speaker: string
  text: string
}

export interface MeetingRequest {
  meeting_id?: string | null
  utterances?: Utterance[] | null
  transcript?: string | null
}

export interface EvidenceChunk {
  speaker: string
  text: string
  importance: number
  decision_score: number
  is_decision: boolean
  is_action: boolean
}

export interface DecisionBullet {
  bullet: string
  confidence: number
  speakers: string
  evidence: string[]
}

export interface ActionBullet {
  action: string
  assignee: string | null
  deadline: string | null
  speakers: string
  evidence: string[]
}

export interface MeetingIntelligenceResponse {
  meeting_id: string
  summary: string
  key_decisions: DecisionBullet[]
  action_items: ActionBullet[]
  selected_evidence: EvidenceChunk[]
  bullet_fusion_used: boolean
}

export interface HealthResponse {
  status: 'ok' | 'loading'
  device: string | null
  bullet_fusion_available: boolean
}
