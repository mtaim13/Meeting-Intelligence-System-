import type { HealthResponse } from '../types'

interface HealthBadgeProps {
  health: HealthResponse | null
  error: string | null
}

export function HealthBadge({ health, error }: HealthBadgeProps) {
  if (error) {
    return (
      <div className="health health--down" title={error}>
        <span className="health__dot" />
        API offline
      </div>
    )
  }

  if (!health || health.status === 'loading') {
    return (
      <div className="health health--loading">
        <span className="health__dot" />
        Models loading…
      </div>
    )
  }

  return (
    <div className="health health--ok">
      <span className="health__dot" />
      Ready · {health.device ?? 'cpu'}
      {!health.bullet_fusion_available && (
        <span className="health__note"> · fusion off</span>
      )}
    </div>
  )
}
