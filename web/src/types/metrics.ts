export interface TrainingMetrics {
  step: number;
  elapsed_sec: number;
  total_steps: number;
  sps: number;
  episodes: number;
  avg_return: number;
  buffer_size: number;
  loss: number;
  policy_loss: number;
  value_loss: number;
  entropy: number;
  ratio: number;
  approx_kl: number;
  clip_fraction: number;
  explained_var: number;
  grad_norm: number;
  lr: number;
}

export type MetricsKey = keyof TrainingMetrics;

export interface WebSocketMessage {
  type: 'metrics';
  env: string;
  distributed: TrainingMetrics[];
  single: TrainingMetrics[];
}

export interface VideoStatus {
  status: 'idle' | 'generating' | 'completed' | 'error';
  progress?: number;
  error?: string;
}

export type EnvKey = 'hopper' | 'walker2d' | 'halfcheetah';

export interface EnvironmentInfo {
  name: string;
  description: string;
  difficulty: number;
}
