import { TrainingMetrics, VideoStatus, EnvKey, EnvironmentInfo } from '@/types/metrics';

const API_BASE = '/api';

export async function getEnvironments(): Promise<{ environments: Record<EnvKey, EnvironmentInfo>; active: string }> {
  try {
    const resp = await fetch(`${API_BASE}/environments`);
    if (!resp.ok) return { environments: {} as Record<EnvKey, EnvironmentInfo>, active: 'hopper' };
    return await resp.json();
  } catch {
    return { environments: {} as Record<EnvKey, EnvironmentInfo>, active: 'hopper' };
  }
}

export async function getDistributedMetrics(env: EnvKey): Promise<TrainingMetrics[]> {
  try {
    const resp = await fetch(`${API_BASE}/metrics/distributed?env=${env}`);
    if (!resp.ok) return [];
    return await resp.json();
  } catch {
    return [];
  }
}

export async function getSingleMetrics(env: EnvKey): Promise<TrainingMetrics[]> {
  try {
    const resp = await fetch(`${API_BASE}/metrics/single?env=${env}`);
    if (!resp.ok) return [];
    return await resp.json();
  } catch {
    return [];
  }
}

export async function startDistributedTraining(env: EnvKey): Promise<{ status: string }> {
  const resp = await fetch(`${API_BASE}/training/distributed/start?env=${env}`, { method: 'POST' });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Server error: ${resp.status} - ${text.substring(0, 100)}`);
  }
  return resp.json();
}

export async function startSingleTraining(env: EnvKey): Promise<{ status: string }> {
  const resp = await fetch(`${API_BASE}/training/single/start?env=${env}`, { method: 'POST' });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Server error: ${resp.status} - ${text.substring(0, 100)}`);
  }
  return resp.json();
}

export async function stopTraining(env?: EnvKey): Promise<{ status: string }> {
  const url = env ? `${API_BASE}/training/stop?env=${env}` : `${API_BASE}/training/stop`;
  const resp = await fetch(url, { method: 'POST' });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Server error: ${resp.status} - ${text.substring(0, 100)}`);
  }
  return resp.json();
}

export async function generateVideos(env: EnvKey): Promise<{ status: string }> {
  const resp = await fetch(`${API_BASE}/videos/generate?env=${env}`, { method: 'POST' });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Server error: ${resp.status} - ${text.substring(0, 100)}`);
  }
  return resp.json();
}

export async function getVideoStatus(env: EnvKey): Promise<VideoStatus> {
  const resp = await fetch(`${API_BASE}/videos/status?env=${env}`);
  return resp.json();
}
