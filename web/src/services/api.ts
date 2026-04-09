import { TrainingMetrics, VideoStatus } from '@/types/metrics';

const API_BASE = '/api';

export async function getDistributedMetrics(): Promise<TrainingMetrics[]> {
  try {
    const resp = await fetch(`${API_BASE}/metrics/distributed`);
    if (!resp.ok) return [];
    return await resp.json();
  } catch {
    return [];
  }
}

export async function getSingleMetrics(): Promise<TrainingMetrics[]> {
  try {
    const resp = await fetch(`${API_BASE}/metrics/single`);
    if (!resp.ok) return [];
    return await resp.json();
  } catch {
    return [];
  }
}

export async function startDistributedTraining(): Promise<{ status: string }> {
  const resp = await fetch(`${API_BASE}/training/distributed/start`, { method: 'POST' });
  return resp.json();
}

export async function startSingleTraining(): Promise<{ status: string }> {
  const resp = await fetch(`${API_BASE}/training/single/start`, { method: 'POST' });
  return resp.json();
}

export async function stopTraining(): Promise<{ status: string }> {
  const resp = await fetch(`${API_BASE}/training/stop`, { method: 'POST' });
  return resp.json();
}

export async function generateVideos(): Promise<{ status: string }> {
  const resp = await fetch(`${API_BASE}/videos/generate`, { method: 'POST' });
  return resp.json();
}

export async function getVideoStatus(): Promise<VideoStatus> {
  const resp = await fetch(`${API_BASE}/videos/status`);
  return resp.json();
}
