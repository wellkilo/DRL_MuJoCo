import { create } from 'zustand';
import { TrainingMetrics, VideoStatus } from '@/types/metrics';

interface TrainingState {
  isRunning: boolean;
  isLoading: boolean;
  error: string | null;
  distributedMetrics: TrainingMetrics[];
  singleMetrics: TrainingMetrics[];
  videoStatus: VideoStatus;
  activeTab: 'distributed' | 'single' | 'comparison' | 'video';
  setIsRunning: (running: boolean) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setDistributedMetrics: (metrics: TrainingMetrics[]) => void;
  setSingleMetrics: (metrics: TrainingMetrics[]) => void;
  setVideoStatus: (status: VideoStatus) => void;
  setActiveTab: (tab: 'distributed' | 'single' | 'comparison' | 'video') => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  isRunning: false,
  isLoading: false,
  error: null,
  distributedMetrics: [],
  singleMetrics: [],
  videoStatus: { status: 'idle' },
  activeTab: 'distributed',
  setIsRunning: (running) => set({ isRunning: running }),
  setIsLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setDistributedMetrics: (metrics) => set({ distributedMetrics: metrics }),
  setSingleMetrics: (metrics) => set({ singleMetrics: metrics }),
  setVideoStatus: (status) => set({ videoStatus: status }),
  setActiveTab: (tab) => set({ activeTab: tab }),
}));
