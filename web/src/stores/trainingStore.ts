import { create } from 'zustand';
import { TrainingMetrics, VideoStatus } from '@/types/metrics';

interface TrainingState {
  isRunning: boolean;
  distributedMetrics: TrainingMetrics[];
  singleMetrics: TrainingMetrics[];
  videoStatus: VideoStatus;
  activeTab: 'distributed' | 'single' | 'comparison' | 'video';
  setIsRunning: (running: boolean) => void;
  setDistributedMetrics: (metrics: TrainingMetrics[]) => void;
  setSingleMetrics: (metrics: TrainingMetrics[]) => void;
  setVideoStatus: (status: VideoStatus) => void;
  setActiveTab: (tab: 'distributed' | 'single' | 'comparison' | 'video') => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  isRunning: false,
  distributedMetrics: [],
  singleMetrics: [],
  videoStatus: { status: 'idle' },
  activeTab: 'distributed',
  setIsRunning: (running) => set({ isRunning: running }),
  setDistributedMetrics: (metrics) => set({ distributedMetrics: metrics }),
  setSingleMetrics: (metrics) => set({ singleMetrics: metrics }),
  setVideoStatus: (status) => set({ videoStatus: status }),
  setActiveTab: (tab) => set({ activeTab: tab }),
}));
