import { create } from 'zustand';
import { TrainingMetrics, VideoStatus, EnvKey, EnvironmentInfo } from '@/types/metrics';

interface TrainingState {
  runningEnvs: Record<EnvKey, boolean>;
  isLoading: boolean;
  error: string | null;
  activeEnv: EnvKey;
  activeTab: 'distributed' | 'single' | 'comparison' | 'video';
  environments: Record<EnvKey, EnvironmentInfo>;
  // Per-environment metrics
  distributedMetrics: TrainingMetrics[];
  singleMetrics: TrainingMetrics[];
  videoStatus: VideoStatus;
  setEnvRunning: (env: EnvKey, running: boolean) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setActiveEnv: (env: EnvKey) => void;
  setDistributedMetrics: (metrics: TrainingMetrics[]) => void;
  setSingleMetrics: (metrics: TrainingMetrics[]) => void;
  setVideoStatus: (status: VideoStatus) => void;
  setActiveTab: (tab: 'distributed' | 'single' | 'comparison' | 'video') => void;
  setEnvironments: (envs: Record<EnvKey, EnvironmentInfo>) => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  runningEnvs: { hopper: false, walker2d: false, halfcheetah: false },
  isLoading: false,
  error: null,
  activeEnv: 'hopper',
  activeTab: 'distributed',
  distributedMetrics: [],
  singleMetrics: [],
  videoStatus: { status: 'idle' },
  environments: {
    hopper: { name: 'Hopper-v5', description: '单腿跳跃机器人', difficulty: 1 },
    walker2d: { name: 'Walker2d-v5', description: '双腿行走机器人', difficulty: 2 },
    halfcheetah: { name: 'HalfCheetah-v5', description: '半猎豹奔跑机器人', difficulty: 3 },
  },
  setEnvRunning: (env, running) =>
    set((state) => ({
      runningEnvs: { ...state.runningEnvs, [env]: running },
    })),
  setIsLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setActiveEnv: (env) => set({ activeEnv: env, distributedMetrics: [], singleMetrics: [] }),
  setDistributedMetrics: (metrics) => set({ distributedMetrics: metrics }),
  setSingleMetrics: (metrics) => set({ singleMetrics: metrics }),
  setVideoStatus: (status) => set({ videoStatus: status }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setEnvironments: (envs) => set({ environments: envs }),
}));
