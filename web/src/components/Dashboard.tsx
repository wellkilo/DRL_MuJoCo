'use client';

import { useCallback } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import { useTrainingStream } from '@/hooks/useTrainingStream';
import { useMetricsHistory } from '@/hooks/useMetricsHistory';
import { wsManager } from '@/services/websocket';
import {
  startDistributedTraining,
  startSingleTraining,
  stopTraining,
} from '@/services/api';
import { DistributedTab } from '@/components/Tabs/DistributedTab';
import { SingleTab } from '@/components/Tabs/SingleTab';
import { ComparisonTab } from '@/components/Tabs/ComparisonTab';
import { VideoSection } from '@/components/Video/VideoSection';

export default function Dashboard() {
  const {
    isRunning,
    activeTab,
    setIsRunning,
    setActiveTab,
  } = useTrainingStore();

  useMetricsHistory();
  useTrainingStream();

  const handleStartDistributed = useCallback(async () => {
    const data = await startDistributedTraining();
    if (data.status === 'started' || data.status === 'already running') {
      setIsRunning(true);
      wsManager.connect();
    }
  }, [setIsRunning]);

  const handleStartSingle = useCallback(async () => {
    const data = await startSingleTraining();
    if (data.status === 'started' || data.status === 'already running') {
      setIsRunning(true);
      wsManager.connect();
    }
  }, [setIsRunning]);

  const handleStop = useCallback(async () => {
    await stopTraining();
    setIsRunning(false);
    wsManager.disconnect();
  }, [setIsRunning]);

  const tabs = [
    { key: 'distributed' as const, label: '分布式训练' },
    { key: 'single' as const, label: '单机训练' },
    { key: 'comparison' as const, label: '性能对比' },
    { key: 'video' as const, label: '视频演示' },
  ];

  return (
    <div className="mx-auto max-w-[1600px] rounded-2xl bg-white shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-[#667eea] to-[#764ba2] px-8 py-8 text-center text-white">
        <h1 className="text-3xl font-bold mb-1">DRL MuJoCo 分布式训练监控</h1>
        <p className="opacity-90">实时监控强化学习训练过程，对比单机与分布式性能</p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center py-5 px-6 bg-bg-light border-b border-border">
        <button
          className="px-8 py-3 rounded-lg text-sm font-semibold text-white bg-gradient-to-r from-[#667eea] to-[#764ba2] shadow-md transition-all hover:-translate-y-0.5 hover:shadow-[0_4px_12px_rgba(102,126,234,0.4)] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          onClick={handleStartDistributed}
          disabled={isRunning}
        >
          启动分布式训练
        </button>
        <button
          className="px-8 py-3 rounded-lg text-sm font-semibold text-white bg-gradient-to-r from-[#28a745] to-[#20c997] shadow-md transition-all hover:-translate-y-0.5 hover:shadow-[0_4px_12px_rgba(40,167,69,0.4)] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          onClick={handleStartSingle}
          disabled={isRunning}
        >
          启动单机训练
        </button>
        <button
          className="px-8 py-3 rounded-lg text-sm font-semibold text-white bg-danger shadow-md transition-all hover:-translate-y-0.5 hover:shadow-[0_4px_12px_rgba(220,53,69,0.4)] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          onClick={handleStop}
          disabled={!isRunning}
        >
          停止训练
        </button>
      </div>

      {/* Status */}
      <div
        className={`px-8 py-3 text-center font-semibold text-base border-b ${
          isRunning
            ? 'bg-[#d4edda] text-[#155724] border-[#28a745]'
            : 'bg-[#fff3cd] text-text-primary border-[#ffc107]'
        }`}
      >
        {isRunning ? '状态: 训练中...' : '状态: 已停止'}
      </div>

      {/* Tabs */}
      <div className="flex bg-bg-light border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            className={`px-8 py-4 cursor-pointer border-none bg-transparent text-sm font-semibold transition-all border-b-3 ${
              activeTab === tab.key
                ? 'text-primary border-b-3 border-primary bg-white'
                : 'text-text-secondary border-b-3 border-transparent hover:text-primary'
            }`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-8">
        {activeTab === 'distributed' && <DistributedTab />}
        {activeTab === 'single' && <SingleTab />}
        {activeTab === 'comparison' && <ComparisonTab />}
        {activeTab === 'video' && <VideoSection />}
      </div>
    </div>
  );
}