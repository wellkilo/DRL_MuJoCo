'use client';

import { useCallback } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import { useTrainingStream } from '@/hooks/useTrainingStream';
import { useMetricsHistory } from '@/hooks/useMetricsHistory';
import { useTheme } from '@/hooks/useTheme';
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
    isLoading,
    error,
    activeTab,
    setIsRunning,
    setIsLoading,
    setError,
    setActiveTab,
    distributedMetrics,
    singleMetrics,
  } = useTrainingStore();

  const { isDark, toggleTheme } = useTheme();

  useMetricsHistory();
  useTrainingStream();

  const handleStartDistributed = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await startDistributedTraining();
      if (data.status === 'started' || data.status === 'already running') {
        setIsRunning(true);
        wsManager.connect();
      } else {
        setError(`启动分布式训练失败: ${data.status}`);
      }
    } catch (e) {
      setError(`请求失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setIsLoading(false);
    }
  }, [setIsRunning, setIsLoading, setError]);

  const handleStartSingle = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await startSingleTraining();
      if (data.status === 'started' || data.status === 'already running') {
        setIsRunning(true);
        wsManager.connect();
      } else {
        setError(`启动单机训练失败: ${data.status}`);
      }
    } catch (e) {
      setError(`请求失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setIsLoading(false);
    }
  }, [setIsRunning, setIsLoading, setError]);

  const handleStop = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      await stopTraining();
      setIsRunning(false);
      wsManager.disconnect();
    } catch (e) {
      setError(`停止训练失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setIsLoading(false);
    }
  }, [setIsRunning, setIsLoading, setError]);

  // Get latest metric values for KPI cards
  const activeMetrics = activeTab === 'single' ? singleMetrics : distributedMetrics;
  const latest = activeMetrics[activeMetrics.length - 1];

  const tabs = [
    { key: 'distributed' as const, label: '分布式训练', icon: '⚡' },
    { key: 'single' as const, label: '单机训练', icon: '🖥️' },
    { key: 'comparison' as const, label: '性能对比', icon: '📊' },
    { key: 'video' as const, label: '视频演示', icon: '🎬' },
  ];

  const kpiCards = [
    {
      label: '训练速度',
      value: latest?.sps != null ? latest.sps.toLocaleString() : '—',
      unit: 'SPS',
      color: 'text-chart-blue',
      bgColor: 'from-chart-blue/10 to-chart-blue/5',
      borderColor: 'border-chart-blue/20',
      icon: '🚀',
    },
    {
      label: '平均回报',
      value: latest?.avg_return != null ? latest.avg_return.toFixed(1) : '—',
      unit: '',
      color: 'text-chart-amber',
      bgColor: 'from-chart-amber/10 to-chart-amber/5',
      borderColor: 'border-chart-amber/20',
      icon: '📈',
    },
    {
      label: '总损失',
      value: latest?.loss != null ? latest.loss.toFixed(4) : '—',
      unit: '',
      color: 'text-chart-rose',
      bgColor: 'from-chart-rose/10 to-chart-rose/5',
      borderColor: 'border-chart-rose/20',
      icon: '📉',
    },
    {
      label: 'Buffer 大小',
      value: latest?.buffer_size != null ? latest.buffer_size.toLocaleString() : '—',
      unit: '',
      color: 'text-chart-emerald',
      bgColor: 'from-chart-emerald/10 to-chart-emerald/5',
      borderColor: 'border-chart-emerald/20',
      icon: '💾',
    },
  ];

  return (
    <div className="mx-auto max-w-[1680px] animate-fade-in">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-text-primary tracking-tight">
              DRL MuJoCo
              <span className="text-primary ml-2 font-medium text-lg md:text-xl">训练监控</span>
            </h1>
            <p className="text-text-secondary text-sm mt-1">
              实时监控强化学习训练过程 · 对比单机与分布式性能
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-surface-600 border border-border-dark hover:bg-surface-500 hover:border-border-light transition-all duration-200"
              title={isDark ? '切换到浅色主题' : '切换到深色主题'}
            >
              {isDark ? '☀️' : '🌙'}
            </button>

            {/* Status indicator */}
            <div className={`flex items-center gap-2.5 px-4 py-2 rounded-full text-sm font-medium ${
              isRunning
                ? 'bg-success/10 text-success-light border border-success/20'
                : 'bg-surface-600 text-text-secondary border border-border-dark'
            }`}>
              <span className="relative flex h-2.5 w-2.5">
                {isRunning && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75" />
                )}
                <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
                  isRunning ? 'bg-success' : 'bg-text-muted'
                }`} />
              </span>
              {isRunning ? '训练中' : '已停止'}
            </div>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mb-4 flex items-center gap-3 px-4 py-3 rounded-xl bg-danger/10 border border-danger/20 text-danger-light text-sm animate-slide-up">
          <span>❌</span>
          <span className="flex-1">{error}</span>
          <button
            onClick={() => setError(null)}
            className="text-danger-light/60 hover:text-danger-light transition-colors text-lg leading-none"
          >
            ×
          </button>
        </div>
      )}

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4 mb-6">
        {kpiCards.map((card) => (
          <div
            key={card.label}
            className={`relative overflow-hidden rounded-xl border ${card.borderColor} bg-gradient-to-br ${card.bgColor} backdrop-blur-sm p-4 md:p-5 transition-all duration-300 hover:scale-[1.02]`}
          >
            <div className="flex items-start justify-between mb-2">
              <span className="text-text-secondary text-xs font-medium uppercase tracking-wider">
                {card.label}
              </span>
              <span className="text-lg">{card.icon}</span>
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className={`text-xl md:text-2xl font-bold font-mono ${card.color}`}>
                {card.value}
              </span>
              {card.unit && (
                <span className="text-text-muted text-xs font-medium">{card.unit}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="glass-card p-4 mb-6">
        <div className="flex flex-wrap gap-3 items-center justify-center">
          <button
            className="btn-primary"
            onClick={handleStartDistributed}
            disabled={isRunning || isLoading}
          >
            {isLoading ? '⏳ 处理中...' : '⚡ 启动分布式训练'}
          </button>
          <button
            className="btn-success"
            onClick={handleStartSingle}
            disabled={isRunning || isLoading}
          >
            {isLoading ? '⏳ 处理中...' : '🖥️ 启动单机训练'}
          </button>
          <button
            className="btn-danger"
            onClick={handleStop}
            disabled={!isRunning || isLoading}
          >
            {isLoading ? '⏳ 处理中...' : '⏹ 停止训练'}
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-1 mb-4 bg-surface-700 rounded-xl p-1.5 border border-border-dark">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
              activeTab === tab.key
                ? 'bg-primary/15 text-primary-light shadow-sm border border-primary/20'
                : 'text-text-secondary hover:text-text-primary hover:bg-surface-600 border border-transparent'
            }`}
            onClick={() => setActiveTab(tab.key)}
          >
            <span className="text-base">{tab.icon}</span>
            <span className="hidden sm:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="animate-slide-up">
        {activeTab === 'distributed' && <DistributedTab isDark={isDark} />}
        {activeTab === 'single' && <SingleTab isDark={isDark} />}
        {activeTab === 'comparison' && <ComparisonTab isDark={isDark} />}
        {activeTab === 'video' && <VideoSection />}
      </div>
    </div>
  );
}