'use client';

import { useEffect, useRef, useState } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import { generateVideos, getVideoStatus } from '@/services/api';

export function VideoSection() {
  const { videoStatus, setVideoStatus, isRunning } = useTrainingStore();
  const videoDistributedRef = useRef<HTMLVideoElement>(null);
  const videoSingleRef = useRef<HTMLVideoElement>(null);
  const [videoKey, setVideoKey] = useState(0);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;

    if (videoStatus.status === 'generating') {
      interval = setInterval(async () => {
        const status = await getVideoStatus();
        setVideoStatus(status);
        if (status.status === 'completed') {
          setVideoKey((k) => k + 1);
        }
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [videoStatus.status, setVideoStatus]);

  const handleGenerateVideos = async () => {
    await generateVideos();
    setVideoStatus({ status: 'generating' });
  };

  const playAll = () => {
    videoDistributedRef.current?.play();
    videoSingleRef.current?.play();
  };

  const pauseAll = () => {
    videoDistributedRef.current?.pause();
    videoSingleRef.current?.pause();
  };

  const resetAll = () => {
    if (videoDistributedRef.current) {
      videoDistributedRef.current.currentTime = 0;
      videoDistributedRef.current.pause();
    }
    if (videoSingleRef.current) {
      videoSingleRef.current.currentTime = 0;
      videoSingleRef.current.pause();
    }
  };

  const getVideoStatusText = () => {
    switch (videoStatus.status) {
      case 'generating':
        return '正在生成视频...';
      case 'completed':
        return '视频生成完成！点击"生成对比视频"可重新生成';
      case 'error':
        return '生成失败: ' + (videoStatus.error || 'Unknown error');
      default:
        return '点击下方按钮生成训练结果视频';
    }
  };

  const statusConfig: Record<string, { bg: string; text: string; border: string; icon: string }> = {
    idle: { bg: 'bg-dark-600/50', text: 'text-text-secondary', border: 'border-border-dark', icon: '🎬' },
    generating: { bg: 'bg-warning/10', text: 'text-warning-light', border: 'border-warning/20', icon: '⏳' },
    completed: { bg: 'bg-success/10', text: 'text-success-light', border: 'border-success/20', icon: '✅' },
    error: { bg: 'bg-danger/10', text: 'text-danger-light', border: 'border-danger/20', icon: '❌' },
  };

  const distSrc = `/api/videos/distributed?t=${videoKey}`;
  const singleSrc = `/api/videos/single?t=${videoKey}`;
  const currentStatus = statusConfig[videoStatus.status] || statusConfig.idle;

  return (
    <div className="space-y-5">
      {/* Status Banner */}
      <div className={`flex items-center justify-center gap-2 py-3 px-4 rounded-xl border ${currentStatus.bg} ${currentStatus.text} ${currentStatus.border} text-sm font-medium`}>
        <span>{currentStatus.icon}</span>
        {getVideoStatusText()}
        {videoStatus.status === 'generating' && (
          <span className="inline-flex">
            <span className="animate-pulse">.</span>
            <span className="animate-pulse [animation-delay:0.2s]">.</span>
            <span className="animate-pulse [animation-delay:0.4s]">.</span>
          </span>
        )}
      </div>

      {/* Video Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-card p-5">
          <h3 className="flex items-center gap-2 mb-4 text-sm font-semibold text-text-primary">
            <span className="inline-flex items-center justify-center w-6 h-6 rounded-md bg-primary/15 text-primary text-xs">⚡</span>
            分布式训练
          </h3>
          <video
            key={`dist-${videoKey}`}
            ref={videoDistributedRef}
            className="w-full rounded-lg bg-dark-900 aspect-video"
            controls
            src={distSrc}
          />
        </div>
        <div className="glass-card p-5">
          <h3 className="flex items-center gap-2 mb-4 text-sm font-semibold text-text-primary">
            <span className="inline-flex items-center justify-center w-6 h-6 rounded-md bg-accent/15 text-accent text-xs">🖥️</span>
            单机训练
          </h3>
          <video
            key={`single-${videoKey}`}
            ref={videoSingleRef}
            className="w-full rounded-lg bg-dark-900 aspect-video"
            controls
            src={singleSrc}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-3 justify-center flex-wrap">
        <button
          className="btn-primary"
          onClick={handleGenerateVideos}
          disabled={isRunning || videoStatus.status === 'generating'}
        >
          🎬 生成对比视频
        </button>
        <button
          className="btn-success"
          onClick={playAll}
        >
          ▶️ 全部播放
        </button>
        <button
          className="btn-ghost"
          onClick={pauseAll}
        >
          ⏸️ 全部暂停
        </button>
        <button
          className="btn-ghost"
          onClick={resetAll}
        >
          🔄 全部重置
        </button>
      </div>
    </div>
  );
}
