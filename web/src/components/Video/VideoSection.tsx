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
        // When generation completes, increment key to force video reload
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
        return '视频状态: 空闲';
    }
  };

  const statusColorMap: Record<string, string> = {
    idle: 'bg-bg-light text-text-primary',
    generating: 'bg-[#fff3cd] text-[#856404]',
    completed: 'bg-[#d4edda] text-[#155724]',
    error: 'bg-[#f8d7da] text-[#721c24]',
  };

  // Add cache-busting query parameter to force browser to reload videos
  const distSrc = `/api/videos/distributed?t=${videoKey}`;
  const singleSrc = `/api/videos/single?t=${videoKey}`;

  return (
    <div className="bg-white border-t border-border p-8 -m-8 mt-0">
      <h2 className="text-center text-text-primary text-xl font-bold mb-6">
        视频演示
      </h2>
      <div
        className={`text-center py-3 px-4 mb-6 font-semibold rounded-lg ${statusColorMap[videoStatus.status] || 'bg-bg-light text-text-primary'}`}
      >
        {getVideoStatusText()}
      </div>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(400px,1fr))] gap-8 mb-6">
        <div className="bg-bg-light rounded-xl p-6 shadow-sm border border-border">
          <h3 className="text-center mb-4 text-text-primary text-lg font-semibold">
            分布式训练
          </h3>
          <video
            key={`dist-${videoKey}`}
            ref={videoDistributedRef}
            className="w-full rounded-lg bg-black"
            controls
            src={distSrc}
          />
        </div>
        <div className="bg-bg-light rounded-xl p-6 shadow-sm border border-border">
          <h3 className="text-center mb-4 text-text-primary text-lg font-semibold">
            单机训练
          </h3>
          <video
            key={`single-${videoKey}`}
            ref={videoSingleRef}
            className="w-full rounded-lg bg-black"
            controls
            src={singleSrc}
          />
        </div>
      </div>
      <div className="flex gap-4 justify-center flex-wrap">
        <button
          className="px-8 py-3 rounded-lg text-sm font-semibold text-white bg-gradient-to-r from-[#667eea] to-[#764ba2] shadow-md transition-all hover:-translate-y-0.5 hover:shadow-[0_4px_12px_rgba(102,126,234,0.4)] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          onClick={handleGenerateVideos}
          disabled={isRunning || videoStatus.status === 'generating'}
        >
          生成对比视频
        </button>
        <button
          className="px-8 py-3 rounded-lg text-sm font-semibold text-white bg-gradient-to-r from-[#28a745] to-[#20c997] shadow-md transition-all hover:-translate-y-0.5 hover:shadow-[0_4px_12px_rgba(40,167,69,0.4)]"
          onClick={playAll}
        >
          全部播放
        </button>
        <button
          className="px-8 py-3 rounded-lg text-sm font-semibold text-white bg-gradient-to-r from-[#667eea] to-[#764ba2] shadow-md transition-all hover:-translate-y-0.5 hover:shadow-[0_4px_12px_rgba(102,126,234,0.4)]"
          onClick={pauseAll}
        >
          全部暂停
        </button>
        <button
          className="px-8 py-3 rounded-lg text-sm font-semibold text-white bg-gradient-to-r from-[#667eea] to-[#764ba2] shadow-md transition-all hover:-translate-y-0.5 hover:shadow-[0_4px_12px_rgba(102,126,234,0.4)]"
          onClick={resetAll}
        >
          全部重置
        </button>
      </div>
    </div>
  );
}
