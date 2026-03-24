import React, { useEffect, useRef } from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { generateVideos, getVideoStatus } from '../../services/api';

export const VideoSection: React.FC = () => {
  const { videoStatus, setVideoStatus, isRunning } = useTrainingStore();
  const videoDistributedRef = useRef<HTMLVideoElement>(null);
  const videoSingleRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;

    if (videoStatus.status === 'generating') {
      interval = setInterval(async () => {
        const status = await getVideoStatus();
        setVideoStatus(status);
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
        return '视频生成完成！';
      case 'error':
        return '生成失败: ' + (videoStatus.error || 'Unknown error');
      default:
        return '视频状态: 空闲';
    }
  };

  return (
    <div className="video-section">
      <h2>视频演示</h2>
      <div className={`video-status ${videoStatus.status}`}>
        {getVideoStatusText()}
      </div>
      <div className="video-container">
        <div className="video-card">
          <h3>分布式训练</h3>
          <video
            ref={videoDistributedRef}
            id="videoDistributed"
            className="video-player"
            controls
            src="/api/videos/distributed"
          />
        </div>
        <div className="video-card">
          <h3>单机训练</h3>
          <video
            ref={videoSingleRef}
            id="videoSingle"
            className="video-player"
            controls
            src="/api/videos/single"
          />
        </div>
      </div>
      <div className="video-controls">
        <button
          id="genVideoBtn"
          className="btn btn-primary"
          onClick={handleGenerateVideos}
          disabled={isRunning || videoStatus.status === 'generating'}
        >
          生成对比视频
        </button>
        <button className="btn btn-success" onClick={playAll}>
          全部播放
        </button>
        <button className="btn btn-primary" onClick={pauseAll}>
          全部暂停
        </button>
        <button className="btn btn-primary" onClick={resetAll}>
          全部重置
        </button>
      </div>
    </div>
  );
};
