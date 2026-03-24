import { useCallback, useEffect } from 'react';
import { useTrainingStore } from './stores/trainingStore';
import { useTrainingStream } from './hooks/useTrainingStream';
import { useMetricsHistory } from './hooks/useMetricsHistory';
import { wsManager } from './services/websocket';
import {
  startDistributedTraining,
  startSingleTraining,
  stopTraining,
} from './services/api';
import { DistributedTab } from './components/Tabs/DistributedTab';
import { SingleTab } from './components/Tabs/SingleTab';
import { ComparisonTab } from './components/Tabs/ComparisonTab';
import { VideoSection } from './components/Video/VideoSection';
import './styles/global.scss';
import favIcon from './assets/favicon.svg';

function App() {
  useEffect(() => {
    const faviconEl = document.querySelector("link[rel*='icon']") as HTMLLinkElement;
    
    if (faviconEl) {
      faviconEl.href = favIcon;
    }
  }, []);
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

  const getStatusText = () => {
    return isRunning ? '状态: 训练中...' : '状态: 已停止';
  };

  return (
    <div className="container">
      <div className="header">
        <h1>DRL MuJoCo 分布式训练监控</h1>
        <p>实时监控强化学习训练过程，对比单机与分布式性能</p>
      </div>

      <div className="controls">
        <button
          id="startDistBtn"
          className="btn btn-primary"
          onClick={handleStartDistributed}
          disabled={isRunning}
        >
          启动分布式训练
        </button>
        <button
          id="startSingleBtn"
          className="btn btn-success"
          onClick={handleStartSingle}
          disabled={isRunning}
        >
          启动单机训练
        </button>
        <button
          id="stopBtn"
          className="btn btn-danger"
          onClick={handleStop}
          disabled={!isRunning}
        >
          停止训练
        </button>
      </div>

      <div className={`status ${isRunning ? 'running' : ''}`}>
        <span id="status">{getStatusText()}</span>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'distributed' ? 'active' : ''}`}
          onClick={() => setActiveTab('distributed')}
        >
          分布式训练
        </button>
        <button
          className={`tab ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          单机训练
        </button>
        <button
          className={`tab ${activeTab === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveTab('comparison')}
        >
          性能对比
        </button>
        <button
          className={`tab ${activeTab === 'video' ? 'active' : ''}`}
          onClick={() => setActiveTab('video')}
        >
          视频演示
        </button>
      </div>

      {activeTab === 'distributed' && <DistributedTab />}
      {activeTab === 'single' && <SingleTab />}
      {activeTab === 'comparison' && <ComparisonTab />}
      {activeTab === 'video' && <VideoSection />}
    </div>
  );
}

export default App;
