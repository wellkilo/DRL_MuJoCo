import React from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { MetricsChart } from '../Charts/MetricsChart';

export const SingleTab: React.FC = () => {
  const { singleMetrics } = useTrainingStore();

  return (
    <div className="tab-content active">
      <div className="charts-grid">
        <div className="chart-card">
          <div className="chart-wrapper">
            <MetricsChart
              title="训练速度 (SPS)"
              data={singleMetrics}
              metricKey="sps"
              color="#667eea"
              label="Single SPS"
              beginAtZero={true}
            />
          </div>
        </div>
        <div className="chart-card">
          <div className="chart-wrapper">
            <MetricsChart
              title="平均回报"
              data={singleMetrics}
              metricKey="avg_return"
              color="#f59e0b"
              label="Single Return"
            />
          </div>
        </div>
        <div className="chart-card">
          <div className="chart-wrapper">
            <MetricsChart
              title="总损失"
              data={singleMetrics}
              metricKey="loss"
              color="#dc3545"
              label="Single Loss"
            />
          </div>
        </div>
        <div className="chart-card">
          <div className="chart-wrapper">
            <MetricsChart
              title="Buffer 大小"
              data={singleMetrics}
              metricKey="buffer_size"
              color="#28a745"
              label="Single Buffer"
              beginAtZero={true}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
