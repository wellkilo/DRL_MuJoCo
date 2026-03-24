import React from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { MetricsChart } from '../Charts/MetricsChart';

export const DistributedTab: React.FC = () => {
  const { distributedMetrics } = useTrainingStore();

  return (
    <div className="tab-content active">
      <div className="charts-grid">
        <div className="chart-card">
          <div className="chart-wrapper">
            <MetricsChart
              title="训练速度 (SPS)"
              data={distributedMetrics}
              metricKey="sps"
              color="#667eea"
              label="Distributed SPS"
              beginAtZero={true}
            />
          </div>
        </div>
        <div className="chart-card">
          <div className="chart-wrapper">
            <MetricsChart
              title="平均回报"
              data={distributedMetrics}
              metricKey="avg_return"
              color="#f59e0b"
              label="Distributed Return"
            />
          </div>
        </div>
        <div className="chart-card">
          <div className="chart-wrapper">
            <MetricsChart
              title="总损失"
              data={distributedMetrics}
              metricKey="loss"
              color="#dc3545"
              label="Distributed Loss"
            />
          </div>
        </div>
        <div className="chart-card">
          <div className="chart-wrapper">
            <MetricsChart
              title="Buffer 大小"
              data={distributedMetrics}
              metricKey="buffer_size"
              color="#28a745"
              label="Distributed Buffer"
              beginAtZero={true}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
