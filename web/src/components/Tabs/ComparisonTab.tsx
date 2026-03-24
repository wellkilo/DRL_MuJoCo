import React from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { ComparisonChart } from '../Charts/ComparisonChart';

export const ComparisonTab: React.FC = () => {
  const { distributedMetrics, singleMetrics } = useTrainingStore();

  return (
    <div className="tab-content active">
      <div className="charts-grid">
        <div className="chart-card">
          <div className="chart-wrapper">
            <ComparisonChart
              title="训练速度对比 (SPS)"
              distData={distributedMetrics}
              singleData={singleMetrics}
              metricKey="sps"
              beginAtZero={true}
            />
          </div>
        </div>
        <div className="chart-card">
          <div className="chart-wrapper">
            <ComparisonChart
              title="平均回报对比"
              distData={distributedMetrics}
              singleData={singleMetrics}
              metricKey="avg_return"
            />
          </div>
        </div>
        <div className="chart-card">
          <div className="chart-wrapper">
            <ComparisonChart
              title="总损失对比"
              distData={distributedMetrics}
              singleData={singleMetrics}
              metricKey="loss"
            />
          </div>
        </div>
        <div className="chart-card">
          <div className="chart-wrapper">
            <ComparisonChart
              title="Buffer 大小对比"
              distData={distributedMetrics}
              singleData={singleMetrics}
              metricKey="buffer_size"
              beginAtZero={true}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
