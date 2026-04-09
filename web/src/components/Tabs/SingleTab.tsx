'use client';

import { useTrainingStore } from '@/stores/trainingStore';
import { MetricsChart } from '@/components/Charts/MetricsChart';

export function SingleTab() {
  const { singleMetrics } = useTrainingStore();

  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(500px,1fr))] gap-6">
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
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
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
          <MetricsChart
            title="平均回报"
            data={singleMetrics}
            metricKey="avg_return"
            color="#f59e0b"
            label="Single Return"
          />
        </div>
      </div>
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
          <MetricsChart
            title="总损失"
            data={singleMetrics}
            metricKey="loss"
            color="#dc3545"
            label="Single Loss"
          />
        </div>
      </div>
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
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
  );
}
