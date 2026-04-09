'use client';

import { useTrainingStore } from '@/stores/trainingStore';
import { MetricsChart } from '@/components/Charts/MetricsChart';

export function DistributedTab() {
  const { distributedMetrics } = useTrainingStore();

  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(500px,1fr))] gap-6">
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
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
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
          <MetricsChart
            title="平均回报"
            data={distributedMetrics}
            metricKey="avg_return"
            color="#f59e0b"
            label="Distributed Return"
          />
        </div>
      </div>
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
          <MetricsChart
            title="总损失"
            data={distributedMetrics}
            metricKey="loss"
            color="#dc3545"
            label="Distributed Loss"
          />
        </div>
      </div>
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
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
  );
}
