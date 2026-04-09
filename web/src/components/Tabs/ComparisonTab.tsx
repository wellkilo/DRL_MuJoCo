'use client';

import { useTrainingStore } from '@/stores/trainingStore';
import { ComparisonChart } from '@/components/Charts/ComparisonChart';

export function ComparisonTab() {
  const { distributedMetrics, singleMetrics } = useTrainingStore();

  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(500px,1fr))] gap-6">
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
          <ComparisonChart
            title="训练速度对比 (SPS)"
            distData={distributedMetrics}
            singleData={singleMetrics}
            metricKey="sps"
            beginAtZero={true}
          />
        </div>
      </div>
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
          <ComparisonChart
            title="平均回报对比"
            distData={distributedMetrics}
            singleData={singleMetrics}
            metricKey="avg_return"
          />
        </div>
      </div>
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
          <ComparisonChart
            title="总损失对比"
            distData={distributedMetrics}
            singleData={singleMetrics}
            metricKey="loss"
          />
        </div>
      </div>
      <div className="rounded-xl p-6 shadow-sm border border-border bg-white">
        <div className="h-[300px]">
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
  );
}
