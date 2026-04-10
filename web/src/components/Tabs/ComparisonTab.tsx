'use client';

import { useTrainingStore } from '@/stores/trainingStore';
import { ComparisonChart } from '@/components/Charts/ComparisonChart';

const chartCards = [
  {
    title: '训练速度对比 (SPS)',
    metricKey: 'sps' as const,
    beginAtZero: true,
    icon: '⚡',
    accent: 'border-l-chart-violet',
  },
  {
    title: '平均回报对比',
    metricKey: 'avg_return' as const,
    beginAtZero: false,
    icon: '📊',
    accent: 'border-l-chart-amber',
  },
  {
    title: '总损失对比',
    metricKey: 'loss' as const,
    beginAtZero: false,
    icon: '📉',
    accent: 'border-l-chart-rose',
  },
  {
    title: 'Buffer 大小对比',
    metricKey: 'buffer_size' as const,
    beginAtZero: true,
    icon: '💾',
    accent: 'border-l-chart-emerald',
  },
];

interface ComparisonTabProps {
  isDark?: boolean;
}

export function ComparisonTab({ isDark = true }: ComparisonTabProps) {
  const { distributedMetrics, singleMetrics } = useTrainingStore();

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {chartCards.map((card) => (
        <div
          key={card.metricKey}
          className={`glass-card-hover border-l-4 ${card.accent} p-5`}
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="text-base">{card.icon}</span>
            <h3 className="text-sm font-semibold text-text-primary">
              {card.title}
            </h3>
          </div>
          <div className="h-[280px]">
            <ComparisonChart
              title={card.title}
              distData={distributedMetrics}
              singleData={singleMetrics}
              metricKey={card.metricKey}
              beginAtZero={card.beginAtZero}
              isDark={isDark}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
