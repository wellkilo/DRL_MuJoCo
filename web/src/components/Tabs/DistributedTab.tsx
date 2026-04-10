'use client';

import { useTrainingStore } from '@/stores/trainingStore';
import { MetricsChart } from '@/components/Charts/MetricsChart';

const chartCards = [
  {
    title: '训练速度 (SPS)',
    metricKey: 'sps' as const,
    color: '#6366f1',
    label: 'Distributed SPS',
    beginAtZero: true,
    icon: '🚀',
    accent: 'border-l-chart-blue',
  },
  {
    title: '平均回报',
    metricKey: 'avg_return' as const,
    color: '#f59e0b',
    label: 'Distributed Return',
    beginAtZero: false,
    icon: '📈',
    accent: 'border-l-chart-amber',
  },
  {
    title: '总损失',
    metricKey: 'loss' as const,
    color: '#f43f5e',
    label: 'Distributed Loss',
    beginAtZero: false,
    icon: '📉',
    accent: 'border-l-chart-rose',
  },
  {
    title: 'Buffer 大小',
    metricKey: 'buffer_size' as const,
    color: '#10b981',
    label: 'Distributed Buffer',
    beginAtZero: true,
    icon: '💾',
    accent: 'border-l-chart-emerald',
  },
];

interface DistributedTabProps {
  isDark?: boolean;
}

export function DistributedTab({ isDark = true }: DistributedTabProps) {
  const { distributedMetrics } = useTrainingStore();

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
            <MetricsChart
              title={card.title}
              data={distributedMetrics}
              metricKey={card.metricKey}
              color={card.color}
              label={card.label}
              beginAtZero={card.beginAtZero}
              isDark={isDark}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
