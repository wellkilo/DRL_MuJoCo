'use client';

import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { TrainingMetrics, MetricsKey } from '@/types/metrics';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface MetricsChartProps {
  title: string;
  data: TrainingMetrics[];
  metricKey: MetricsKey;
  color: string;
  label?: string;
  beginAtZero?: boolean;
  isDark?: boolean;
}

export function MetricsChart({
  title,
  data,
  metricKey,
  color,
  label,
  beginAtZero = false,
  isDark = true,
}: MetricsChartProps) {
  const safeData = data || [];
  const gridColor = isDark ? 'rgba(51, 65, 85, 0.3)' : 'rgba(203, 213, 225, 0.5)';
  const tickColor = isDark ? '#64748b' : '#94a3b8';
  const legendColor = isDark ? '#94a3b8' : '#475569';
  const tooltipBg = isDark ? '#1a2035' : '#ffffff';
  const tooltipTitle = isDark ? '#f1f5f9' : '#0f172a';
  const tooltipBody = isDark ? '#94a3b8' : '#475569';
  const tooltipBorder = isDark ? '#334155' : '#e2e8f0';

  const chartData = {
    labels: safeData.map((m) => m.step),
    datasets: [
      {
        label: label || title,
        data: safeData.map((m) => m[metricKey]),
        borderColor: color,
        backgroundColor: color + (isDark ? '15' : '25'),
        fill: true,
        tension: 0.4,
        pointRadius: safeData.length > 100 ? 0 : 2,
        pointHoverRadius: 4,
        pointBackgroundColor: color,
        pointBorderColor: 'transparent',
        borderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    interaction: {
      intersect: false,
      mode: 'index' as const,
    },
    scales: {
      x: {
        title: { display: true, text: 'Step', color: tickColor, font: { size: 11 } },
        ticks: { color: tickColor, font: { size: 10 }, maxTicksLimit: 8 },
        grid: { color: gridColor, drawBorder: false },
        border: { display: false },
      },
      y: {
        beginAtZero,
        ticks: { color: tickColor, font: { size: 10 }, maxTicksLimit: 6 },
        grid: { color: gridColor, drawBorder: false },
        border: { display: false },
      },
    },
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        align: 'end' as const,
        labels: {
          color: legendColor,
          font: { size: 11 },
          usePointStyle: true,
          pointStyle: 'circle',
          padding: 16,
          boxWidth: 6,
        },
      },
      title: { display: false },
      tooltip: {
        backgroundColor: tooltipBg,
        titleColor: tooltipTitle,
        bodyColor: tooltipBody,
        borderColor: tooltipBorder,
        borderWidth: 1,
        padding: 12,
        cornerRadius: 8,
        displayColors: true,
        boxPadding: 4,
      },
    },
  };

  if (safeData.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-text-muted">
        <span className="text-3xl mb-2">📊</span>
        <span className="text-sm">暂无数据</span>
        <span className="text-xs mt-1 text-text-muted/60">启动训练后将自动展示</span>
      </div>
    );
  }

  return <Line data={chartData} options={options} />;
}
