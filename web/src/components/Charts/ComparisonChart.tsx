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

interface ComparisonChartProps {
  title: string;
  distData: TrainingMetrics[];
  singleData: TrainingMetrics[];
  metricKey: MetricsKey;
  beginAtZero?: boolean;
  isDark?: boolean;
}

export function ComparisonChart({
  title,
  distData,
  singleData,
  metricKey,
  beginAtZero = false,
  isDark = true,
}: ComparisonChartProps) {
  const allSteps = Array.from(
    new Set([...distData.map((m) => m.step), ...singleData.map((m) => m.step)])
  ).sort((a, b) => a - b);

  const gridColor = isDark ? 'rgba(51, 65, 85, 0.3)' : 'rgba(203, 213, 225, 0.5)';
  const tickColor = isDark ? '#64748b' : '#94a3b8';
  const legendColor = isDark ? '#94a3b8' : '#475569';
  const tooltipBg = isDark ? '#1a2035' : '#ffffff';
  const tooltipTitle = isDark ? '#f1f5f9' : '#0f172a';
  const tooltipBody = isDark ? '#94a3b8' : '#475569';
  const tooltipBorder = isDark ? '#334155' : '#e2e8f0';
  const fillAlpha = isDark ? '08' : '18';

  const chartData = {
    labels: allSteps,
    datasets: [
      {
        label: 'Distributed',
        data: allSteps.map((step) => {
          const m = distData.find((x) => x.step === step);
          return m ? m[metricKey] : null;
        }),
        borderColor: '#6366f1',
        backgroundColor: `rgba(99, 102, 241, 0.${fillAlpha})`,
        fill: true,
        tension: 0.4,
        pointRadius: allSteps.length > 100 ? 0 : 2,
        pointHoverRadius: 4,
        pointBackgroundColor: '#6366f1',
        pointBorderColor: 'transparent',
        borderWidth: 2,
      },
      {
        label: 'Single',
        data: allSteps.map((step) => {
          const m = singleData.find((x) => x.step === step);
          return m ? m[metricKey] : null;
        }),
        borderColor: '#06b6d4',
        backgroundColor: `rgba(6, 182, 212, 0.${fillAlpha})`,
        fill: true,
        tension: 0.4,
        pointRadius: allSteps.length > 100 ? 0 : 2,
        pointHoverRadius: 4,
        pointBackgroundColor: '#06b6d4',
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

  return <Line data={chartData} options={options} />;
}
