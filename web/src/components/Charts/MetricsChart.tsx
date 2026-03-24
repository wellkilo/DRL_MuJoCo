import React from 'react';
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
import { TrainingMetrics, MetricsKey } from '../../types/metrics';

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
}

export const MetricsChart: React.FC<MetricsChartProps> = ({
  title,
  data,
  metricKey,
  color,
  label,
  beginAtZero = false,
}) => {
  const safeData = data || [];
  const chartData = {
    labels: safeData.map(m => m.step),
    datasets: [
      {
        label: label || title,
        data: safeData.map(m => m[metricKey]),
        borderColor: color,
        backgroundColor: color + '20',
        fill: true,
        tension: 0.4,
        pointRadius: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    scales: {
      x: { title: { display: true, text: 'Step' } },
      y: { beginAtZero },
    },
    plugins: {
      legend: { display: true, position: 'top' as const },
      title: { display: true, text: title },
    },
  };

  return <Line data={chartData} options={options} />;
};
