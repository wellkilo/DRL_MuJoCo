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

interface ComparisonChartProps {
  title: string;
  distData: TrainingMetrics[];
  singleData: TrainingMetrics[];
  metricKey: MetricsKey;
  beginAtZero?: boolean;
}

export const ComparisonChart: React.FC<ComparisonChartProps> = ({
  title,
  distData,
  singleData,
  metricKey,
  beginAtZero = false,
}) => {
  const allSteps = Array.from(new Set([
    ...distData.map(m => m.step),
    ...singleData.map(m => m.step),
  ])).sort((a, b) => a - b);

  const chartData = {
    labels: allSteps,
    datasets: [
      {
        label: 'Distributed',
        data: allSteps.map(step => {
          const m = distData.find(x => x.step === step);
          return m ? m[metricKey] : null;
        }),
        borderColor: '#667eea',
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 2,
      },
      {
        label: 'Single',
        data: allSteps.map(step => {
          const m = singleData.find(x => x.step === step);
          return m ? m[metricKey] : null;
        }),
        borderColor: '#20c997',
        backgroundColor: 'rgba(32, 201, 151, 0.1)',
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
