'use client';

import { useEffect } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import { getDistributedMetrics, getSingleMetrics } from '@/services/api';

export function useMetricsHistory() {
  const { activeEnv, setDistributedMetrics, setSingleMetrics } = useTrainingStore();

  useEffect(() => {
    async function loadHistory() {
      const [dist, single] = await Promise.all([
        getDistributedMetrics(activeEnv),
        getSingleMetrics(activeEnv),
      ]);
      setDistributedMetrics(dist);
      setSingleMetrics(single);
    }
    loadHistory();
  }, [activeEnv, setDistributedMetrics, setSingleMetrics]);
}