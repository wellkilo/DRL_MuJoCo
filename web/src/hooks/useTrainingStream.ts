'use client';

import { useEffect } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import { wsManager } from '@/services/websocket';

export function useTrainingStream() {
  const { activeEnv, setDistributedMetrics, setSingleMetrics } = useTrainingStore();

  useEffect(() => {
    const unsubscribe = wsManager.subscribe((distributed, single, env) => {
      // Only update metrics if the data is for the active environment
      if (!env || env === activeEnv) {
        setDistributedMetrics(distributed);
        setSingleMetrics(single);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [activeEnv, setDistributedMetrics, setSingleMetrics]);
}