'use client';

import { useEffect } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import { wsManager } from '@/services/websocket';

export function useTrainingStream() {
  const { setDistributedMetrics, setSingleMetrics } = useTrainingStore();

  useEffect(() => {
    const unsubscribe = wsManager.subscribe((distributed, single) => {
      setDistributedMetrics(distributed);
      setSingleMetrics(single);
    });

    return () => {
      unsubscribe();
    };
  }, [setDistributedMetrics, setSingleMetrics]);
}