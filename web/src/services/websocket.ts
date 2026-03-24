import { TrainingMetrics, WebSocketMessage } from '../types/metrics';

type MetricsCallback = (distributed: TrainingMetrics[], single: TrainingMetrics[]) => void;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private subscribers: Set<MetricsCallback> = new Set();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private isManualClose: boolean = false;

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.isManualClose = false;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/training`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('[WebSocket] Connected');
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const msg: WebSocketMessage = JSON.parse(event.data);
        if (msg.type === 'metrics') {
          this.notifySubscribers(msg.distributed || [], msg.single || []);
        }
      } catch (e) {
        console.error('[WebSocket] Parse error:', e);
      }
    };

    this.ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
    };

    this.ws.onclose = () => {
      console.log('[WebSocket] Closed');
      if (!this.isManualClose) {
        this.scheduleReconnect();
      }
    };
  }

  disconnect() {
    this.isManualClose = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  subscribe(callback: MetricsCallback) {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  private notifySubscribers(distributed: TrainingMetrics[], single: TrainingMetrics[]) {
    this.subscribers.forEach(cb => cb(distributed, single));
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.isManualClose) {
        this.connect();
      }
    }, 1000);
  }
}

export const wsManager = new WebSocketManager();
