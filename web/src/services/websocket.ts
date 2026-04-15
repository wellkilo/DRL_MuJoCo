import { TrainingMetrics, WebSocketMessage } from '@/types/metrics';

type MetricsCallback = (distributed: TrainingMetrics[], single: TrainingMetrics[], env?: string) => void;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private subscribers: Set<MetricsCallback> = new Set();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private isManualClose: boolean = false;
  private reconnectAttempts: number = 0;
  private static readonly MAX_RECONNECT_ATTEMPTS = 10;
  private static readonly BASE_DELAY = 1000;

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.isManualClose = false;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/training`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('[WebSocket] Connected');
      this.reconnectAttempts = 0;
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'metrics') {
          this.notifySubscribers(msg.distributed || [], msg.single || [], msg.env);
        } else if (msg.type === 'training_stopped') {
          // Training process crashed or ended — dispatch event so UI updates
          window.dispatchEvent(new CustomEvent('training-stopped', { detail: { env: msg.env, returncode: msg.returncode, error_detail: msg.error_detail || '' } }));
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
    this.reconnectAttempts = 0;
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

  private notifySubscribers(distributed: TrainingMetrics[], single: TrainingMetrics[], env?: string) {
    this.subscribers.forEach(cb => cb(distributed, single, env));
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= WebSocketManager.MAX_RECONNECT_ATTEMPTS) {
      console.warn('[WebSocket] Max reconnect attempts reached, giving up');
      return;
    }

    if (this.reconnectTimer) return;

    // Exponential backoff: 1s, 2s, 4s, 8s, 16s... capped at 30s
    const delay = Math.min(
      WebSocketManager.BASE_DELAY * Math.pow(2, this.reconnectAttempts),
      30000
    );

    this.reconnectAttempts++;
    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${WebSocketManager.MAX_RECONNECT_ATTEMPTS})`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.isManualClose) {
        this.connect();
      }
    }, delay);
  }
}

export const wsManager = new WebSocketManager();
