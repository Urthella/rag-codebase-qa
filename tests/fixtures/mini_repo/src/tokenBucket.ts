// Minimal token bucket used as an indexing fixture.

export interface Decision {
  allowed: boolean;
  remaining: number;
}

/**
 * Token bucket: refills continuously at refillPerSecond up to capacity.
 */
export class TokenBucket {
  private tokens: number;
  private lastMs: number;

  constructor(private readonly capacity: number, private readonly refillPerSecond: number) {
    this.tokens = capacity;
    this.lastMs = Date.now();
  }

  /** Refill the bucket based on elapsed time, then try to consume one token. */
  check(nowMs: number): Decision {
    const elapsedMs = Math.max(0, nowMs - this.lastMs);
    this.tokens = Math.min(this.capacity, this.tokens + (elapsedMs * this.refillPerSecond) / 1000);
    this.lastMs = nowMs;
    let allowed = false;
    if (this.tokens >= 1) {
      this.tokens -= 1;
      allowed = true;
    }
    return { allowed, remaining: Math.floor(this.tokens) };
  }
}
