// Fixture TypeScript file for chunker tests. Line numbers matter.
import { thing } from "./thing";

export interface RateLimitDecision {
  allowed: boolean;
  remaining: number;
}

/**
 * Checks whether a request is allowed.
 */
export function checkLimit(count: number, limit: number): RateLimitDecision {
  const allowed = count <= limit;
  return { allowed, remaining: Math.max(0, limit - count) };
}

export const refillTokens = (tokens: number, elapsedMs: number, rate: number): number => {
  return tokens + (elapsedMs * rate) / 1000;
};

// A tiny in-memory bucket.
export class Bucket {
  private tokens: number;

  constructor(private readonly capacity: number) {
    this.tokens = capacity;
  }

  /** Consume one token if available. */
  consume(): boolean {
    if (this.tokens >= 1) {
      this.tokens -= 1;
      return true;
    }
    return false;
  }
}

export type Algorithm = "fixed-window" | "token-bucket";
