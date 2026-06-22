import type { AgentEventMap, AgentEventName } from "./types.js";

type Handler<K extends AgentEventName> = (payload: AgentEventMap[K]) => void;

export class AgentEventBus {
  private handlers = new Map<AgentEventName, Set<(payload: unknown) => void>>();

  on<K extends AgentEventName>(name: K, handler: Handler<K>): () => void {
    const set = this.handlers.get(name) ?? new Set<(payload: unknown) => void>();
    const wrapped = handler as (payload: unknown) => void;
    set.add(wrapped);
    this.handlers.set(name, set);
    return () => set.delete(wrapped);
  }

  emit<K extends AgentEventName>(name: K, payload: AgentEventMap[K]): void {
    const set = this.handlers.get(name);
    if (!set) return;
    for (const handler of set) handler(payload);
  }
}
