import { cardData, precon, scripts } from "@fyendal/cards";
import {
  applyIntent,
  createGame,
  legalIntents,
  projectStateFor,
  projectStateForReplay,
} from "@fyendal/engine";
import type { Decklist, GameIntent, GameView } from "@fyendal/shared";
import { botDefinition, type BotDefinition } from "./registry.js";

/** Which observation(s) to capture per decision. "partial" is exactly what
 * the acting bot's policy saw; "full" is the full-information replay view;
 * "both" records both (larger rows, richer training signal). */
export type ObservationMode = "partial" | "full" | "both";

/** One recorded decision point: the legal intents offered to the acting bot
 * and the intent it chose. `chosenIndex` is always a valid index into
 * `legal`; `legal` is the advertised set from `legalIntents`, plus the
 * chosen intent appended when the engine accepts something outside the
 * advertised set (declarative multi-card defender staging). Intents use
 * per-game instance ids, so the (legal, chosenIndex) pair is the stable
 * offline action encoding within an episode. */
export interface RecordedDecision {
  episodeId: string;
  decisionIndex: number;
  turn: number;
  seat: 0 | 1;
  /** PendingDecision kind, or "priority" for open priority windows. */
  decisionKind: string;
  /** Partial view: exactly what the acting bot's policy observed. */
  observation: GameView | null;
  /** Full-information replay view (both players' hidden zones). */
  fullObservation: GameView | null;
  legal: GameIntent[];
  chosenIndex: number;
  chosen: GameIntent;
  decisionMs: number;
}

export interface RecordBotMatchOptions {
  left: string;
  right: string;
  seed: number;
  maxSteps?: number;
  /** Injectable monotonic clock keeps recordings deterministic. */
  now?: () => number;
  /** Defaults to `${left}-vs-${right}-seed-${seed}`. */
  episodeId?: string;
  /** Defaults to "partial". */
  observation?: ObservationMode;
  /** Optional streaming sink, called once per decision (in order). */
  onDecision?: (row: RecordedDecision) => void;
}

export interface BotMatchRecording {
  episodeId: string;
  bots: [string, string];
  seed: number;
  winner: 0 | 1 | null;
  turns: number;
  steps: number;
  decisions: [number, number];
  complete: boolean;
  rows: RecordedDecision[];
}

function registeredPoolDeck(definition: BotDefinition): Decklist {
  const registered = precon(definition.deckId);
  if (!registered) throw new Error(`missing bot deck ${definition.deckId}`);
  return {
    heroId: registered.pool.heroId,
    weaponIds: [...registered.pool.weaponIds],
    equipment: {},
    deck: [...registered.pool.deck, ...(registered.pool.sideboard ?? [])],
  };
}

function presentedDeck(definition: BotDefinition, opponent: Decklist): Decklist {
  const registered = precon(definition.deckId);
  if (!registered) throw new Error(`missing bot deck ${definition.deckId}`);
  return {
    heroId: registered.pool.heroId,
    ...definition.presentationFor(opponent, "first"),
  };
}

/** Bot-only information rules applied to the projected view before the policy
 * sees it (and before it is recorded). The shared engine projection is left
 * untouched so the human client keeps showing the real game:
 * - Own starting decklist: the bot knows what its deck started with
 *   (composition only — the list is sorted, so order is hidden).
 * - Opponent pitch: contents are public while in the pitch zone, but the
 *   pitch order is private, so the list is sorted (order hidden).
 * - Opponent deck contents stay hidden (projection already omits them);
 *   own pitch stays exact and ordered (projection already provides it). */
function applyBotInfoRules(
  view: GameView,
  actor: 0 | 1,
  startingDecks: [Decklist, Decklist],
): void {
  const own = view.players[actor];
  const opp = view.players[(1 - actor) as 0 | 1];
  if (!own || !opp) throw new Error("bot view missing player");
  // Own starting deck: composition known, order hidden (sorted cardIds).
  (own as { startingDeck?: string[] }).startingDeck = [...startingDecks[actor].deck].sort();
  // Opponent pitch: contents known, order hidden (sorted by cardId).
  opp.pitch.sort((a, b) => (a.cardId < b.cardId ? -1 : a.cardId > b.cardId ? 1 : 0));
}

/** Canonical intent key for comparing intents: recursively sorts object
 * keys before stringifying, so structurally identical intents compare
 * equal regardless of key insertion order. Array order stays significant. */
function intentKey(intent: GameIntent): string {
  return canonicalJson(intent);
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, entryValue]) => `${JSON.stringify(key)}:${canonicalJson(entryValue)}`);
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/** Headless bot-vs-bot match with per-decision recording for offline RL.
 * Runs the same deterministic loop as evaluateBotMatch, but captures the
 * full legal intent array offered at every decision point alongside the
 * intent the bot chose. Pure engine: no server, DB, or network.
 *
 * The bot sees only public player information: it receives the projected
 * seat view (never the full internal state), so no hidden zones leak into
 * decisions or recordings. */
export function recordBotMatch(options: RecordBotMatchOptions): BotMatchRecording {
  const left = botDefinition(options.left);
  const right = botDefinition(options.right);
  if (!left || !right) throw new Error("unknown bot definition");
  if (left.format !== right.format) throw new Error("bots must share a format");
  const leftPool = registeredPoolDeck(left);
  const rightPool = registeredPoolDeck(right);
  const episodeId = options.episodeId ?? `${left.id}-vs-${right.id}-seed-${options.seed}`;
  const observationMode: ObservationMode = options.observation ?? "partial";
  // Starting decklists (post-presentation): the bot knows what its own deck
  // started with. Captured here so applyBotInfoRules can attach them.
  const startingDecks: [Decklist, Decklist] = [
    presentedDeck(left, rightPool),
    presentedDeck(right, leftPool),
  ];
  let state = createGame({
    decklists: startingDecks,
    cards: cardData,
    scripts,
    seed: options.seed,
    startPlayer: 0,
  });
  const definitions = [left, right] as const;
  const decisions: [number, number] = [0, 0];
  const rows: RecordedDecision[] = [];
  const now = options.now ?? (() => performance.now());
  const maxSteps = options.maxSteps ?? 2_000;
  let steps = 0;
  for (; steps < maxSteps && state.winner === null; steps++) {
    const actor = (state.pendingDecision?.player ?? state.priorityPlayer) as 0 | 1;
    const legal = legalIntents(state, actor).filter((intent) => intent.kind !== "concede");
    if (legal.length === 0) throw new Error(`bot ${definitions[actor].id} has no legal intent`);
    const decisionKind = state.pendingDecision?.kind ?? "priority";
    const turn = state.turn;
    // The policy sees exactly what a player would: the projected seat view
    // (own hand + public zones; the opponent's hidden zones are concealed by
    // projectStateFor). The full internal `state` is deliberately withheld:
    // the turn planner's rollout adapter clones it, and that clone only
    // scrubs deck order -- the opponent's hand would remain visible to the
    // bot. Data collection must not leak hidden information.
    const view = projectStateFor(state, actor);
    // Bot-only info rules: own starting decklist (unordered), opponent
    // pitch unordered. Applied before the policy sees the view and before
    // it is recorded, so policy input and dataset agree.
    applyBotInfoRules(view, actor, startingDecks);
    const observation = observationMode === "full" ? null : view;
    const fullObservation =
      observationMode === "partial" ? null : projectStateForReplay(state);
    const startedAt = now();
    const intent = definitions[actor].chooseIntent({
      seat: actor,
      view,
      legal,
      cards: cardData,
      state: undefined,
    });
    const elapsed = Math.max(0, now() - startedAt);
    decisions[actor]++;
    const applied = applyIntent(state, actor, intent);
    if (!applied.ok) throw new Error(`bot ${definitions[actor].id} returned ${intent.kind}: ${applied.error}`);
    const chosenKey = intentKey(intent);
    let chosenIndex = legal.findIndex((candidate) => intentKey(candidate) === chosenKey);
    // Widen back to GameIntent[]: the filter above narrows the element type
    // via an inferred type predicate, but the appended chosen intent is
    // typed as the full GameIntent union.
    const legalForRow: GameIntent[] = [...legal];
    if (chosenIndex === -1) {
      // Declarative intents (e.g. multi-card defender staging) are
      // engine-legal but sit outside the advertised candidate set. Append
      // the chosen intent to the row's legal array so chosenIndex is always
      // a valid index into legal.
      legalForRow.push({ ...intent });
      chosenIndex = legalForRow.length - 1;
    }
    const row: RecordedDecision = {
      episodeId,
      decisionIndex: rows.length,
      turn,
      seat: actor,
      decisionKind,
      observation,
      fullObservation,
      legal: legalForRow,
      chosenIndex,
      chosen: intent,
      decisionMs: elapsed,
    };
    rows.push(row);
    options.onDecision?.(row);
    state = applied.state;
  }
  return {
    episodeId,
    bots: [left.id, right.id],
    seed: options.seed,
    winner: state.winner === 0 || state.winner === 1 ? state.winner : null,
    turns: state.turn,
    steps,
    decisions,
    complete: state.winner !== null,
    rows,
  };
}

/** Serializes recorded rows as JSONL (one decision per line). */
export function toJsonl(rows: readonly RecordedDecision[]): string {
  return rows.map((row) => JSON.stringify(row)).join("\n") + (rows.length > 0 ? "\n" : "");
}
