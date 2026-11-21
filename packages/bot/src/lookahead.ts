/**
 * Offensive lookahead (the project owner's design, 2026-10-06): copy the game
 * WITHOUT the opponent, search the acting player's own offensive lines, and
 * report which of their hand cards the best line uses -- the rest are free to
 * block -- and that line's value.
 *
 * Information hygiene. The copy is built from the authoritative state but
 * learns nothing hidden:
 * - the player's own deck is shuffled, except the bottom run of cards they
 *   pitched earlier this game, whose order they legitimately know;
 * - the opponent's hand, deck and face-down arsenal are emptied, and the
 *   opponent plays passively (never blocks, reacts or plays).
 *
 * On-hit effects. With a passive opponent every attack would hit, so each
 * attack the player makes is a chance node: a HIT branch (damage lands,
 * on-hit effects fire) and a NO-HIT branch (the opponent gets a shield for
 * the next damage), weighted by the hit rate measured from real games for
 * that attack's power: value = p * V(hit) + (1 - p) * V(no hit).
 *
 * With `blockers` (owner, 2026-10-15) the opponent instead blocks each
 * attack by a simple rule (see LookaheadOptions.blockers), so what is played
 * after the block decides how much lands; with `creditDamage` the damage
 * that lands is credited, as in the trainer's ledger.
 *
 * Value is fab-value.ts's per-card ledger, own side only, until the player's
 * turn ends, without the idle charge (an unused card is the one that can
 * block): -RATE per card committed, + attack power threatened when a link
 * resolves, +RATE per card drawn, +1 per action point gained (not the turn's
 * first), +1 per life or non-pitch resource gained.
 */
import { applyIntent, legalIntents, projectStateFor, type GameState } from "@fyendal/engine";
import type { GameIntent } from "@fyendal/shared";
import { RATE, committedCards, isMatchIntent } from "./fab-value.js";

type Seat = 0 | 1;

/** A blocking opponent's card must stop at least this much (wounding-bot.ts). */
const MIN_STOPPED_PER_CARD = 2;

/** Hit rate per attack power, from resolved links of real games. */
export class HitRates {
  private readonly counts = new Map<number, [number, number]>();

  private bucket(power: number): number {
    return Math.max(0, Math.min(10, Math.round(power)));
  }

  record(power: number, hit: boolean): void {
    const key = this.bucket(power);
    const row = this.counts.get(key) ?? [0, 0];
    row[0] += hit ? 1 : 0;
    row[1] += 1;
    this.counts.set(key, row);
  }

  /** Laplace-smoothed: 0.5 with no data. */
  rate(power: number): number {
    const [hits, total] = this.counts.get(this.bucket(power)) ?? [0, 0];
    return (hits + 1) / (total + 2);
  }
}

export interface LookaheadResult {
  /** Expected value of the best line found. */
  value: number;
  /** Probability (over hit/no-hit branches) that each of the player's hand
   * cards is committed by the best line. Hand cards absent were not used. */
  uses: Record<number, number>;
  transitions: number;
}

export interface LookaheadOptions {
  /** Applied intents the search may spend. */
  budget?: number;
  /** Softmax temperature (value points) for combining the lines through
   * each option at the first decision of the searched turn. */
  temperature?: number;
  /** Ids of cards this player has pitched this game (known deck bottom). */
  pitchedEver?: ReadonlySet<number>;
  /** Shuffle seed. */
  seed?: number;
  /** A blocking opponent (owner, 2026-10-15; off by default). Instead of the
   * hit / no-hit chance node, the opponent blocks each of our attacks the way
   * the Wounding Blow bot does, with `cards` unseen cards of `defense` each:
   * a card blocks while it stops at least 2 damage and more than `reserve`
   * cards remain (all of them facing lethal). A block is a damage shield
   * worth the defence, so power added after it -- attack reactions, weapon
   * pumps -- lands. */
  blockers?: { cards: number; defense: number; reserve: number };
  /** Credit the attacker each point of damage that lands on the opponent
   * (fab-value.ts creditDamageDealt), so a blocked attack is worth less. */
  creditDamage?: boolean;
}

function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4_294_967_296;
  };
}

/** The opponent-free copy. */
export function lookaheadCopy(state: GameState, seat: Seat, opts: LookaheadOptions = {}): GameState {
  const { cardsRef, scriptsRef, log: _log, gameStats, ...serializable } = state;
  const currentTurnStats = gameStats.turns.at(-1);
  const copy = JSON.parse(JSON.stringify({
    ...serializable,
    gameStats: { turns: currentTurnStats ? [currentTurnStats] : [] },
    log: [],
  })) as GameState;
  copy.cardsRef = cardsRef;
  copy.scriptsRef = scriptsRef;

  // Own deck: keep the known bottom (cards pitched earlier, in order), shuffle the rest.
  const me = copy.players[seat]!;
  const pitched = opts.pitchedEver ?? new Set<number>();
  let keep = 0;
  while (keep < me.deck.length && pitched.has(me.deck[me.deck.length - 1 - keep]!.instanceId)) keep++;
  const unknown = me.deck.slice(0, me.deck.length - keep);
  const known = me.deck.slice(me.deck.length - keep);
  const random = rng(opts.seed ?? 1);
  for (let i = unknown.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [unknown[i], unknown[j]] = [unknown[j]!, unknown[i]!];
  }
  me.deck = [...unknown, ...known];

  // Opponent: nothing hidden survives.
  const opp = copy.players[1 - seat]!;
  opp.hand = [];
  opp.deck = [];
  opp.arsenal = opp.arsenal.filter((card) => !card.faceDown);
  return copy;
}

/** The passive opponent's pick: never block, react or play. */
function passive(legal: GameIntent[]): GameIntent | undefined {
  const pass = legal.find((i) => i.kind === "pass");
  const noBlock = legal.find((i) => i.kind === "defend" && i.instanceIds.length === 0);
  if (noBlock) return noBlock;
  if (pass) return pass;
  const decline = legal.find((i) => i.kind === "choose" && /^(no|decline|done|keep|skip|pass)/i.test(i.optionId));
  return decline ?? legal.find((i) => !isMatchIntent(i));
}

function actor(state: GameState): Seat {
  return (state.pendingDecision?.player ?? state.priorityPlayer) as Seat;
}

/** Every legal intent, plays first, pass last. Pitch ORDER is part of the
 * option: pitching stops once the cost is paid, so for a cost of 3
 * red-then-blue pitches both cards and blue-then-red only the blue -- the
 * engine advertises each order that is legal and every one is searched. */
function candidates(state: GameState, seat: Seat): GameIntent[] {
  const legal = legalIntents(state, seat).filter((i) => !isMatchIntent(i) && i.kind !== "stage-defenders");
  const order = (i: GameIntent) => (i.kind === "pass" ? 2 : i.kind === "choose" ? 1 : 0);
  return [...legal].sort((x, y) => order(x) - order(y));
}

/** Own-side ledger value of one transition. */
function stepValue(before: GameState, after: GameState, seat: Seat, intent: GameIntent | null,
  pitchOf: (cardId: string) => number, creditDamage = false): number {
  let v = intent ? -RATE * committedCards(intent) : 0;
  if (creditDamage) {
    const lost = before.players[1 - seat]!.life - after.players[1 - seat]!.life;
    if (lost > 0) v += lost;
  }
  const b = before.players[seat]!;
  const a = after.players[seat]!;
  // Threatened power of links that resolved in this transition (attacker = seat).
  for (let i = 0; i < after.chain.length; i++) {
    const link = after.chain[i]!;
    if (link.attacker !== seat || !link.resolved || before.chain[i]?.resolved) continue;
    v += Math.max(0, link.finalAttack ?? 0);
  }
  if (after.turn !== before.turn) return v;
  const drawn = (Number(a.flags.cardsDrawnThisTurn) || 0) - (Number(b.flags.cardsDrawnThisTurn) || 0);
  if (drawn > 0) v += RATE * drawn;
  const turnAp = before.phase === "start" && after.phase !== "start" && after.activePlayer === seat ? 1 : 0;
  const ap = a.actionPoints - b.actionPoints - turnAp;
  if (ap > 0) v += ap;
  if (a.life > b.life) v += a.life - b.life;
  const pitchedBefore = new Set(b.pitch.map((c) => c.instanceId));
  let pitched = 0;
  for (const card of a.pitch) if (!pitchedBefore.has(card.instanceId)) pitched += pitchOf(card.cardId);
  const gained = a.resources - b.resources - pitched;
  if (gained > 0) v += gained;
  return v;
}

/** Hand cards an intent commits. */
function usedIds(intent: GameIntent): number[] {
  switch (intent.kind) {
    case "play-card":
    case "play-from-arsenal":
    case "play-from-zone":
      return [intent.instanceId, ...intent.pitchInstanceIds, ...(intent.alternativeCostCardInstanceIds ?? [])];
    case "activate-ability":
      return [...intent.pitchInstanceIds, ...(intent.alternativeCostCardInstanceIds ?? [])];
    default:
      return [];
  }
}

interface Node { value: number; uses: Map<number, number> }

/**
 * Search `seat`'s own turn from `state` (a lookahead copy). Starts wherever
 * the copy is: on the opponent's turn the passive opponent finishes it first,
 * then `seat`'s next turn is searched. Stops when `seat`'s turn ends.
 */
export function searchOffense(state: GameState, seat: Seat, hitRates: HitRates,
  pitchOf: (cardId: string) => number, opts: LookaheadOptions = {}): LookaheadResult {
  const budget = opts.budget ?? 3000;
  const blockers = opts.blockers;
  const creditDamage = opts.creditDamage === true;
  const temperature = opts.temperature ?? 1;
  let spent = 0;
  // The turn searched: this one if it is ours, else our next one.
  const ourTurn = state.activePlayer === seat ? state.turn : state.turn + 1;
  const done = (s: GameState): boolean => s.winner !== null || s.turn > ourTurn;

  const apply = (s: GameState, who: Seat, intent: GameIntent): GameState | null => {
    spent++;
    const r = applyIntent(s, who, intent);
    return r.ok ? r.state : null;
  };

  // `limit` is the transition count this subtree may reach (a share of the
  // budget): every decision splits what is left evenly among its options,
  // passing first (it is nearly free and is the floor every line must beat).
  // Transpositions: playing the same cards in another order often reaches
  // the same position. Each position is scored once (after the turn's first
  // decision, where the softmax applies) and reused.
  const seen = new Map<string, Node>();
  const positionKey = (s: GameState): string => {
    const { log: _l, gameStats: _g, cardsRef: _c, scriptsRef: _s, ...rest } = s;
    return JSON.stringify(rest);
  };

  const visit = (s: GameState, depth: number, limit: number, own: number): Node => {
    const empty: Node = { value: 0, uses: new Map() };
    if (done(s) || depth > 60) return empty;
    const key = own > 0 ? positionKey(s) : null;
    const known = key ? seen.get(key) : undefined;
    if (known) return known;
    const result = search(s, depth, limit, own, empty);
    if (key) seen.set(key, result);
    return result;
  };

  const search = (s: GameState, depth: number, limit: number, own: number, empty: Node): Node => {
    const who = actor(s);
    // Only our own turn is searched: the opponent, and we ourselves on the
    // opponent's turn (no blocking), play passively.
    if (who !== seat || s.activePlayer !== seat) {
      const legal = legalIntents(s, who).filter((i) => !isMatchIntent(i));
      const pick = passive(legal);
      if (!pick) return empty;
      const link = s.chain.at(-1);
      const ourAttack = s.pendingDecision?.kind === "defend" && link && link.attacker === seat && !link.resolved;
      if (ourAttack && blockers) {
        // Blocking opponent: a shield worth the cards it would block with.
        const view = projectStateFor(s, seat).chain.at(-1);
        let left = Math.max(0, (view?.attackValue ?? 0) - (view?.defenseValue ?? 0));
        const opp = s.players[who]!;
        const lethal = left >= opp.life;
        const spare = Number(opp.flags.lookaheadBlockers ?? blockers.cards);
        let used = 0, shield = 0;
        while (used < spare && left > 0) {
          const stops = Math.min(blockers.defense, left);
          if (!lethal && (stops < MIN_STOPPED_PER_CARD || spare - used <= blockers.reserve)) break;
          used++; shield += stops; left -= stops;
        }
        const next = apply(s, who, pick);
        if (!next) return empty;
        const target = next.players[who]!;
        target.flags.lookaheadBlockers = spare - used;
        if (shield > 0) target.flags.preventNextDamage = (Number(target.flags.preventNextDamage) || 0) + shield;
        return sub(s, next, null, depth, limit, own);
      }
      if (ourAttack && spent + 2 <= limit) {
        // Chance node: hit / no hit, weighted by the measured hit rate.
        const power = Math.max(0, projectStateFor(s, seat).chain.at(-1)?.attackValue ?? 0);
        const p = hitRates.rate(power);
        const hit = apply(s, who, pick);
        const noHit = apply(s, who, pick);
        if (!hit || !noHit) return empty;
        noHit.players[who]!.flags.preventNextDamage = 999;
        const half = spent + Math.floor((limit - spent) / 2);
        const a = sub(s, hit, null, depth, half, own);
        const b = sub(s, noHit, null, depth, limit, own);
        const uses = new Map<number, number>();
        for (const [id, w] of a.uses) uses.set(id, (uses.get(id) ?? 0) + p * w);
        for (const [id, w] of b.uses) uses.set(id, (uses.get(id) ?? 0) + (1 - p) * w);
        return { value: p * a.value + (1 - p) * b.value, uses };
      }
      const next = apply(s, who, pick);
      return next ? sub(s, next, null, depth, limit, own) : empty;
    }
    // Our decision. Out of budget: pass (or the first option).
    const all = candidates(s, seat);
    const pass = all.filter((i) => i.kind === "pass");
    const options = spent < limit ? [...pass, ...all.filter((i) => i.kind !== "pass")] : (pass.length ? pass : all.slice(0, 1));
    const children: { node: Node; intent: GameIntent }[] = [];
    options.forEach((intent, k) => {
      const share = spent + Math.max(1, Math.floor((limit - spent) / (options.length - k)));
      const next = apply(s, seat, intent);
      if (!next) return;
      const node = sub(s, next, intent, depth, Math.max(share, spent), own + 1);
      const uses = new Map(node.uses);
      for (const id of usedIds(intent)) uses.set(id, 1);
      children.push({ node: { value: node.value, uses }, intent });
    });
    if (!children.length) return empty;
    const top = Math.max(...children.map((c) => c.node.value));
    if (own > 0 || temperature <= 0) return children.find((c) => c.node.value === top)!.node;
    // First decision of the turn: every opening's best line, softmax-weighted,
    // so near-equal lines share the credit instead of one winning by a hair.
    const weights = children.map((c) => Math.exp((c.node.value - top) / temperature));
    const total = weights.reduce((x, y) => x + y, 0);
    const uses = new Map<number, number>();
    let value = 0;
    children.forEach((c, i) => {
      const w = weights[i]! / total;
      value += w * c.node.value;
      for (const [id, u] of c.node.uses) uses.set(id, (uses.get(id) ?? 0) + w * u);
    });
    return { value, uses };
  };

  const sub = (before: GameState, after: GameState, intent: GameIntent | null, depth: number,
    limit: number, own: number): Node => {
    const here = stepValue(before, after, seat, intent, pitchOf, creditDamage);
    const rest = visit(after, depth + 1, limit, own);
    return { value: here + rest.value, uses: rest.uses };
  };

  const root = visit(state, 0, budget, 0);
  const hand = new Set(state.players[seat]!.hand.map((c) => c.instanceId));
  const uses: Record<number, number> = {};
  for (const [id, w] of root.uses) if (hand.has(id)) uses[id] = Math.round(w * 1000) / 1000;
  return { value: Math.round(root.value * 1000) / 1000, uses, transitions: spent };
}
