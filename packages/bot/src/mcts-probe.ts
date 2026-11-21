// Scratch probe (not for commit): is a playout search a better teacher than the
// Cindra bot? Plays Cindra bot vs a random CC bot; at sampled Cindra decisions
// (defend decisions, and decisions that offer an activation) scores every
// candidate by K bot-vs-bot playouts to the end of the game, each from a fresh
// determinization of the hidden cards (own deck order below the known bottom;
// the opponent's hand, deck and face-down arsenal redealt from their union).
// The same K determinizations are used for every candidate (common random numbers).
// usage: tsx mcts-probe.ts <games> <playouts K> <sample prob> <max candidates> [seed]
// prints one JSON line per probed position, then a summary line.
import { cardData, precon, scripts } from "@fyendal/cards";
import { applyIntent, createGame, legalIntents, projectStateFor, type GameState } from "@fyendal/engine";
import type { Decklist, GameIntent, GameView } from "@fyendal/shared";
import { botDefinition, type BotDefinition } from "./registry.js";

type Seat = 0 | 1;
const [gamesArg = "2", kArg = "8", probArg = "0.2", capArg = "8", seedArg = "1"] = process.argv.slice(2);
const K = Number(kArg), PROB = Number(probArg), CAP = Number(capArg);
const OPPONENTS = ["cindra", "hala", "ira", "jarl"];
const FAST = process.env.FAST === "1";
const DEFEND_PROB = Number(process.env.DEFEND_PROB ?? probArg);

function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}
function shuffle<T>(a: T[], r: () => number): void {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j]!, a[i]!];
  }
}

function poolDeck(bot: BotDefinition): Decklist {
  const reg = precon(bot.deckId)!;
  return { heroId: reg.pool.heroId, weaponIds: [...reg.pool.weaponIds], equipment: {},
    deck: [...reg.pool.deck, ...(reg.pool.sideboard ?? [])] };
}
function botDeck(bot: BotDefinition, opponent: Decklist, order: "first" | "second"): Decklist {
  const reg = precon(bot.deckId)!;
  return { heroId: reg.pool.heroId, ...bot.presentationFor(opponent, order) };
}
function botView(view: GameView, seat: Seat, decks: Decklist[]): GameView {
  const own = view.players[seat] as { startingDeck?: string[] };
  own.startingDeck = [...decks[seat]!.deck].sort();
  view.players[(1 - seat) as Seat].pitch.sort((a, b) => (a.cardId < b.cardId ? -1 : a.cardId > b.cardId ? 1 : 0));
  return view;
}

function actor(state: GameState): Seat {
  return (state.pendingDecision?.player ?? state.priorityPlayer) as Seat;
}
/** The worker's legal list (rl-worker.ts advance). */
function legalFor(state: GameState, seat: Seat): GameIntent[] {
  const pending = state.pendingDecision;
  let legal = legalIntents(state, seat).filter((i) => i.kind !== "concede");
  if (pending?.kind === "defend") {
    const staged = pending.staged ?? [];
    legal = legal.map((i) => i.kind === "stage-defenders" ? { ...i, instanceIds: [...staged, ...i.instanceIds] } : i);
  } else if (legal.length > 1) {
    legal = legal.filter((i) => (pending === null && i.kind !== "activate-ability") || applyIntent(state, seat, i).ok);
  }
  return legal;
}
/** One bot move; a rejected pick falls back to the first that applies. */
function botStep(state: GameState, seat: Seat, bot: BotDefinition, decks: Decklist[], legal = legalFor(state, seat)): { state: GameState; intent: GameIntent } {
  const intent = bot.chooseIntent({ seat, view: botView(projectStateFor(state, seat), seat, decks), legal, cards: cardData, state: undefined });
  const applied = applyIntent(state, seat, intent);
  if (applied.ok) return { state: applied.state, intent };
  for (const c of legal) {
    const a = applyIntent(state, seat, c);
    if (a.ok) return { state: a.state, intent: c };
  }
  throw new Error(`bot ${bot.id}: nothing applies`);
}

/** A copy with the hidden cards re-dealt, from `seat`'s point of view. */
function determinize(state: GameState, seat: Seat, r: () => number): GameState {
  const { cardsRef, scriptsRef, log: _log, gameStats, ...rest } = state;
  const copy = JSON.parse(JSON.stringify({ ...rest, gameStats: { turns: gameStats.turns.slice(-1) }, log: [] })) as GameState;
  copy.cardsRef = cardsRef;
  copy.scriptsRef = scriptsRef;
  (copy as { rngState: number }).rngState = Math.floor(r() * 2 ** 31) + 1;
  // Own deck: the run of already-pitched cards at the bottom is known.
  const me = copy.players[seat]!;
  let keep = 0;
  while (keep < me.deck.length && (me.deck[me.deck.length - 1 - keep]!.pitchCount ?? 0) > 0) keep++;
  const own = me.deck.slice(0, me.deck.length - keep);
  const ids = own.map((c) => c.cardId);
  shuffle(ids, r);
  own.forEach((c, i) => { c.cardId = ids[i]!; });
  // Opponent: hand, deck and face-down arsenal share one pool of card ids.
  const opp = copy.players[1 - seat]!;
  const hidden = [...opp.hand, ...opp.deck, ...opp.arsenal.filter((c) => c.faceDown)];
  const pool = hidden.map((c) => c.cardId);
  shuffle(pool, r);
  hidden.forEach((c, i) => { c.cardId = pool[i]!; });
  return copy;
}

/** A playout move without the trial-apply filter (38% of a game's time):
 * the bot picks from the raw list, and only a rejected pick re-asks it on the
 * filtered list. */
function fastStep(state: GameState, seat: Seat, bot: BotDefinition, decks: Decklist[]): GameState {
  const pending = state.pendingDecision;
  let legal = legalIntents(state, seat).filter((i) => i.kind !== "concede");
  if (pending?.kind === "defend") {
    const staged = pending.staged ?? [];
    legal = legal.map((i) => i.kind === "stage-defenders" ? { ...i, instanceIds: [...staged, ...i.instanceIds] } : i);
  }
  const intent = bot.chooseIntent({ seat, view: botView(projectStateFor(state, seat), seat, decks), legal, cards: cardData, state: undefined });
  const applied = applyIntent(state, seat, intent);
  return applied.ok ? applied.state : botStep(state, seat, bot, decks).state;
}

function playout(state: GameState, bots: BotDefinition[], decks: Decklist[]): Seat | null {
  let s = state, steps = 0;
  while (s.winner === null && s.turn <= 150 && steps++ < 6000) {
    const seat = actor(s);
    s = FAST ? fastStep(s, seat, bots[seat]!, decks) : botStep(s, seat, bots[seat]!, decks).state;
  }
  return s.winner === 0 || s.winner === 1 ? s.winner : null;
}

const key = (i: GameIntent) => JSON.stringify(i);
function describe(state: GameState, i: GameIntent): string {
  const it = i as Record<string, unknown>;
  const name = (id: unknown) => {
    for (const p of state.players) for (const z of Object.values(p) as unknown[]) {
      if (Array.isArray(z)) {
        for (const c of z as { instanceId?: number; cardId?: string }[]) if (c?.instanceId === id) return cardData[c.cardId!]?.name ?? c.cardId;
      } else if (z && typeof z === "object" && (z as { instanceId?: number }).instanceId === id) return cardData[(z as { cardId: string }).cardId]?.name;
    }
    for (const p of state.players) for (const z of Object.values(p.equipment ?? {})) if (z && (z as { instanceId: number }).instanceId === id) return cardData[(z as { cardId: string }).cardId]?.name;
    return String(id);
  };
  const parts: string[] = [i.kind];
  if (it.instanceId !== undefined) parts.push(String(name(it.instanceId)));
  if (it.sourceInstanceId !== undefined) parts.push(String(name(it.sourceInstanceId)));
  if (Array.isArray(it.instanceIds)) parts.push(`[${it.instanceIds.map(name).join(", ")}]`);
  if (it.optionId !== undefined) parts.push(`option ${String(it.optionId)}`);
  return parts.join(" ");
}

const cindra = botDefinition("cindra")!;
const top = rng(Number(seedArg) * 7919);
const sum = { positions: 0, playouts: 0, ms: 0, disagree: 0, sigDisagree: 0, gainSig: 0, byKind: {} as Record<string, { n: number; sig: number }> };
for (let g = 0; g < Number(gamesArg); g++) {
  const me = (g % 2) as Seat;
  const oppBot = botDefinition(OPPONENTS[Math.floor(top() * OPPONENTS.length)]!)!;
  const bots: BotDefinition[] = [];
  bots[me] = cindra; bots[1 - me] = oppBot;
  const decks = [0, 1].map((s) => botDeck(bots[s]!, poolDeck(bots[1 - s]!), s === 0 ? "first" : "second"));
  let state = createGame({ decklists: decks as [Decklist, Decklist], cards: cardData, scripts, seed: Math.floor(top() * 2 ** 30), startPlayer: 0 });
  let steps = 0;
  while (state.winner === null && state.turn <= 150 && steps++ < 6000) {
    const seat = actor(state);
    const legal = legalFor(state, seat);
    const pending = state.pendingDecision;
    const kind = pending?.kind === "defend" ? ((pending.staged ?? []).length ? "defend_more" : "defend_first")
      : legal.some((i) => i.kind === "activate-ability") ? "activate_offered" : "";
    if (seat === me && legal.length > 1 && kind && top() < (kind.startsWith("defend") ? DEFEND_PROB : PROB)) {
      const t0 = performance.now();
      const pick = botStep(state, seat, cindra, decks, legal).intent;
      let cands = legal.filter((c) => applyIntent(state, seat, c).ok);
      // The bot's pick first (a declarative staging pick may not be advertised), then the rest.
      cands = [pick, ...cands.filter((c) => key(c) !== key(pick))].slice(0, CAP);
      const seeds = Array.from({ length: K }, () => Math.floor(top() * 2 ** 30) + 1);
      const scored = cands.map((c) => {
        const out: number[] = [];
        for (const sd of seeds) {
          const r = rng(sd);
          const a = applyIntent(determinize(state, seat, r), seat, c);
          // Illegal in this determinization (a hidden card it relied on): a loss.
          out.push(a.ok && playout(a.state, bots, decks) === me ? 1 : 0);
        }
        return { c, rate: out.reduce((x, y) => x + y, 0) / K, out };
      });
      const ms = performance.now() - t0;
      const botS = scored[0]!;
      const botRate = botS.rate;
      const best = scored.reduce((a, b) => (b.rate > a.rate ? b : a));
      // Paired over the common seeds (McNemar): seeds the best wins and the
      // bot's pick loses (b) against the reverse (c). With several candidates
      // the max is optimistic, so the bar is 2.5 SDs, not 2.
      let b = 0, c = 0;
      best.out.forEach((w, i) => { if (w && !botS.out[i]) b++; if (!w && botS.out[i]) c++; });
      const sig = b - c > 2.5 * Math.sqrt(Math.max(b + c, 1));
      sum.positions++; sum.playouts += K * scored.length; sum.ms += ms;
      if (key(best.c) !== key(pick) && best.rate > botRate) sum.disagree++;
      if (sig) { sum.sigDisagree++; sum.gainSig += best.rate - botRate; }
      const bk = (sum.byKind[kind] ??= { n: 0, sig: 0 });
      bk.n++; bk.sig += sig ? 1 : 0;
      console.log(JSON.stringify({ g, turn: state.turn, kind, vs: oppBot.id, bot: describe(state, pick), botRate,
        best: describe(state, best.c), bestRate: best.rate, b, c, sig, n: scored.length, ms: Math.round(ms),
        all: scored.map((x) => `${describe(state, x.c)}=${x.rate.toFixed(2)}`) }));
    }
    state = botStep(state, seat, bots[seat]!, decks, legal).state;
  }
  console.error(`game ${g}: winner ${state.winner} (cindra seat ${me}) turn ${state.turn}`);
}
console.log(JSON.stringify({ summary: true, ...sum, msPerPlayout: +(sum.ms / Math.max(sum.playouts, 1)).toFixed(1) }));
