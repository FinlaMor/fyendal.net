/**
 * RL self-play worker: hosts many headless games and speaks JSON lines on
 * stdin/stdout, one request per line, one response per line, in order.
 *
 *   {"op":"catalog"}                          -> {"cards":[[id,name,pitch,type,subtypes],...]}
 *   {"op":"new","id":G,"decks":[D,D],"seed":S,"start":0|1,"maxTurns":T,
 *    "bots":[name|null,name|null],"lookahead":bool,"chargeDamage":bool,"creditDamage":bool,"damageOnce":bool,"blockCredit":number,
 *    "lookaheadBlocks":bool}
 *     A seat named in "bots" is played inside the worker by that registered
 *     bot (packages/bot/src/registry.ts) with its own precon, presented
 *     against the other seat's deck; its deck entry may be null.
 *                                             -> observation or terminal
 *   {"op":"step","id":G,"index":I}            -> observation or terminal
 *   {"op":"botpick","id":G,"bot":name}    -> {"id":G,"index":I,"fallback":bool} (bot's pick, not applied)
 *   {"op":"close","id":G}                     -> {"id":G,"closed":true}
 *
 * A deck D is {"hero":key,"equipment":[key...],"deck":[key...]} where a key is
 * a functional key ("name|pitch") or a printing id, or {"bot":name}: that
 * registered bot's precon, sideboarded by the bot's own rule against the
 * opponent. The response to "new" also carries "starting": each seat's
 * starting main deck as card ids. Equipment keys are sorted
 * into weapons, one card per equipment slot, and inventory (the rest).
 *
 * An observation is {"id","seat","kind","turn","view","legal"}: the acting
 * seat's projected view (no hidden information) and its legal intents; with
 * The view also carries "knownBottom": the acting seat's known deck bottom
 * (card ids, draw order). With "lookahead" also "plan": {"value", "uses": {handInstanceId: probability}}
 * from lookahead.ts ("lookaheadBlocks": its opponent blocks, and with
 * "creditDamage" its line value credits the damage that lands). A
 * terminal is {"id","done":true,"winner","turn","values","seats"}, where
 * values[i] is the FaB value (fab-value.ts) of the i-th decision of the
 * game and seats[i] the seat that made it.
 * A game past maxTurns (default 150 single turns) or MAX_STEPS ends with
 * winner null. Any failure is {"id","error"}; the game is dropped.
 */
import { createInterface } from "node:readline";
import { cardData, precon, scripts } from "@fyendal/cards";
import { applyIntent, createGame, legalIntents, projectStateFor } from "@fyendal/engine";
import type { Decklist, EquipmentSlot, GameIntent, GameView } from "@fyendal/shared";
import { ValueBook } from "./fab-value.js";
import { HitRates, lookaheadCopy, searchOffense, type LookaheadResult } from "./lookahead.js";
import { botDefinition, type BotDefinition } from "./registry.js";
import type { RecordedDecision } from "./recording.js";
import { chooseWoundingIntent, woundingBotDecklist } from "./wounding-bot.js";

type GameState = ReturnType<typeof createGame>;

/** A deck by its cards, or a registered bot's precon ({"bot": name}),
 * presented (sideboarded) by that bot's own rule against the opponent. */
type DeckSpec = { hero: string; equipment: string[]; deck: string[] } | { bot: string };

interface Game {
  state: GameState;
  legal: GameIntent[];
  seat: 0 | 1;
  kind: string;
  rows: RecordedDecision[];
  steps: number;
  maxTurns: number;
  /** Per seat: a registered bot that plays that seat inside the worker. */
  bots: [BotDefinition | null, BotDefinition | null];
  decks: [Decklist, Decklist];
  book: ValueBook;
  /** Offensive lookahead (lookahead.ts) for policy seats, when requested. */
  lookahead: boolean;
  pitchedEver: [Set<number>, Set<number>];
  resolvedLinks: Set<string>;
  plans: [{ key: string; plan: LookaheadResult } | null, { key: string; plan: LookaheadResult } | null];
  /** Bot picks the engine rejected, replaced by the first applicable intent. */
  botFallbacks: number;
  /** Return the public game log with the terminal response. */
  record: boolean;
  /** When recording: the whole log (the engine keeps only its last 200
   * lines), turn headers and each seat's choices; the last log texts seen. */
  trace?: string[];
  traceSeen?: string[];
  traceTurn?: number;
  /** Lookahead settings for this game (defaults: both turns, LOOKAHEAD_BUDGET). */
  lookaheadBudget: number;
  lookaheadOwnTurn: boolean;
  /** The lookahead's opponent blocks (lookahead.ts blockers). */
  lookaheadBlocks: boolean;
  /** The lookahead credits damage dealt (with lookaheadBlocks and creditDamage). */
  lookaheadCredit: boolean;
}

/** Transitions per lookahead search: full width, cut at this many steps
 * (~1.4 ms each; owner 2026-10-07: 3000 on both turns). */
const LOOKAHEAD_BUDGET = 3000;

/** Lookahead cost counters, reported by {"op":"stats"}. */
const lookaheadStats = { calls: 0, ms: 0, steps: 0 };

/** Hit rate per attack power over every real game this worker has run. */
const hitRates = new HitRates();

/** Bookkeeping at every decision point: pitched cards (the known deck
 * bottom) and resolved links (the hit rates). */
function track(game: Game): void {
  const state = game.state;
  for (const seat of [0, 1] as const) {
    for (const card of state.players[seat]!.pitch) game.pitchedEver[seat].add(card.instanceId);
  }
  state.chain.forEach((link, index) => {
    if (!link.resolved) return;
    const key = `${state.turn}:${index}:${link.attackingCard.instanceId}`;
    if (game.resolvedLinks.has(key)) return;
    game.resolvedLinks.add(key);
    hitRates.record(link.finalAttack ?? 0, link.hit);
  });
}

/** The lookahead for a policy seat: computed at the first action-phase
 * decision of its turn (the rest of that turn) and at its first defend
 * decision of each opponent turn (its next turn), reused in between. */
function plan(game: Game, seat: 0 | 1): LookaheadResult | null {
  const state = game.state;
  const link = state.chain.at(-1);
  let key: string | null = null;
  // Both turns (owner, 2026-10-07: "give the lookahead the best chance"):
  // at the first action-phase decision of our turn, plan the rest of it; at
  // our first defend decision of each opponent turn, plan our next turn (our
  // hand changes only by what we block, so one plan per opponent turn).
  if (game.lookaheadOwnTurn && state.activePlayer === seat && state.phase === "action" && !state.pendingDecision) key = `own:${state.turn}`;
  else if (state.pendingDecision?.kind === "defend" && link) key = `def:${state.turn}`;
  const cached = game.plans[seat];
  if (key && cached?.key !== key) {
    const copy = lookaheadCopy(state, seat, { pitchedEver: game.pitchedEver[seat], seed: state.turn * 7919 + seat });
    const started = performance.now();
    const opp = state.players[1 - seat]!;
    // Unseen blockers: the opponent's hand now, or after it draws up on its
    // turn when we plan our next turn from a defend decision.
    const cards = key.startsWith("def:") ? Math.max(opp.hand.length, opp.intellect ?? 4) : opp.hand.length;
    const blockers = game.lookaheadBlocks ? { cards, defense: 3, reserve: opp.arsenal.length > 0 ? 0 : 1 } : undefined;
    const result = searchOffense(copy, seat, hitRates, (cardId) => cardData[cardId]?.pitch ?? 0,
      { budget: game.lookaheadBudget, blockers, creditDamage: game.lookaheadCredit });
    lookaheadStats.calls++;
    lookaheadStats.ms += performance.now() - started;
    lookaheadStats.steps += result.transitions;
    game.plans[seat] = { key, plan: result };
  }
  // A plan is about the current turn (own) or our next one (made while defending).
  const current = game.plans[seat];
  if (!current) return null;
  const madeOn = Number(current.key.split(":")[1]);
  return madeOn === state.turn ? current.plan : null;
}

const MAX_STEPS = 6_000;
const SLOTS: EquipmentSlot[] = ["head", "chest", "arms", "legs"];

/** The cards package's functional key ("name|pitch"), which it does not export. */
function functionalKey(name: string, pitch?: number): string {
  return `${name.trim().toLowerCase().replace(/\s+/g, " ")}|${pitch ?? 0}`;
}

const byKey = new Map<string, string>();
for (const card of Object.values(cardData)) {
  const key = functionalKey(card.name, card.pitch);
  if (!byKey.has(key)) byKey.set(key, card.id);
}

function resolve(key: string): string {
  if (cardData[key]) return key;
  const id = byKey.get(key);
  if (!id) throw new Error(`unknown card key "${key}"`);
  return id;
}

function decklist(spec: Exclude<DeckSpec, { bot: string }>): Decklist {
  const weaponIds: string[] = [];
  const equipment: Partial<Record<EquipmentSlot, string>> = {};
  const inventory: string[] = [];
  for (const key of spec.equipment) {
    const id = resolve(key);
    const data = cardData[id]!;
    if (data.cardType === "weapon") {
      weaponIds.push(id);
      continue;
    }
    const slot = SLOTS.find((s) => (data.subtypes ?? []).includes(s));
    if (slot && !equipment[slot]) equipment[slot] = id;
    else inventory.push(id);
  }
  return {
    heroId: resolve(spec.hero),
    weaponIds,
    equipment,
    deck: spec.deck.map(resolve),
    ...(inventory.length ? { inventory } : {}),
  };
}

/** A bot's deck as recording.ts presents it: its registered pool, sideboarded
 * by its own presentation rule against the opponent's list. */
function botDeck(bot: BotDefinition, opponent: Decklist, order: "first" | "second"): Decklist {
  if (bot.id === (WOUNDING as string)) return woundingBotDecklist(resolve);
  const registered = precon(bot.deckId);
  if (!registered) throw new Error(`missing bot deck ${bot.deckId}`);
  return { heroId: registered.pool.heroId, ...bot.presentationFor(opponent, order) };
}

/** A bot's registered pool as a plain list: the reference a precon is
 * sideboarded against when the opponent is a precon too (recording.ts). */
function poolDeck(bot: BotDefinition): Decklist {
  if (bot.id === (WOUNDING as string)) return woundingBotDecklist(resolve);
  const registered = precon(bot.deckId);
  if (!registered) throw new Error(`missing bot deck ${bot.deckId}`);
  return {
    heroId: registered.pool.heroId,
    weaponIds: [...registered.pool.weaponIds],
    equipment: {},
    deck: [...registered.pool.deck, ...(registered.pool.sideboard ?? [])],
  };
}

/** The teaching bot (wounding-bot.ts): not a registered precon, so it lives
 * only here -- "wounding" as a bot seat or a {"bot":"wounding"} deck. */
const WOUNDING = "wounding";
const woundingBot = {
  id: WOUNDING,
  format: "cc",
  deckId: "",
  username: "Wounding Blow Bot",
  deckName: "Wounding Blow",
  chooseIntent: chooseWoundingIntent,
  chooseDecision: (input: Parameters<typeof chooseWoundingIntent>[0]) => ({ intent: chooseWoundingIntent(input) }),
  presentationFor: () => {
    const { heroId: _hero, ...presented } = woundingBotDecklist(resolve);
    return presented;
  },
} as unknown as BotDefinition;

function definition(name: string): BotDefinition {
  if (name === WOUNDING) return woundingBot;
  const bot = botDefinition(name);
  if (!bot) throw new Error(`unknown bot ${name}`);
  return bot;
}

/** recording.ts applyBotInfoRules: the bot knows its own starting list
 * (unordered) and sees the opponent's pitch unordered. */
function botView(view: GameView, seat: 0 | 1, decks: [Decklist, Decklist]): GameView {
  const own = view.players[seat] as { startingDeck?: string[] };
  own.startingDeck = [...decks[seat].deck].sort();
  view.players[(1 - seat) as 0 | 1].pitch.sort((a, b) => (a.cardId < b.cardId ? -1 : a.cardId > b.cardId ? 1 : 0));
  return view;
}

/** The view sent to Python: everything but the human log. */
function slimView(view: GameView): GameView {
  const { log: _log, logEntries: _entries, ...rest } = view;
  return { ...rest, log: [] } as GameView;
}

/** A free-form "name a card" decision has no advertised intents: offer the
 * names of every card the deciding seat can see, which the engine validates. */
function nameIntents(view: GameView): GameIntent[] {
  const names = new Set<string>();
  const add = (cards: readonly { cardId: string }[] | undefined) => {
    for (const card of cards ?? []) {
      const name = cardData[card.cardId]?.name;
      if (name) names.add(name);
    }
  };
  for (const p of view.players) {
    add(p.hand); add(p.arsenal); add(p.pitch); add(p.graveyard); add(p.banish);
    add(p.board); add(p.weapons); add(Object.values(p.equipment).filter((c) => !!c) as { cardId: string }[]);
  }
  for (const link of view.chain) add([link.attackingCard, ...link.defendingCards]);
  return [...names].sort().slice(0, 24).map((name) => ({ kind: "choose", optionId: name }) as GameIntent);
}

const games = new Map<number, Game>();

/** Card name of an instance anywhere in the players' zones. */
function instanceName(state: GameState, instanceId: unknown): string {
  let found: string | undefined;
  const walk = (node: unknown, depth: number): void => {
    if (found || depth > 4 || node === null || typeof node !== "object") return;
    const rec = node as Record<string, unknown>;
    if (rec.instanceId === instanceId && typeof rec.cardId === "string") {
      found = cardData[rec.cardId]?.name ?? rec.cardId;
      return;
    }
    for (const value of Object.values(rec)) walk(value, depth + 1);
  };
  walk(state.players, 0);
  return found ?? String(instanceId);
}

function describeIntent(state: GameState, intent: GameIntent): string {
  const it = intent as Record<string, unknown>;
  const parts: string[] = [intent.kind];
  if (it.instanceId !== undefined) parts.push(instanceName(state, it.instanceId));
  if (it.sourceInstanceId !== undefined) parts.push(instanceName(state, it.sourceInstanceId));
  if (Array.isArray(it.instanceIds)) parts.push(`[${it.instanceIds.map((x) => instanceName(state, x)).join(", ")}]`);
  if (Array.isArray(it.pitchInstanceIds) && it.pitchInstanceIds.length) {
    parts.push(`pitch [${it.pitchInstanceIds.map((x) => instanceName(state, x)).join(", ")}]`);
  }
  if (it.optionId !== undefined) parts.push(`option ${String(it.optionId)}`);
  return parts.join(" ");
}

/** Append the log lines added since the last call (the engine drops its
 * oldest lines past 200) and a header at each new turn. */
function traceLog(game: Game): void {
  const state = game.state;
  const texts = state.log.map((entry) => entry.publicText).filter((text): text is string => !!text);
  const prev = game.traceSeen ?? [];
  let drop = 0;
  while (drop < prev.length && !prev.slice(drop).every((text, k) => texts[k] === text)) drop++;
  game.trace ??= [];
  game.trace.push(...texts.slice(prev.length - drop));
  game.traceSeen = texts;
  if (game.traceTurn !== state.turn) {
    game.traceTurn = state.turn;
    const p = state.players.map((player) => {
      const pl = player as unknown as Record<string, unknown[] | number>;
      const n = (key: string) => (Array.isArray(pl[key]) ? (pl[key] as unknown[]).length : "?");
      return `life ${pl.life} hand ${n("hand")} deck ${n("deck")} arsenal ${n("arsenal")}`;
    });
    game.trace.push(`=== TURN ${state.turn} (active P${state.activePlayer}) | P0 ${p[0]} | P1 ${p[1]} | value P0 ${game.book.total(0).toFixed(1)} P1 ${game.book.total(1).toFixed(1)}`);
  }
}

/** The card whose script owns the pending choose-target / optional-effect
 * decision, when the deciding seat may know it: its own card, or one the
 * projected view already shows. Prompts like "Pay 1 to create Vigor?" name no
 * card, so without this the model cannot tell whose effect it is answering. */
function decisionSource(state: GameState, view: GameView, seat: 0 | 1): { instanceId: number; cardId: string; owner: number } | null {
  const pd = state.pendingDecision as (typeof state.pendingDecision & {
    sourceInstanceId?: number; scriptSourceSnapshot?: { seat: number; card: { instanceId: number; cardId: string } };
  }) | null;
  const id = pd?.sourceInstanceId ?? pd?.scriptSourceSnapshot?.card.instanceId;
  if (id === undefined) return null;
  let hit: { cardId: string; owner: number } | null = null;
  const find = (node: unknown, owner: number, depth: number): void => {
    if (hit || depth > 5 || node === null || typeof node !== "object") return;
    const rec = node as Record<string, unknown>;
    if (rec.instanceId === id && typeof rec.cardId === "string") {
      hit = { cardId: rec.cardId, owner: typeof rec.owner === "number" ? rec.owner : owner };
      return;
    }
    for (const value of Object.values(rec)) find(value, owner, depth + 1);
  };
  state.players.forEach((p, i) => find(p, i, 0));
  if (!hit) find(state.stack, -1, 0);
  if (!hit) find(state.chain, -1, 0);
  const snap = pd?.scriptSourceSnapshot;
  if (!hit && snap && snap.card.instanceId === id) hit = { cardId: snap.card.cardId, owner: snap.seat };
  if (!hit) return null;
  const found = hit as { cardId: string; owner: number };
  if (found.owner !== seat) {
    // Not ours: only if the view shows that very card face up.
    let shown = false;
    const look = (node: unknown, depth: number): void => {
      if (shown || depth > 6 || node === null || typeof node !== "object") return;
      const rec = node as Record<string, unknown>;
      if (rec.instanceId === id && rec.cardId === found.cardId) { shown = true; return; }
      for (const value of Object.values(rec)) look(value, depth + 1);
    };
    look(view, 0);
    if (!shown) return null;
  }
  return { instanceId: id, cardId: found.cardId, owner: found.owner };
}

function advance(id: number, game: Game): object {
  const state = game.state;
  track(game);
  if (game.record) traceLog(game);
  if (state.winner !== null || game.steps >= MAX_STEPS || state.turn > game.maxTurns) {
    games.delete(id);
    const winner = state.winner === 0 || state.winner === 1 ? state.winner : null;
    game.book.observe(game.rows, projectStateFor(state, 0), state);
    return {
      id, done: true, winner, turn: state.turn,
      values: game.book.result(game.rows),
      seats: game.rows.map((row) => row.seat),
      botFallbacks: game.botFallbacks,
      ...(game.record ? { log: game.trace ?? [] } : {}),
    };
  }
  const seat = (state.pendingDecision?.player ?? state.priorityPlayer) as 0 | 1;
  const view = projectStateFor(state, seat);
  game.book.observe(game.rows, view, state);
  const pending = state.pendingDecision;
  let legal = pending?.kind === "choose-name"
    ? nameIntents(view)
    : legalIntents(state, seat).filter((intent) => intent.kind !== "concede");
  // Staging is declarative: stage-defenders REPLACES the staged set, and
  // legalIntents advertises one single-card intent per card that may join
  // it. Sent as advertised, each pick swaps the staged card and a defend
  // decision never ends; make each one "the staged set plus this card".
  if (pending?.kind === "defend") {
    const staged = pending.staged ?? [];
    legal = legal.map((intent) => intent.kind === "stage-defenders"
      ? { ...intent, instanceIds: [...staged, ...intent.instanceIds] }
      : intent);
  }
  // legalIntents can advertise intents applyIntent then rejects (a pass at a
  // nested reroll/prevention prompt; an activation whose effect-card cost
  // cannot be paid); keep only what applies. Defend staging is exempt: its
  // intents are declarative and always offered.
  if (pending?.kind !== "defend" && legal.length > 1) {
    // At an open priority window only activations have been seen to fail;
    // trial-applying every play there halves throughput for nothing.
    legal = legal.filter((intent) =>
      (pending === null && intent.kind !== "activate-ability") || applyIntent(state, seat, intent).ok);
  }
  if (legal.length === 0) {
    const pd = state.pendingDecision;
    const other = legalIntents(state, (1 - seat) as 0 | 1).filter((intent) => intent.kind !== "concede").map((i) => i.kind);
    throw new Error(`seat ${seat} has no legal intent; other seat has ${JSON.stringify(other)}; stack ${state.stack.length}; chain ${state.chain.length} (decision ${pd?.kind ?? "none"} for ${pd?.player ?? "-"}: ${pd?.prompt ?? ""}; options ${JSON.stringify(pd?.options ?? [])}; phase ${state.phase})`);
  }
  game.legal = legal;
  game.seat = seat;
  game.kind = state.pendingDecision?.kind ?? "priority";
  const bot = game.bots[seat];
  if (bot) {
    record(id, game);
    let intent = bot.chooseIntent({ seat, view: botView(view, seat, game.decks), legal, cards: cardData, state: undefined });
    let applied = applyIntent(state, seat, intent);
    if (!applied.ok) {
      // A bot staging defenders declaratively, or a stale pick: fall back to
      // the first advertised intent that applies rather than lose the game.
      const fallback = legal.find((candidate) => applyIntent(state, seat, candidate).ok);
      if (!fallback) throw new Error(`bot ${bot.id} ${intent.kind} rejected: ${applied.error}`);
      game.botFallbacks++;
      intent = fallback;
      applied = applyIntent(state, seat, intent);
    }
    const row = game.rows[game.rows.length - 1]!;
    row.chosen = intent;
    if (game.record && legal.length > 1) game.trace?.push(`  > BOT P${seat} (${game.kind}): ${describeIntent(state, intent)}`);
    game.book.commit(game.rows, intent);
    game.state = (applied as { state: GameState }).state;
    game.steps++;
    return advance(id, game);
  }
  record(id, game);
  const lookahead = game.lookahead ? plan(game, seat) : null;
  return { id, seat, kind: game.kind, turn: state.turn, row: game.rows.length - 1, view: { ...slimView(view), knownBottom: knownBottom(game, seat), decisionSource: decisionSource(state, view, seat) }, legal,
    ...(lookahead ? { plan: { value: lookahead.value, uses: lookahead.uses, steps: lookahead.transitions } } : {}) };
}

/** The acting seat's known deck bottom, in draw order: the run of cards it
 * pitched earlier this game at the bottom of its deck (as lookaheadCopy
 * keeps them). Card ids only; the seat legitimately knows them. */
function knownBottom(game: Game, seat: 0 | 1): string[] {
  const deck = game.state.players[seat]!.deck;
  const pitched = game.pitchedEver[seat];
  let keep = 0;
  while (keep < deck.length && pitched.has(deck[deck.length - 1 - keep]!.instanceId)) keep++;
  return deck.slice(deck.length - keep).map((card) => card.cardId);
}

function record(id: number, game: Game): void {
  const state = game.state;
  game.rows.push({
    episodeId: String(id),
    decisionIndex: game.rows.length,
    turn: state.turn,
    seat: game.seat,
    decisionKind: game.kind,
    observation: null,
    fullObservation: null,
    legal: [],
    chosenIndex: -1,
    chosen: { kind: "pass" },
    decisionMs: 0,
  });
}

function handle(req: { op: string; id?: number; decks?: [DeckSpec | null, DeckSpec | null]; seed?: number; start?: number; index?: number; maxTurns?: number; bots?: [string | null, string | null]; lookahead?: boolean; record?: boolean; lookaheadBudget?: number; lookaheadOwnTurn?: boolean; chargeDamage?: boolean; creditDamage?: boolean; damageOnce?: boolean; blockCredit?: number; lookaheadBlocks?: boolean; bot?: string }): object {
  if (req.op === "catalog") {
    return {
      cards: Object.values(cardData).map((c) => [c.id, c.name, c.pitch ?? 0, c.cardType, c.subtypes ?? []]),
    };
  }
  const id = req.id!;
  if (req.op === "new") {
    const bots = (req.bots ?? [null, null]).map((name) => (name ? definition(name) : null)) as
      [BotDefinition | null, BotDefinition | null];
    const start = (req.start ?? 0) as 0 | 1;
    // Each seat's deck: a bot seat plays its own precon; a policy seat plays
    // its spec, which may itself name a bot's precon.
    const precons = [0, 1].map((seat) => {
      const spec = req.decks?.[seat];
      return bots[seat] ?? (spec && "bot" in spec ? definition(spec.bot) : null);
    });
    const given = [0, 1].map((seat) => (precons[seat] ? null : decklist(req.decks![seat] as Exclude<DeckSpec, { bot: string }>)));
    const decks = [0, 1].map((seat) => {
      const bot = precons[seat];
      if (!bot) return given[seat]!;
      const other = precons[1 - seat];
      const opponent = given[1 - seat] ?? poolDeck(other!);
      return botDeck(bot, opponent, seat === start ? "first" : "second");
    }) as [Decklist, Decklist];
    const state = createGame({
      decklists: decks,
      cards: cardData,
      scripts,
      seed: req.seed ?? 0,
      startPlayer: start,
    });
    const game: Game = { state, legal: [], seat: 0, kind: "", rows: [], steps: 0, maxTurns: req.maxTurns ?? 150, bots, decks,
      book: new ValueBook((cardId) => cardData[cardId]?.pitch ?? 0, { chargeDamageTaken: req.chargeDamage === true, creditDamageDealt: req.creditDamage === true,
                                                                                  damageOnce: req.damageOnce === true,
                                                                                  blockCredit: typeof req.blockCredit === "number" ? req.blockCredit : undefined }),
      lookahead: req.lookahead === true, pitchedEver: [new Set(), new Set()], resolvedLinks: new Set(), plans: [null, null],
      botFallbacks: 0, record: req.record === true,
      lookaheadBudget: req.lookaheadBudget ?? LOOKAHEAD_BUDGET, lookaheadOwnTurn: req.lookaheadOwnTurn ?? true,
      lookaheadBlocks: req.lookaheadBlocks === true, lookaheadCredit: req.lookaheadBlocks === true && req.creditDamage === true };
    games.set(id, game);
    // Each seat's starting main deck (card ids): a player knows its own list,
    // which the policy's deck-composition tokens are computed from.
    return { ...advance(id, game), starting: decks.map((deck) => deck.deck) };
  }
  if (req.op === "step") {
    const game = games.get(id);
    if (!game) throw new Error(`no game ${id}`);
    const intent = game.legal[req.index!];
    if (!intent) throw new Error(`index ${req.index} out of range (${game.legal.length})`);
    const row = game.rows[game.rows.length - 1]!;
    const applied = applyIntent(game.state, game.seat, intent);
    if (!applied.ok) {
      const pd = game.state.pendingDecision;
      throw new Error(`${intent.kind} rejected: ${applied.error} (decision ${pd?.kind ?? "none"} for ${pd?.player ?? "-"}: ${pd?.prompt ?? ""}; options ${JSON.stringify(pd?.options ?? [])}; phase ${game.state.phase}; seat ${game.seat}; priority ${game.state.priorityPlayer}; legal ${JSON.stringify(game.legal.map((i) => i.kind))})`);
    }
    row.chosen = intent;
    row.chosenIndex = req.index!;
    if (game.record) {
      game.trace?.push(`  > POLICY P${game.seat} (${game.kind}, ${game.legal.length} options): ${describeIntent(game.state, intent)}`);
      if (game.legal.length > 1) game.trace?.push(`      options: ${game.legal.map((o) => describeIntent(game.state, o)).join(" | ")}`);
    }
    game.book.commit(game.rows, intent);
    game.state = applied.state;
    game.steps++;
    return advance(id, game);
  }
  if (req.op === "botpick") {
    // Imitation data: what the registered bot req.bot would choose at this
    // policy decision, as an index into the advertised legal list. Nothing is
    // applied; the caller steps with the index. A pick that is not one of the
    // advertised intents falls back as a bot seat does (first that applies).
    const game = games.get(id);
    if (!game) throw new Error(`no game ${id}`);
    const bot = definition(req.bot!);
    const view = projectStateFor(game.state, game.seat);
    const intent = bot.chooseIntent({ seat: game.seat, view: botView(view, game.seat, game.decks), legal: game.legal,
                                      cards: cardData, state: undefined });
    let index = game.legal.indexOf(intent);
    if (index < 0) {
      const key = JSON.stringify(intent);
      index = game.legal.findIndex((candidate) => JSON.stringify(candidate) === key);
    }
    if (index >= 0) return { id, index, fallback: false };
    // A bot blocks declaratively: "stage-defenders [A, B, C]" (or "defend
    // [A, B]") in one intent, where the advertised list stages one card at a
    // time (staged set + one card). Walk toward the bot's set: stage its next
    // unstaged card, and once all of it is staged, the final defend.
    const pd = game.state.pendingDecision;
    if (pd?.kind === "defend" && (intent.kind === "stage-defenders" || intent.kind === "defend")) {
      const staged = new Set<number>(pd.staged ?? []);
      const want = (intent as { instanceIds?: number[] }).instanceIds ?? [];
      const next = want.find((card) => !staged.has(card));
      index = next !== undefined
        ? game.legal.findIndex((c) => c.kind === "stage-defenders" && c.instanceIds.at(-1) === next)
        : want.length === staged.size
          ? game.legal.findIndex((c) => c.kind === "defend")
          : -1;
      if (index >= 0) return { id, index, fallback: false };
    }
    index = game.legal.findIndex((candidate) => applyIntent(game.state, game.seat, candidate).ok);
    if (index < 0) throw new Error(`bot ${bot.id} ${intent.kind}: no legal intent applies`);
    return { id, index, fallback: true, intent, staged: game.state.pendingDecision?.kind === "defend" ? (game.state.pendingDecision.staged ?? []) : undefined };
  }
  if (req.op === "stats") {
    return { lookahead: { ...lookaheadStats, ms: Math.round(lookaheadStats.ms) } };
  }
  if (req.op === "close") {
    games.delete(id);
    return { id, closed: true };
  }
  throw new Error(`unknown op ${req.op}`);
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  if (!line.trim()) return;
  let req: Parameters<typeof handle>[0] | undefined;
  let out: object;
  try {
    req = JSON.parse(line);
    out = handle(req!);
  } catch (error) {
    if (req?.id !== undefined) games.delete(req.id);
    out = { id: req?.id ?? null, error: error instanceof Error ? error.message : String(error) };
  }
  process.stdout.write(JSON.stringify(out) + "\n");
});
