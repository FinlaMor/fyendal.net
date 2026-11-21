// Scratch driver (not for commit): plays the Wounding Blow bot and prints a trace / results.
// usage: tsx wounding-bot-drive.ts trace | games <n> <opponent: wounding|cindra|ira|hala|jarl>
import { cardData, precon, scripts } from "@fyendal/cards";
import { applyIntent, createGame, legalIntents, projectStateFor } from "@fyendal/engine";
import type { Decklist, GameIntent } from "@fyendal/shared";
import { isMatchIntent } from "./fab-value.js";
import { botDefinition } from "./registry.js";
import { chooseWoundingIntent, woundingBotDecklist } from "./wounding-bot.js";

const byKey = new Map<string, string>();
for (const card of Object.values(cardData)) {
  const key = `${card.name.trim().toLowerCase().replace(/\s+/g, " ")}|${card.pitch ?? 0}`;
  if (!byKey.has(key)) byKey.set(key, card.id);
}
const resolve = (key: string) => byKey.get(key) ?? (() => { throw new Error(key); })();

type Chooser = (seat: 0 | 1, view: any, legal: GameIntent[]) => GameIntent;

function play(seed: number, start: 0 | 1, opp: string, trace: boolean) {
  const mine = woundingBotDecklist(resolve);
  let other: Decklist = mine;
  let oppChoose: Chooser = (seat, view, legal) => chooseWoundingIntent({ seat, view, legal, cards: cardData });
  if (opp !== "wounding") {
    const bot = botDefinition(opp)!;
    const reg = precon(bot.deckId)!;
    other = { heroId: reg.pool.heroId, ...bot.presentationFor(mine, start === 1 ? "first" : "second") };
    oppChoose = (seat, view, legal) => bot.chooseIntent({ seat, view, legal, cards: cardData });
  }
  let state = createGame({ decklists: [mine, other], cards: cardData, scripts, seed, startPlayer: start });
  let steps = 0, rejected = 0;
  const dmg: [number, number] = [0, 0], turnsOf: [number, number] = [0, 0]; let lastTurn = -1;
  while (state.winner === null && state.turn <= 150 && steps < 20000) {
    if (state.turn !== lastTurn) { lastTurn = state.turn; turnsOf[state.activePlayer as 0 | 1]++; }
    const lifeBefore = [state.players[0].life, state.players[1].life];
    const activeNow = state.activePlayer as 0 | 1;
    const seat = (state.pendingDecision?.player ?? state.priorityPlayer) as 0 | 1;
    const view = projectStateFor(state, seat);
    const pending = state.pendingDecision;
    let legal = legalIntents(state, seat).filter((i) => !isMatchIntent(i));
    if (pending?.kind === "defend") {
      const staged = pending.staged ?? [];
      legal = legal.map((i) => i.kind === "stage-defenders" ? { ...i, instanceIds: [...staged, ...i.instanceIds] } : i);
    } else if (legal.length > 1) {
      legal = legal.filter((i) => (pending === null && i.kind !== "activate-ability") || applyIntent(state, seat, i).ok);
    }
    const intent = seat === 0 ? chooseWoundingIntent({ seat, view, legal, cards: cardData }) : oppChoose(seat, view, legal);
    let applied = applyIntent(state, seat, intent);
    if (!applied.ok) {
      rejected++;
      if (trace) console.log(`  REJECTED ${JSON.stringify(intent)}: ${applied.error}`);
      const fb = legal.find((i) => applyIntent(state, seat, i).ok)!;
      applied = applyIntent(state, seat, fb);
    }
    if (trace && seat === 0 && intent.kind !== "pass") {
      const me = view.players[0], them = view.players[1];
      const link = view.chain[view.chain.length - 1];
      const what = intent.kind === "stage-defenders" ? `stage ${intent.instanceIds.length}`
        : intent.kind === "defend" ? `defend with ${intent.instanceIds.length}`
        : intent.kind === "activate-ability" ? `kodachi (pitch ${intent.pitchInstanceIds.length})`
        : intent.kind;
      const ctx = pending?.kind === "defend" ? ` vs ${link?.attackValue} power` : "";
      if (pending?.kind === "choose-target" && state.turn < 4) console.log(`   prompt: ${pending.prompt} options ${JSON.stringify(pending.options)} chose ${JSON.stringify(intent)}`);
      console.log(`t${state.turn} ${pending?.kind ?? "action"}${ctx}: ${what} | hand ${me.hand.length} arsenal ${me.arsenal.length} life ${me.life} opp ${them.life}`);
    }
    state = (applied as any).state;
    steps++;
    // damage dealt by the active player this step = the other seat's life loss
    dmg[activeNow] += Math.max(0, lifeBefore[1 - activeNow]! - state.players[1 - activeNow]!.life);
  }
  return { winner: state.winner, turn: state.turn, rejected, life: [state.players[0].life, state.players[1].life], dmg, turnsOf };
}

const [mode, n, opp] = process.argv.slice(2);
if (mode === "trace") {
  const r = play(1, 0, n ?? "wounding", true);
  console.log(r);
} else {
  const count = Number(n ?? 20);
  let wins = 0, turns = 0, rejected = 0, unfinished = 0; const dmg: [number, number] = [0, 0], tof: [number, number] = [0, 0];
  for (let k = 0; k < count; k++) {
    const r = play(1000 + k, (k % 2) as 0 | 1, opp ?? "wounding", false);
    if (r.winner === 0) wins++;
    if (r.winner === null) unfinished++;
    turns += r.turn; rejected += r.rejected;
    for (const k of [0, 1] as const) { dmg[k] += r.dmg[k]!; tof[k] += r.turnsOf[k]!; }
  }
  console.log(JSON.stringify({ opp, games: count, wounding_bot_wins: wins, unfinished, mean_turns: turns / count, rejected,
    wounding_dmg_per_turn: +(dmg[0]! / tof[0]!).toFixed(2), opp_dmg_per_turn: +(dmg[1]! / tof[1]!).toFixed(2) }));
}
