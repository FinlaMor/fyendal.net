// Scratch (not for commit): where a bot-vs-bot game spends its time.
import { cardData, precon, scripts } from "@fyendal/cards";
import { applyIntent, createGame, legalIntents, projectStateFor, type GameState } from "@fyendal/engine";
import type { Decklist, GameIntent, GameView } from "@fyendal/shared";
import { botDefinition, type BotDefinition } from "./registry.js";
type Seat = 0 | 1;
const t: Record<string, number> = { legal: 0, filter: 0, project: 0, choose: 0, apply: 0 };
const time = <T>(k: string, f: () => T): T => { const s = performance.now(); const r = f(); t[k]! += performance.now() - s; return r; };
const pool = (b: BotDefinition): Decklist => { const r = precon(b.deckId)!; return { heroId: r.pool.heroId, weaponIds: [...r.pool.weaponIds], equipment: {}, deck: [...r.pool.deck, ...(r.pool.sideboard ?? [])] }; };
const deck = (b: BotDefinition, o: Decklist, ord: "first" | "second"): Decklist => ({ heroId: precon(b.deckId)!.pool.heroId, ...b.presentationFor(o, ord) });
let decisions = 0;
for (const opp of ["hala", "ira", "jarl"]) {
  const bots = [botDefinition("cindra")!, botDefinition(opp)!];
  const decks = [deck(bots[0]!, pool(bots[1]!), "first"), deck(bots[1]!, pool(bots[0]!), "second")];
  let s: GameState = createGame({ decklists: decks as [Decklist, Decklist], cards: cardData, scripts, seed: 11, startPlayer: 0 });
  let n = 0;
  while (s.winner === null && s.turn <= 150 && n++ < 6000) {
    const seat = (s.pendingDecision?.player ?? s.priorityPlayer) as Seat;
    const pending = s.pendingDecision;
    let legal = time("legal", () => legalIntents(s, seat).filter((i) => i.kind !== "concede"));
    if (pending?.kind === "defend") legal = legal.map((i) => i.kind === "stage-defenders" ? { ...i, instanceIds: [...(pending.staged ?? []), ...i.instanceIds] } : i);
    else if (legal.length > 1) legal = time("filter", () => legal.filter((i) => (pending === null && i.kind !== "activate-ability") || applyIntent(s, seat, i).ok));
    const view = time("project", () => projectStateFor(s, seat)) as GameView;
    (view.players[seat] as { startingDeck?: string[] }).startingDeck = [...decks[seat]!.deck].sort();
    const intent = time("choose", () => bots[seat]!.chooseIntent({ seat, view, legal, cards: cardData, state: undefined })) as GameIntent;
    let a = time("apply", () => applyIntent(s, seat, intent));
    if (!a.ok) a = legal.map((c) => applyIntent(s, seat, c)).find((x) => x.ok)!;
    s = (a as { state: GameState }).state;
    decisions++;
  }
  console.log(opp, "turn", s.turn, "winner", s.winner);
}
console.log("decisions", decisions, Object.fromEntries(Object.entries(t).map(([k, v]) => [k, Math.round(v)])));
