// Scratch probe (not for commit): at the start of each of the CC bot's turns vs the
// Wounding Blow bot, score every opening option with the offensive lookahead and
// compare the best weapon-first opening with the best card-first opening.
// usage: tsx lookahead-probe.ts <deck: ira|cindra|hala|jarl> <games> <budget per option>
import { cardData, precon, scripts } from "@fyendal/cards";
import { applyIntent, createGame, legalIntents, projectStateFor, type GameState } from "@fyendal/engine";
import type { Decklist, GameIntent } from "@fyendal/shared";
import { committedCards, RATE } from "./fab-value.js";
import { HitRates, lookaheadCopy, searchOffense } from "./lookahead.js";
import { botDefinition } from "./registry.js";
import { chooseWoundingIntent, woundingBotDecklist } from "./wounding-bot.js";

const byKey = new Map<string, string>();
for (const card of Object.values(cardData)) {
  const key = `${card.name.trim().toLowerCase().replace(/\s+/g, " ")}|${card.pitch ?? 0}`;
  if (!byKey.has(key)) byKey.set(key, card.id);
}
const resolve = (key: string) => byKey.get(key) ?? (() => { throw new Error(key); })();
const pitchOf = (id: string) => cardData[id]?.pitch ?? 0;
const nameOf = (state: GameState, iid: number): string => {
  for (const p of state.players) for (const zone of Object.values(p) as unknown[]) {
    if (Array.isArray(zone)) for (const c of zone as { instanceId?: number; cardId?: string }[]) {
      if (c?.instanceId === iid && c.cardId) return cardData[c.cardId]?.name ?? c.cardId;
    }
  }
  return String(iid);
};

const label = (i: GameIntent): string => {
  const it = i as { instanceId?: number; sourceInstanceId?: number; pitchInstanceIds?: number[] };
  const id = it.instanceId ?? it.sourceInstanceId;
  const pitch = it.pitchInstanceIds?.length ? ` pitch ${it.pitchInstanceIds.map((x) => nameOf(lastState, x)).join("+")}` : "";
  return `${id !== undefined ? nameOf(lastState, id) : ""}${pitch}`;
};
let lastState: GameState;
const [deck = "ira", n = "6", budgetArg = "1000"] = process.argv.slice(2);
const budget = Number(budgetArg);
const bot = botDefinition(deck)!;
const hitRates = new HitRates();
const BLOCKS = process.env.BLOCKS === "1";
const optsFor = (st: GameState) => {
  const opp = st.players[0]!;
  return BLOCKS ? { blockers: { cards: opp.hand.length, defense: 3, reserve: opp.arsenal.length > 0 ? 0 : 1 }, creditDamage: true } : {};
};
const ar = { n: 0, reactBetter: 0, passBetter: 0, tie: 0, botReacted: 0, agree: 0 };
const tally = { turns: 0, both: 0, weaponBetter: 0, cardBetter: 0, tie: 0, botWeaponFirst: 0, gap: 0 };

for (let g = 0; g < Number(n); g++) {
  const start = (g % 2) as 0 | 1;
  const mine = woundingBotDecklist(resolve);
  const reg = precon(bot.deckId)!;
  const other: Decklist = { heroId: reg.pool.heroId, ...bot.presentationFor(mine, start === 1 ? "first" : "second") };
  let state = createGame({ decklists: [mine, other], cards: cardData, scripts, seed: 500 + g, startPlayer: start });
  let lastProbe = -1, steps = 0;
  while (state.winner === null && state.turn <= 150 && steps++ < 20000) {
    const seat = (state.pendingDecision?.player ?? state.priorityPlayer) as 0 | 1;
    const view = projectStateFor(state, seat);
    const pending = state.pendingDecision;
    let legal = legalIntents(state, seat).filter((i) => i.kind !== "concede");
    if (pending?.kind === "defend") {
      const staged = pending.staged ?? [];
      legal = legal.map((i) => i.kind === "stage-defenders" ? { ...i, instanceIds: [...staged, ...i.instanceIds] } : i);
    } else if (legal.length > 1) {
      legal = legal.filter((i) => applyIntent(state, seat, i).ok);
    }
    // Probe: the CC bot's first decision of its action phase.
    if (seat === 1 && state.activePlayer === 1 && state.phase === "action" && !pending && state.chain.length === 0 && lastProbe !== state.turn) {
      lastProbe = state.turn;
      tally.turns++;
      lastState = state;
      const weaponName = (i: GameIntent) => i.kind === "activate-ability" ? nameOf(state, i.sourceInstanceId) : "";
      const isWeapon = (i: GameIntent) => i.kind === "activate-ability" && state.players[1]!.weapons.some((w) => w.instanceId === i.sourceInstanceId);
      const scored: { intent: GameIntent; v: number; weapon: boolean }[] = [];
      for (const intent of legal) {
        if (intent.kind === "pass") continue;
        const applied = applyIntent(state, 1, intent);
        if (!applied.ok) continue;
        const copy = lookaheadCopy(applied.state, 1, { seed: state.turn * 7919 + 1 });
        const r = searchOffense(copy, 1, hitRates, pitchOf, { budget, temperature: 0, ...optsFor(state) });
        scored.push({ intent, v: -RATE * committedCards(intent) + r.value, weapon: isWeapon(intent) });
      }
      const w = scored.filter((s) => s.weapon), c = scored.filter((s) => !s.weapon && s.intent.kind.startsWith("play"));
      const choice = bot.chooseIntent({ seat: 1, view, legal, cards: cardData });
      if (w.length && c.length) {
        tally.both++;
        const bw = Math.max(...w.map((s) => s.v)), bc = Math.max(...c.map((s) => s.v));
        if (bw > bc + 0.01) tally.weaponBetter++; else if (bc > bw + 0.01) tally.cardBetter++; else tally.tie++;
        tally.gap += bw - bc;
        if (choice.kind === "activate-ability" && isWeapon(choice)) tally.botWeaponFirst++;
        if (tally.both <= 8) {
          const top = [...scored].sort((a, b) => b.v - a.v).slice(0, 4)
            .map((s) => `${s.intent.kind} ${label(s.intent)} = ${s.v.toFixed(2)}`);
          console.log(`t${state.turn} weapon best ${bw.toFixed(2)} card best ${bc.toFixed(2)} | bot chose ${choice.kind} ${weaponName(choice)}\n    ${top.join("\n    ")}`);
        }
      }
    }
    // Probe 2: the CC bot's attack-reaction decisions with a card to play.
    if (seat === 1 && pending?.kind === "attack-reaction" && legal.some((i) => i.kind.startsWith("play")) && legal.some((i) => i.kind === "pass")) {
      const score = (intent: GameIntent) => {
        const applied = applyIntent(state, 1, intent);
        if (!applied.ok) return -Infinity;
        const copy = lookaheadCopy(applied.state, 1, { seed: state.turn * 7919 + 3 });
        return -RATE * committedCards(intent) + searchOffense(copy, 1, hitRates, pitchOf, { budget, temperature: 0, ...optsFor(state) }).value;
      };
      if (process.env.TRACE === "1" && legal.some((i) => i.kind === "play-card" && nameOf(state, i.instanceId) === "Swordmaster's Shine")) {
        for (const intent of [legal.find((i) => i.kind === "pass")!, legal.find((i) => i.kind === "play-card" && nameOf(state, i.instanceId) === "Swordmaster's Shine")!]) {
          let st = (applyIntent(state, 1, intent) as { state: GameState }).state;
          const lines: string[] = [`${intent.kind}: opp life ${st.players[0]!.life}`];
          for (let k = 0; k < 12 && st.chain.length && !st.chain.at(-1)!.resolved; k++) {
            const who = (st.pendingDecision?.player ?? st.priorityPlayer) as 0 | 1;
            const lg = legalIntents(st, who).filter((i) => i.kind !== "concede");
            const pk = lg.find((i) => i.kind === "pass") ?? lg[0]!;
            st = (applyIntent(st, who, pk) as { state: GameState }).state;
            const l = projectStateFor(st, 1).chain.at(-1);
            lines.push(`  P${who} ${pk.kind} ${st.pendingDecision?.kind ?? "-"} | link ${l?.attackValue}/${l?.defenseValue} resolved ${st.chain.at(-1)?.resolved} final ${st.chain.at(-1)?.finalAttack} | opp life ${st.players[0]!.life} stack ${st.stack.length}`);
          }
          const ap = (applyIntent(state, 1, intent) as { state: GameState }).state;
          for (const b of [1000, 20000]) {
            const r = searchOffense(lookaheadCopy(ap, 1, { seed: 3 }), 1, hitRates, pitchOf, { budget: b, temperature: 0, ...optsFor(state) });
            lines.push(`  search budget ${b}: value ${r.value} transitions ${r.transitions}`);
          }
          const cp = lookaheadCopy(ap, 1, { seed: 3 });
          lines.push(`  copy: chain ${cp.chain.length} stack ${cp.stack.length} pending ${cp.pendingDecision?.kind} prio ${cp.priorityPlayer} active ${cp.activePlayer} turn ${cp.turn}`);
          console.log(lines.join(" // "));
        }
        process.exit(0);
      }
      const pass = score(legal.find((i) => i.kind === "pass")!);
      const react = Math.max(...legal.filter((i) => i.kind.startsWith("play")).map(score));
      const choice = bot.chooseIntent({ seat: 1, view, legal, cards: cardData });
      ar.n++;
      const better = react > pass + 0.01 ? "react" : pass > react + 0.01 ? "pass" : "tie";
      if (better === "react") ar.reactBetter++; else if (better === "pass") ar.passBetter++; else ar.tie++;
      const botReact = choice.kind.startsWith("play");
      if (botReact) ar.botReacted++;
      if ((better === "react") === botReact) ar.agree++;
      if ((better === "react") !== botReact && ar.n - ar.agree <= 6) {
        lastState = state;
        const link = projectStateFor(state, 1).chain.at(-1);
        console.log(`AR t${state.turn} link ${link?.attackValue}/${link?.defenseValue} opp life ${state.players[0]!.life} hand ${state.players[1]!.hand.length} | pass ${pass.toFixed(2)} | ` +
          legal.filter((i) => i.kind.startsWith("play")).map((i) => `${i.kind} ${label(i)} = ${score(i).toFixed(2)}`).join("; ") + ` | bot: ${choice.kind} ${label(choice)}`);
      }
    }
    const intent = seat === 0 ? chooseWoundingIntent({ seat, view, legal, cards: cardData }) : bot.chooseIntent({ seat, view, legal, cards: cardData });
    let applied = applyIntent(state, seat, intent);
    if (!applied.ok) applied = applyIntent(state, seat, legal.find((i) => applyIntent(state, seat, i).ok)!);
    state = (applied as { state: GameState }).state;
  }
}
console.log(JSON.stringify({ deck, budget, BLOCKS, ar, ...tally, meanGap: +(tally.gap / Math.max(1, tally.both)).toFixed(2) }));
