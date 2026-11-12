/**
 * A deliberately simple teaching bot (owner, 2026-10-11): Ira, Scarlet
 * Revenger with two Harmonized Kodachi and a deck of nothing but red Wounding
 * Blow (cost 0, 4 power, 3 defence, pitch 1). It plays the basics a policy
 * should learn first:
 *
 * - Defend: block with a hand card only while it stops at least 2 damage
 *   (9+ power -> three blockers), keeping one card back to attack with on the
 *   next turn -- unless the arsenal already holds one, then the whole hand may
 *   block. Facing lethal, block with everything.
 * - Attack: pitch a Wounding Blow into each Kodachi (1 power each; a cost-0
 *   card in the pitch zone gives them go again), then swing a Wounding Blow
 *   for 4 -- from the arsenal when there is one -- and arsenal a leftover.
 */
import type { CardView, Decklist, GameIntent } from "@fyendal/shared";
import type { BotPolicyInput } from "./policy.js";

export const WOUNDING_BOT_DECK = {
  hero: "ira, scarlet revenger|0",
  weapons: ["harmonized kodachi|0", "harmonized kodachi|0"],
  card: "wounding blow|1",
  size: 60,
} as const;

/** Each blocker must stop at least this much damage. */
const MIN_STOPPED_PER_CARD = 2;

type Stage = Extract<GameIntent, { kind: "stage-defenders" }>;

function defend(input: BotPolicyInput): GameIntent | undefined {
  const decision = input.view.pendingDecision;
  if (decision?.kind !== "defend") return undefined;
  const me = input.view.players[input.seat];
  const link = input.view.chain[input.view.chain.length - 1];
  const staged = decision.stagedCards ?? [];
  const stagedDefense = staged.reduce((sum, card) => sum + defenseOf(card, input), 0);
  const left = Math.max(0, (link?.attackValue ?? 0) - (link?.defenseValue ?? 0) - stagedDefense);
  const handIds = new Set(me.hand.map((card) => card.instanceId));
  const stagedFromHand = staged.filter((card) => handIds.has(card.instanceId)).length;
  // One card stays back to attack with, unless the arsenal holds it.
  const reserve = me.arsenal.length > 0 ? 0 : 1;
  const lethal = left >= me.life;
  const commit = input.legal.find((intent) => intent.kind === "defend");
  const next = input.legal.find((intent): intent is Stage =>
    intent.kind === "stage-defenders" && handIds.has(intent.instanceIds[intent.instanceIds.length - 1]!));
  if (next && left > 0) {
    const card = me.hand.find((c) => c.instanceId === next.instanceIds[next.instanceIds.length - 1]);
    const stops = Math.min(card ? defenseOf(card, input) : 0, left);
    const roomLeft = me.hand.length - stagedFromHand > reserve;
    if (lethal || (stops >= MIN_STOPPED_PER_CARD && roomLeft)) return next;
  }
  return commit;
}

function defenseOf(card: CardView, input: BotPolicyInput): number {
  return card.defense ?? input.cards[card.cardId]?.defense ?? 0;
}

function attack(input: BotPolicyInput): GameIntent | undefined {
  const me = input.view.players[input.seat];
  const handIds = new Set(me.hand.map((card) => card.instanceId));
  const weaponIds = new Set(me.weapons.map((card) => card.instanceId));
  const fromArsenal = input.legal.find((intent) => intent.kind === "play-from-arsenal");
  const fromHand = input.legal.find((intent) => intent.kind === "play-card" && handIds.has(intent.instanceId)
    && intent.pitchInstanceIds.length === 0);
  if (!fromArsenal && !fromHand) return undefined;
  // A Kodachi swing pitches one hand card; keep one card to finish with.
  const spare = me.hand.length - (fromArsenal ? 0 : 1);
  if (spare >= 1) {
    const kodachi = input.legal.find((intent) => intent.kind === "activate-ability"
      && weaponIds.has(intent.sourceInstanceId)
      && intent.pitchInstanceIds.length === 1 && handIds.has(intent.pitchInstanceIds[0]!));
    if (kodachi) return kodachi;
  }
  return fromArsenal ?? fromHand;
}

function arsenal(input: BotPolicyInput): GameIntent | undefined {
  const decision = input.view.pendingDecision;
  if (decision?.kind !== "arsenal") return undefined;
  const handIds = new Set(input.view.players[input.seat].hand.map((card) => String(card.instanceId)));
  return input.legal.find((intent) => intent.kind === "choose" && handIds.has(intent.optionId));
}

export function chooseWoundingIntent(input: BotPolicyInput): GameIntent {
  const decision = input.view.pendingDecision;
  const chosen = decision?.kind === "defend" ? defend(input)
    : decision?.kind === "arsenal" ? arsenal(input)
    : decision === null && input.view.activePlayer === input.seat && input.view.stack.length === 0
      ? attack(input)
      : undefined;
  if (chosen) return chosen;
  // Every other window (reactions, priority, the opponent's choices): pass if
  // allowed, else the first option.
  return input.legal.find((intent) => intent.kind === "pass")
    ?? input.legal.find((intent) => intent.kind === "choose" && /^(pass|no|decline)$/i.test(intent.optionId))
    ?? input.legal[0]!;
}

/** The bot's deck as a Decklist, given a key -> card id resolver. */
export function woundingBotDecklist(resolve: (key: string) => string): Decklist {
  return {
    heroId: resolve(WOUNDING_BOT_DECK.hero),
    weaponIds: WOUNDING_BOT_DECK.weapons.map(resolve),
    equipment: {},
    deck: Array.from({ length: WOUNDING_BOT_DECK.size }, () => resolve(WOUNDING_BOT_DECK.card)),
  };
}
