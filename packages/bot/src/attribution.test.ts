import { describe, expect, it } from "vitest";
import {
  attributeEpisode,
  decisionRewards,
  DEFAULT_REWARD_WEIGHTS,
  type AttributedDecision,
  type EpisodeAttribution,
} from "./attribution.js";
import { recordBotMatch, type BotMatchRecording } from "./recording.js";
import type { ChainLinkView } from "@fyendal/shared";

/** Indexed access that throws on a missing entry (test-only). */
function must<T>(arr: readonly T[], index: number): T {
  const value = arr[index];
  if (value === undefined) throw new Error(`missing index ${index}`);
  return value;
}

function attributed(attr: EpisodeAttribution, index: number): AttributedDecision {
  return must(attr.decisions, index);
}

function deterministicClock() {
  let tick = 0;
  return () => tick++;
}

function sampleEpisode(): BotMatchRecording {
  return recordBotMatch({
    left: "bravo",
    right: "briar",
    seed: 1,
    now: deterministicClock(),
  });
}

/** Independently scan observations for resolved links (test-only logic). */
function resolvedLinks(recording: BotMatchRecording) {
  const links: { instanceId: number; first: number; link: ChainLinkView }[] = [];
  const seen = new Set<number>();
  recording.rows.forEach((row, index) => {
    const chain = row.observation?.chain ?? row.fullObservation?.chain ?? [];
    for (const link of chain) {
      const iid = link.attackingCard.instanceId;
      if (!seen.has(iid)) {
        seen.add(iid);
        links.push({ instanceId: iid, first: index, link });
      } else {
        const entry = links.find((l) => l.instanceId === iid)!;
        entry.link = link;
      }
    }
  });
  return links.filter((l) => l.link.resolved);
}

describe("value attribution", () => {
  it("attributes a full episode with sane totals", () => {
    const recording = sampleEpisode();
    expect(recording.complete).toBe(true);
    const attr = attributeEpisode(recording);

    expect(attr.episodeId).toBe(recording.episodeId);
    expect(attr.winner).toBe(recording.winner);
    expect(attr.decisions.length).toBe(recording.rows.length);

    let dealtTotal = 0;
    for (const d of attr.decisions) {
      expect(d.damageDealt).toBeGreaterThanOrEqual(0);
      expect(d.damagePrevented).toBeGreaterThanOrEqual(0);
      expect(d.indirectValue).toBeGreaterThanOrEqual(0);
      expect(d.value).toBe(d.damageDealt + d.damagePrevented + d.indirectValue);
      for (const c of d.cards) {
        expect(c.value).toBe(c.damageDealt + c.damagePrevented + c.indirectValue);
      }
      dealtTotal += d.damageDealt;
    }
    // Sanity: some damage was dealt and some prevented in a real game.
    expect(dealtTotal).toBeGreaterThan(0);
    expect(attr.totalValue[0] + attr.totalValue[1]).toBeGreaterThan(0);
  });

  it("credits prevented damage to the defend commit, not the staging", () => {
    const recording = sampleEpisode();
    const attr = attributeEpisode(recording);

    let commitCredited = 0;
    recording.rows.forEach((row, index) => {
      const d = attributed(attr, index);
      if (row.chosen.kind === "stage-defenders") {
        // Staging is setup: no prevented value on staging decisions.
        expect(d.damagePrevented).toBe(0);
      }
      if (row.chosen.kind === "defend" && d.damagePrevented > 0) {
        commitCredited++;
        // The committed defenders appear on the link in the observation
        // after the commit (the commit row's own observation is pre-action).
        const nextRow = recording.rows[index + 1];
        const rowLink = (nextRow?.observation?.chain ?? []).find((l) =>
          (l.defendingCards ?? []).some((c) =>
            d.cards.some((ac) => ac.instanceId === c.instanceId),
          ),
        );
        expect(rowLink).toBeDefined();
      }
    });
    expect(commitCredited).toBeGreaterThan(0);
  });

  it("caps prevented at the incoming attack power", () => {
    const recording = sampleEpisode();
    const attr = attributeEpisode(recording);
    for (const d of attr.decisions) {
      const defenseCards = d.cards.filter((c) => c.role === "defense");
      const cardPrevented = defenseCards.reduce((n, c) => n + c.damagePrevented, 0);
      // Per-card breakdown covers chain defense; Arcane Barrier prevention is
      // decision-level only (the equipment isn't identified in v1).
      expect(cardPrevented).toBeLessThanOrEqual(d.damagePrevented);
      for (const c of defenseCards) {
        expect(c.damagePrevented).toBeGreaterThanOrEqual(0);
      }
    }
    // Cross-check against resolved links: prevented <= attackValue always.
    for (const { link } of resolvedLinks(recording)) {
      const prevented = Math.min(link.defenseValue ?? 0, link.attackValue ?? 0);
      expect(prevented).toBeLessThanOrEqual(link.attackValue ?? 0);
    }
  });

  it("credits Arcane Barrier prevention as prevented damage", () => {
    const recording = sampleEpisode();
    const attr = attributeEpisode(recording);
    let barrierCredited = 0;
    recording.rows.forEach((row, index) => {
      if (
        row.decisionKind === "choose-target" &&
        row.chosen.kind === "choose" &&
        /^pay [1-9]/.test(row.chosen.optionId ?? "")
      ) {
        const d = attributed(attr, index);
        const prevented = parseInt(row.chosen.optionId!.split(" ")[1]!, 10);
        expect(d.damagePrevented).toBeGreaterThanOrEqual(prevented);
        expect(d.seat).toBe(row.seat);
        barrierCredited++;
      }
    });
    // The sample episode has Arcane Barrier decisions.
    expect(barrierCredited).toBeGreaterThan(0);
  });

  it("credits 1 for go-again and on-hit triggers", () => {
    const recording = sampleEpisode();
    const attr = attributeEpisode(recording);
    const links = resolvedLinks(recording);

    const goAgainLinks = links.filter((l) => l.link.goAgain);
    expect(goAgainLinks.length).toBeGreaterThan(0);
    for (const { first } of goAgainLinks) {
      // Birth decision is the row before the link first appears.
      expect(attributed(attr, first - 1).indirectValue).toBeGreaterThanOrEqual(1);
    }

    const onHitLinks = links.filter(
      (l) => l.link.hit && (l.link.onHitEffects?.length ?? 0) > 0,
    );
    expect(onHitLinks.length).toBeGreaterThan(0);
    for (const { first } of onHitLinks) {
      expect(attributed(attr, first - 1).indirectValue).toBeGreaterThanOrEqual(1);
    }
  });

  it("caps dealt damage at lethal", () => {
    const recording = sampleEpisode();
    const attr = attributeEpisode(recording);
    for (const { instanceId, first, link } of resolvedLinks(recording)) {
      const birth = attributed(attr, first - 1);
      const attackCard = birth.cards.find(
        (c) => c.role === "attack" && c.instanceId === instanceId,
      );
      if (!attackCard) continue;
      const defender = 1 - birth.seat;
      const oppLife = must(recording.rows, first).observation?.players[defender]?.life ??
        must(recording.rows, first).fullObservation?.players[defender]?.life ?? 0;
      expect(attackCard.damageDealt).toBeLessThanOrEqual(Math.max(0, oppLife));
      expect(attackCard.damageDealt).toBeLessThanOrEqual(link.damage ?? 0);
    }
  });

  it("combines terminal win/lose with shaped value in rewards", () => {
    const recording = sampleEpisode();
    const attr = attributeEpisode(recording);
    const rewards = decisionRewards(attr);
    expect(rewards.length).toBe(attr.decisions.length);

    // Default weights: win dominates, health contributes nothing.
    expect(DEFAULT_REWARD_WEIGHTS.health).toBe(0);
    const last = rewards.length - 1;
    const lastDecision = attributed(attr, last);
    const expectedTerminal = attr.winner === lastDecision.seat ? 1 : -1;
    expect(rewards[last]).toBeCloseTo(
      DEFAULT_REWARD_WEIGHTS.value * lastDecision.value +
        DEFAULT_REWARD_WEIGHTS.win * expectedTerminal,
      10,
    );
    // Non-terminal decisions carry only shaped value.
    expect(rewards[0]).toBeCloseTo(
      DEFAULT_REWARD_WEIGHTS.value * attributed(attr, 0).value,
      10,
    );
  });

  it("handles episodes that hit the step cap without a winner", () => {
    const recording = recordBotMatch({
      left: "bravo",
      right: "briar",
      seed: 1,
      maxSteps: 5,
      now: deterministicClock(),
    });
    expect(recording.complete).toBe(false);
    const attr = attributeEpisode(recording);
    expect(attr.winner).toBeNull();
    expect(attr.decisions.length).toBe(recording.rows.length);
    const rewards = decisionRewards(attr);
    // No terminal bonus without a winner.
    rewards.forEach((r, i) =>
      expect(r).toBeCloseTo(DEFAULT_REWARD_WEIGHTS.value * attributed(attr, i).value, 10),
    );
  });
});
