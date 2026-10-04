import type { ChainLinkView, GameView } from "@fyendal/shared";
import type { BotMatchRecording } from "./recording.js";

/** Heuristic credit for indirect value: on-hit triggers and go-again grants
 * each count as 1 value for now (Joe's rule). Computed from the resolved
 * chain link, so only realized triggers/grants are credited. */
const INDIRECT_VALUE_CREDIT = 1;

export type AttributedCardRole = "attack" | "defense";

/** Value attributed to a single card. */
export interface CardValueAttribution {
  instanceId: number;
  cardId: string;
  role: AttributedCardRole;
  /** Effective damage dealt, capped at lethal. */
  damageDealt: number;
  /** Effective damage prevented, capped at incoming attack power. */
  damagePrevented: number;
  /** Heuristic credit for on-hit triggers / go-again grants. */
  indirectValue: number;
  value: number;
}

/** Value attributed to one recorded decision. The training reward for a
 * decision is derived from these components (see decisionRewards). */
export interface AttributedDecision {
  decisionIndex: number;
  seat: 0 | 1;
  /** Effective damage dealt to the opponent, capped at lethal. */
  damageDealt: number;
  /** Effective damage prevented, capped at the incoming attack's power. */
  damagePrevented: number;
  /** Heuristic credit for on-hit triggers / go-again grants. */
  indirectValue: number;
  /** Total value: dealt + prevented + indirect. */
  value: number;
  cards: CardValueAttribution[];
}

export interface EpisodeAttribution {
  episodeId: string;
  winner: 0 | 1 | null;
  decisions: AttributedDecision[];
  /** Sum of per-decision values per seat, for sanity checks. */
  totalValue: [number, number];
}

interface LinkRun {
  instanceId: number;
  owner: 0 | 1;
  /** First row index whose observation shows the link. */
  first: number;
  /** Last row index whose observation shows the link. */
  last: number;
  /** Link state at its last appearance (carries resolution outcome). */
  final: ChainLinkView;
  /** Observation at first appearance (pre-damage life totals for lethal cap). */
  firstObservation: GameView;
}

function viewOf(row: BotMatchRecording["rows"][number]): GameView | null {
  return row.observation ?? row.fullObservation;
}

/** Group chain-link appearances into contiguous lifecycles keyed by the
 * attacking card's instance id. A weapon attacking on two different turns
 * produces two runs. */
function linkRuns(rows: BotMatchRecording["rows"]): LinkRun[] {
  const runs: LinkRun[] = [];
  const open = new Map<number, LinkRun>();
  rows.forEach((row, index) => {
    const seen = new Set<number>();
    for (const link of viewOf(row)?.chain ?? []) {
      const instanceId = link.attackingCard.instanceId;
      seen.add(instanceId);
      const existing = open.get(instanceId);
      if (existing) {
        existing.last = index;
        existing.final = link;
      } else {
        const observation = viewOf(row);
        if (!observation) continue;
        open.set(instanceId, {
          instanceId,
          owner: link.attackingCard.owner as 0 | 1,
          first: index,
          last: index,
          final: link,
          firstObservation: observation,
        });
      }
    }
    for (const [instanceId, run] of open) {
      if (!seen.has(instanceId)) {
        runs.push(run);
        open.delete(instanceId);
      }
    }
  });
  for (const run of open.values()) runs.push(run);
  return runs;
}

/** Find the decision that committed defenders for a link: the row before
 * defending cards first appear on it. Returns -1 when nothing was committed. */
function defenseCommitRow(
  rows: BotMatchRecording["rows"],
  run: LinkRun,
): number {
  for (let index = run.first; index <= run.last; index++) {
    const row = rows[index];
    if (!row) continue;
    const link = (viewOf(row)?.chain ?? []).find(
      (candidate) => candidate.attackingCard.instanceId === run.instanceId,
    );
    if (link && (link.defendingCards?.length ?? 0) > 0) return index - 1;
  }
  return -1;
}

/** Intent kinds that play or activate a card (sources of damage). */
const PLAY_INTENT_KINDS = new Set([
  "play-card",
  "play-from-arsenal",
  "play-from-zone",
  "activate-ability",
]);

/** How far back to search for the card play that caused non-physical damage. */
const ARCANE_TRACEBACK_WINDOW = 30;

/** Find the most recent card play by a seat at or before a row (inclusive). */
function recentPlayRow(
  rows: BotMatchRecording["rows"],
  beforeIndex: number,
  seat: 0 | 1,
): number {
  const start = Math.min(beforeIndex, rows.length - 1);
  const end = Math.max(0, start - ARCANE_TRACEBACK_WINDOW);
  for (let index = start; index >= end; index--) {
    const row = rows[index];
    if (row && row.seat === seat && PLAY_INTENT_KINDS.has(row.chosen.kind)) {
      return index;
    }
  }
  return -1;
}

/** An Arcane Barrier prevention decision: "pay P" chosen on a choose-target. */
interface BarrierPrevention {
  rowIndex: number;
  defender: 0 | 1;
  prevented: number;
}

/** Detect Arcane Barrier preventions. The engine uses "pay N" option ids
 * specifically for prevention amounts, so the pattern is a reliable signal. */
function barrierPreventions(rows: BotMatchRecording["rows"]): BarrierPrevention[] {
  const out: BarrierPrevention[] = [];
  rows.forEach((row, rowIndex) => {
    if (row.decisionKind !== "choose-target") return;
    if (row.chosen.kind !== "choose") return;
    const match = /^pay (\d+)$/.exec(row.chosen.optionId ?? "");
    if (!match) return;
    const prevented = parseInt(match[1] ?? "0", 10);
    if (prevented <= 0) return;
    out.push({ rowIndex, defender: row.seat, prevented });
  });
  return out;
}

/** Per-decision value attribution for one recorded episode.
 *
 * Combat value is resolved through chain links, which carry the
 * engine-authoritative outcome:
 * - damage dealt: the link's damage, capped at the opponent's remaining life
 *   (lethal cap), credited to the decision that played the attack;
 * - damage prevented: min(defenseValue, attackValue), credited to the
 *   decision that committed the defenders (staging decisions are setup and
 *   carry no value themselves);
 * - indirect value: +1 when the resolved attack had go-again, +1 when it hit
 *   with on-hit effects (Joe's heuristic).
 *
 * Non-physical (arcane/effect) damage:
 * - prevention: Arcane Barrier "pay P" decisions credit P as prevented;
 * - damage dealt: opponent life loss not explained by combat is credited to
 *   the attacker's most recent card play (heuristic traceback).
 *
 * Only public information is used: everything comes from the recorded
 * (partial) observations. Ward and other out-of-chain prevention are not
 * attributed in this version. */
export function attributeEpisode(recording: BotMatchRecording): EpisodeAttribution {
  const rows = recording.rows;
  const decisions: AttributedDecision[] = rows.map((row) => ({
    decisionIndex: row.decisionIndex,
    seat: row.seat,
    damageDealt: 0,
    damagePrevented: 0,
    indirectValue: 0,
    value: 0,
    cards: [],
  }));

  const runs = linkRuns(rows);
  for (const run of runs) {
    const link = run.final;
    if (!link.resolved) continue;
    const birthRow = run.first - 1;
    if (birthRow < 0 || birthRow >= rows.length) continue;

    const attacker = run.owner;
    const defender = (attacker === 0 ? 1 : 0) as 0 | 1;
    const attackValue = Math.max(0, link.attackValue ?? 0);
    const defenseValue = Math.max(0, link.defenseValue ?? 0);
    // Lethal cap: no more value than the opponent's remaining life.
    const oppLife = Math.max(0, run.firstObservation.players[defender]?.life ?? 0);
    const dealt = Math.min(Math.max(0, link.damage ?? 0), oppLife);
    const prevented = Math.min(defenseValue, attackValue);

    let indirect = 0;
    if (link.goAgain) indirect += INDIRECT_VALUE_CREDIT;
    if (link.hit && (link.onHitEffects?.length ?? 0) > 0) {
      indirect += INDIRECT_VALUE_CREDIT;
    }

    const birth = decisions[birthRow];
    if (!birth) continue;
    birth.damageDealt += dealt;
    birth.indirectValue += indirect;
    birth.cards.push({
      instanceId: run.instanceId,
      cardId: link.attackingCard.cardId,
      role: "attack",
      damageDealt: dealt,
      damagePrevented: 0,
      indirectValue: indirect,
      value: dealt + indirect,
    });

    const commitRow = defenseCommitRow(rows, run);
    const commit = commitRow >= 0 ? decisions[commitRow] : undefined;
    const commitSeatRow = commitRow >= 0 ? rows[commitRow] : undefined;
    if (commit && commitSeatRow && commitSeatRow.seat === defender) {
      commit.damagePrevented += prevented;
      // Split the prevented total across defending cards in order, so no
      // card is credited for more than the remaining incoming damage.
      let remaining = prevented;
      for (const defenderCard of link.defendingCards ?? []) {
        if (remaining <= 0) break;
        const credited = Math.min(Math.max(0, defenderCard.defense ?? 0), remaining);
        remaining -= credited;
        commit.cards.push({
          instanceId: defenderCard.instanceId,
          cardId: defenderCard.cardId,
          role: "defense",
          damageDealt: 0,
          damagePrevented: credited,
          indirectValue: 0,
          value: credited,
        });
      }
    }
  }

  const totalValue: [number, number] = [0, 0];

  // --- Non-physical (arcane/effect) damage ---

  // Arcane Barrier prevention: each "pay P" decision credits P as prevented.
  for (const barrier of barrierPreventions(rows)) {
    const decision = decisions[barrier.rowIndex];
    if (decision) decision.damagePrevented += barrier.prevented;
  }

  // Residual non-physical damage: opponent life loss not explained by combat
  // links is credited to the attacker's most recent card play (heuristic
  // traceback; the packet source isn't in the projected view).
  // Precompute when each link's damage resolved (action row -> damage by owner).
  const combatResolved = new Map<number, [number, number]>();
  for (const run of runs) {
    if (!run.final.resolved) continue;
    let resolvedRow = -1;
    for (let index = run.first; index <= run.last; index++) {
      const row = rows[index];
      const link = row
        ? (viewOf(row)?.chain ?? []).find(
            (candidate) => candidate.attackingCard.instanceId === run.instanceId,
          )
        : undefined;
      if (link?.resolved) {
        resolvedRow = index;
        break;
      }
    }
    if (resolvedRow <= 0) continue;
    const actionRow = resolvedRow - 1;
    const entry = combatResolved.get(actionRow) ?? [0, 0];
    entry[run.owner] += Math.max(0, run.final.damage ?? 0);
    combatResolved.set(actionRow, entry);
  }

  for (let index = 0; index < rows.length - 1; index++) {
    const row = rows[index];
    const next = rows[index + 1];
    if (!row || !next) continue;
    const actor = row.seat;
    const opponent = (actor === 0 ? 1 : 0) as 0 | 1;
    const lifeBefore = viewOf(row)?.players[opponent]?.life;
    const lifeAfter = viewOf(next)?.players[opponent]?.life;
    if (lifeBefore === undefined || lifeAfter === undefined) continue;
    const lost = Math.max(0, lifeBefore - lifeAfter);
    if (lost === 0) continue;
    const combat = combatResolved.get(index)?.[actor] ?? 0;
    const residual = Math.max(0, lost - combat);
    if (residual === 0) continue;
    // Heuristic: credit the attacker's most recent card play. Skipped when
    // no recent play is found (e.g. turn-start blood debt) to avoid
    // misattribution.
    const playRow = recentPlayRow(rows, index, actor);
    const play = playRow >= 0 ? decisions[playRow] : undefined;
    if (play) play.damageDealt += residual;
  }

  for (const decision of decisions) {
    decision.value = decision.damageDealt + decision.damagePrevented + decision.indirectValue;
    totalValue[decision.seat] += decision.value;
  }
  return { episodeId: recording.episodeId, winner: recording.winner, decisions, totalValue };
}

export interface RewardWeights {
  /** Terminal win/lose weight (dominant). */
  win: number;
  /** Per-decision value weight (dense shaping). */
  value: number;
  /** Health weight. Must be 0: health is a feature, not a reward — value
   * already captures life swings, so any nonzero weight double-counts. */
  health: number;
}

/** Default weights reflecting Joe's hierarchy: win/lose > value per card >
 * health (zero). */
export const DEFAULT_REWARD_WEIGHTS: RewardWeights = { win: 1, value: 0.1, health: 0 };

/** Per-decision training rewards. The terminal win/lose outcome is attached
 * to the final decision; every decision carries its shaped value. */
export function decisionRewards(
  attribution: EpisodeAttribution,
  weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
): number[] {
  const last = attribution.decisions.length - 1;
  return attribution.decisions.map((decision, index) => {
    let reward = weights.value * decision.value;
    if (index === last && attribution.winner !== null) {
      reward += weights.win * (attribution.winner === decision.seat ? 1 : -1);
    }
    return reward;
  });
}
