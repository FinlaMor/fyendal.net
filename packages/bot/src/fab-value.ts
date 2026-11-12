/**
 * FaB value for RL training: how much each card a player commits is worth.
 * A port of FAB_Sim's rl_agents/fab_value.py ValueLedger (the project
 * owner's ledger), computed by the RL worker from the full game state at
 * every decision point.
 *
 * A card is worth RATE (3).
 *
 * Charged RATE per card COMMITTED: a card played (from anywhere), a card
 *   pitched, a card paid as a cost, a card declared as a defender --
 *   equipment included.
 * Charged RATE per card left IDLE: a card in the turn player's hand when
 *   their turn ends that is still there after it (not arsenaled, not
 *   discarded); cards drawn by the draw-up are new and not counted. Charged
 *   again each turn it idles.
 * Credited as VALUE produced:
 *   attack power THREATENED when a chain link resolves, however it was
 *     defended -- pitching one card to send a 7 is 2 cards for 7;
 *   damage PREVENTED by blocks, min(total defence, power), and by any other
 *     effect, only what it actually prevented;
 *   life lost outside combat: credited to the turn player when the other
 *     seat lost it;
 *   +RATE per card drawn outside the end-of-turn draw-up; +1 per life gained;
 *   +1 per action point gained (go again included; not the action point
 *     every action phase begins with);
 *   +1 per resource gained other than by pitching -- but resources still
 *     floating at the end of the turn take back that much of the credit.
 *
 * Damage TAKEN is not charged by default (owner, 2026-10-05): this value
 * measures how well each card is used; what damage costs is the life stat
 * the model sees and the win/loss reward. A defender scores what its blocks
 * and prevention stopped, less RATE per card: a 3-defence block on a 1-power
 * attack is 1 - 3 = -2.
 *
 * With chargeDamageTaken (owner reversed the above 2026-10-13, after the
 * ledger was measured ranking a non-blocking loser above a blocking winner):
 * a player is also charged every point of life it loses -- combat damage and
 * life lost outside combat. A card that stops s then scores s - RATE against
 * -s for letting it through, so blocking pays from s >= 2.
 *
 * With damageOnce as well (owner asked to try it, 2026-10-20, after the loss
 * analysis found the policy blocking 72-81% of 1-3 power attacks at high
 * life): chargeDamageTaken counts a stopped point twice, as credit for the
 * block AND as life not charged. damageOnce drops the block credit, so the
 * defender is scored only by the life it loses; a card that stops s then
 * scores -(power - s) - RATE against -power, and pays from s > RATE.
 *
 * With blockCredit k instead (owner chose it 2026-10-21, after damageOnce
 * stopped the policy blocking big attacks: every block at best tied): k of
 * what a block stopped is credited on top of the life it saved, so a card
 * stopping s scores (1 + k)s - RATE against 0 and pays from s > RATE / (1 + k);
 * k = 0.5 puts the break-even at 2, the owner's teaching rule (block if each
 * card stops at least 2). k = 1 is the default; damageOnce means k = 0.
 *
 * With creditDamageDealt (owner, 2026-10-14, after the attacker's credit was
 * measured blind to blocking -- a fully blocked 4 scored the same as a 4
 * that hit): the attacker is also credited every point of COMBAT damage its
 * attack deals, on top of the power it threatened. (Life the opponent loses
 * outside combat is already the turn player's.)
 *
 * ONE-SIDED: every amount counts only for the player it happens to -- an
 * opponent gaining a resource is +1 for them and nothing for you. (FAB_Sim's
 * ledger mirrored every credit onto the opponent and charged the defender
 * the damage it took; the owner changed both 2026-10-05.) Every amount goes
 * to the player's most recent decision when it happens (before their first
 * decision: to their first decision).
 *
 * Detection is by state difference between consecutive decision points:
 * draws from the engine's per-turn draw counter (flags.cardsDrawnThisTurn,
 * which the draw-up also increments just before the per-turn flags reset);
 * resource gains are a rise in resources beyond what newly pitched cards
 * produced; action points gained are a rise within a turn, less the one
 * action point the action phase begins with. Draws and gains in the interval
 * that contains a turn change are not counted (it holds the draw-up and the
 * start-of-turn reset).
 */
import type { GameIntent, GameView } from "@fyendal/shared";

export const RATE = 3;

interface Row {
  seat: 0 | 1;
  chosen: GameIntent;
}

interface Inst {
  instanceId: number;
  cardId: string;
}

interface PlayerLike {
  hand: Inst[];
  pitch: Inst[];
  resources: number;
  actionPoints: number;
  flags: Record<string, unknown>;
}

interface StateLike {
  phase: string;
  players: PlayerLike[];
}

/** Cards an intent commits: the card played, its pitch, cost cards, and
 * declared defenders. */
export function committedCards(intent: GameIntent): number {
  switch (intent.kind) {
    case "play-card":
    case "play-from-arsenal":
    case "play-from-zone":
      return 1 + intent.pitchInstanceIds.length + (intent.alternativeCostCardInstanceIds?.length ?? 0);
    case "activate-ability":
      return intent.pitchInstanceIds.length + (intent.alternativeCostCardInstanceIds?.length ?? 0);
    case "defend":
      return intent.instanceIds.length + (intent.pitchInstanceIds?.length ?? 0);
    default:
      return 0;
  }
}

interface Snapshot {
  hand: Set<number>;
  pitch: Set<number>;
  resources: number;
  actionPoints: number;
  drawn: number;
}

function snapshot(p: PlayerLike): Snapshot {
  return {
    hand: new Set(p.hand.map((c) => c.instanceId)),
    pitch: new Set(p.pitch.map((c) => c.instanceId)),
    resources: p.resources,
    actionPoints: p.actionPoints,
    drawn: Number(p.flags.cardsDrawnThisTurn) || 0,
  };
}

export class ValueBook {
  private readonly values: number[] = [];
  private readonly pending = [0, 0];
  private readonly credited = new Set<string>();
  private last: [Snapshot, Snapshot] | null = null;
  private lastTurn = -1;
  private lastActive: 0 | 1 | null = null;
  private lastLife: [number, number] | null = null;
  private lastPhase = "";
  private readonly gainedResources = [0, 0];
  private rows: readonly Row[] = [];

  /** pitchOf(cardId): the resources a card pitches for. */
  constructor(private readonly pitchOf: (cardId: string) => number,
              private readonly options: { chargeDamageTaken?: boolean; creditDamageDealt?: boolean;
                                          damageOnce?: boolean; blockCredit?: number } = {}) {}

  private lastRow(seat: number): number {
    for (let i = this.rows.length - 1; i >= 0; i--) if (this.rows[i]!.seat === seat) return i;
    return -1;
  }

  private credit(seat: number, amount: number): void {
    if (!amount) return;
    const row = this.lastRow(seat);
    if (row < 0) {
      this.pending[seat]! += amount;
      return;
    }
    while (this.values.length <= row) this.values.push(0);
    this.values[row]! += amount;
  }

  /** The ledger's _add: value produced net of RATE per card, for `seat` only. */
  private net(seat: number, amount: number, cards = 0): void {
    this.credit(seat, amount - RATE * cards);
  }

  /** Charge the decision just made (the last of `rows`) for its cards. */
  commit(rows: readonly Row[], intent: GameIntent): void {
    this.rows = rows;
    const seat = rows[rows.length - 1]?.seat;
    if (seat === undefined) return;
    this.net(seat, 0, committedCards(intent));
  }

  /** Called at every decision point and once at game end, with the rows
   * decided so far, a public view, and the internal state. */
  observe(rows: readonly Row[], view: GameView, state: StateLike): void {
    this.rows = rows;
    const now: [Snapshot, Snapshot] = [snapshot(state.players[0]!), snapshot(state.players[1]!)];
    const turnChanged = view.turn !== this.lastTurn;

    if (turnChanged && this.lastActive !== null && this.last) {
      const prev = this.lastActive;
      // Idle: what the turn player held at their last decision point and
      // still holds now. Drawn cards are new instances; arsenaled, played or
      // discarded cards are gone.
      let idle = 0;
      for (const id of now[prev].hand) if (this.last[prev].hand.has(id)) idle++;
      this.net(prev, 0, idle);
      // Gained resources left floating were not used: take their credit back.
      const unused = Math.min(this.last[prev].resources, this.gainedResources[prev]!);
      if (unused > 0) this.net(prev, -unused);
      this.gainedResources[prev] = 0;
    }

    if (!turnChanged && this.last) {
      for (const seat of [0, 1] as const) {
        const before = this.last[seat];
        const after = now[seat];
        const drawn = after.drawn - before.drawn;
        if (drawn > 0) this.net(seat, RATE * drawn);
        let pitched = 0;
        for (const card of state.players[seat]!.pitch) {
          if (!before.pitch.has(card.instanceId)) pitched += this.pitchOf(card.cardId);
        }
        const gained = after.resources - before.resources - pitched;
        if (gained > 0) {
          this.gainedResources[seat]! += gained;
          this.net(seat, gained);
        }
        // The action point the action phase begins with is not a gain.
        const turnAp = seat === view.activePlayer && this.lastPhase === "start" && state.phase !== "start" ? 1 : 0;
        const ap = after.actionPoints - before.actionPoints - turnAp;
        if (ap > 0) this.net(seat, ap);
      }
    }

    // Combat: each resolved link once.
    const combatDamage = [0, 0];
    view.chain.forEach((link, index) => {
      if (!link.resolved) return;
      const key = `${view.turn}:${index}:${link.attackingCard.instanceId}`;
      if (this.credited.has(key)) return;
      this.credited.add(key);
      const attacker = view.activePlayer;
      const defender = 1 - attacker;
      const power = Math.max(0, link.attackValue ?? 0);
      const blocked = Math.min(Math.max(0, link.defenseValue ?? 0), power);
      const damage = Math.max(0, link.damage ?? 0);
      const prevented = Math.max(0, power - blocked - damage);
      this.net(attacker, power);
      // creditDamageDealt: what landed counts again for the attacker.
      if (this.options.creditDamageDealt) this.net(attacker, damage);
      // With chargeDamageTaken, blockCredit is the share of what a block
      // stopped that is credited on top of the life it saved (damageOnce: 0).
      const share = !this.options.chargeDamageTaken ? 1
        : this.options.damageOnce ? 0 : (this.options.blockCredit ?? 1);
      this.net(defender, share * (blocked + prevented));
      combatDamage[defender]! += damage;
    });

    // Life outside combat.
    const life: [number, number] = [view.players[0].life, view.players[1].life];
    if (this.lastLife) {
      for (const seat of [0, 1] as const) {
        const delta = life[seat] - this.lastLife[seat];
        if (delta > 0) this.net(seat, delta);
        const lost = -delta - combatDamage[seat]!;
        if (delta < 0 && lost > 0 && view.activePlayer !== seat) this.net(view.activePlayer, lost);
        // Every point of life lost, combat damage included (the delta covers both).
        if (delta < 0 && this.options.chargeDamageTaken) this.net(seat, delta);
      }
    }

    this.lastLife = life;
    this.lastPhase = state.phase;
    this.last = now;
    if (turnChanged) {
      this.lastTurn = view.turn;
      this.lastActive = view.activePlayer as 0 | 1;
    }
  }

  /** `seat`'s value so far (its decisions plus pending credit). */
  total(seat: number): number {
    let sum = this.pending[seat]!;
    for (let i = 0; i < this.rows.length; i++) if (this.rows[i]!.seat === seat) sum += this.values[i] ?? 0;
    return sum;
  }

  /** Per-decision values for `rows` (pending credit on each seat's first). */
  result(rows: readonly Row[]): number[] {
    const out = Array.from({ length: rows.length }, (_, i) => this.values[i] ?? 0);
    for (const seat of [0, 1]) {
      const first = rows.findIndex((row) => row.seat === seat);
      if (first >= 0) out[first]! += this.pending[seat]!;
    }
    return out;
  }
}
