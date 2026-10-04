import { describe, expect, it } from "vitest";
import { recordBotMatch, toJsonl, type RecordedDecision } from "./recording.js";

function deterministicClock() {
  let tick = 0;
  return () => tick++;
}

describe("bot match recorder", () => {
  it("records legal intents and the chosen intent at every decision", () => {
    const seen: RecordedDecision[] = [];
    const recording = recordBotMatch({
      left: "bravo",
      right: "briar",
      seed: 44_001,
      maxSteps: 40,
      now: deterministicClock(),
      onDecision: (row) => seen.push(row),
    });

    expect(recording.rows.length).toBeGreaterThan(0);
    expect(recording.rows.length).toBe(recording.steps);
    expect(seen).toEqual(recording.rows);

    for (const row of recording.rows) {
      expect(row.episodeId).toBe("bravo-vs-briar-seed-44001");
      expect(row.legal.length).toBeGreaterThan(0);
      // No concede intents are offered to bots.
      expect(row.legal.every((intent) => intent.kind !== "concede")).toBe(true);
      // The chosen intent resolves to a member of the legal array.
      expect(row.chosenIndex).toBeGreaterThanOrEqual(0);
      expect(row.chosenIndex).toBeLessThan(row.legal.length);
      expect(row.legal[row.chosenIndex]).toEqual(row.chosen);
      // Default observation mode records exactly what the policy saw.
      expect(row.observation).not.toBeNull();
      expect(row.fullObservation).toBeNull();
      expect(row.decisionIndex).toBeGreaterThanOrEqual(0);
      expect(row.seat === 0 || row.seat === 1).toBe(true);
    }

    // Decision indexes are dense and ordered.
    recording.rows.forEach((row, index) => expect(row.decisionIndex).toBe(index));
  });

  it("replays deterministically for the same seed", () => {
    const first = recordBotMatch({
      left: "bravo",
      right: "briar",
      seed: 44_001,
      maxSteps: 40,
      now: deterministicClock(),
    });
    const second = recordBotMatch({
      left: "bravo",
      right: "briar",
      seed: 44_001,
      maxSteps: 40,
      now: deterministicClock(),
    });
    // decisionMs is excluded: wall-clock readings differ between runs.
    const stripTiming = (rows: RecordedDecision[]) =>
      rows.map(({ decisionMs: _decisionMs, ...rest }) => rest);
    expect(stripTiming(second.rows)).toEqual(stripTiming(first.rows));
    expect(second.winner).toBe(first.winner);
  });

  it("supports full and both observation modes", () => {
    const full = recordBotMatch({
      left: "bravo",
      right: "briar",
      seed: 7,
      maxSteps: 5,
      observation: "full",
      now: deterministicClock(),
    });
    expect(full.rows.length).toBeGreaterThan(0);
    for (const row of full.rows) {
      expect(row.observation).toBeNull();
      expect(row.fullObservation).not.toBeNull();
    }

    const both = recordBotMatch({
      left: "bravo",
      right: "briar",
      seed: 7,
      maxSteps: 5,
      observation: "both",
      now: deterministicClock(),
    });
    for (const row of both.rows) {
      expect(row.observation).not.toBeNull();
      expect(row.fullObservation).not.toBeNull();
    }
  });

  it("serializes rows as JSONL with one decision per line", () => {
    const recording = recordBotMatch({
      left: "bravo",
      right: "briar",
      seed: 44_001,
      maxSteps: 10,
      now: deterministicClock(),
    });
    const jsonl = toJsonl(recording.rows);
    const lines = jsonl.trim().split("\n");
    expect(lines.length).toBe(recording.rows.length);
    const parsed = lines.map((line) => JSON.parse(line) as RecordedDecision);
    expect(parsed[0]).toEqual(recording.rows[0]);
    expect(toJsonl([])).toBe("");
  });
});
