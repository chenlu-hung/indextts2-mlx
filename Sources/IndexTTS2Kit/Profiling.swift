import Foundation
import MLX

/// Lightweight wall-clock accumulator for pipeline-stage profiling.
///
/// Stages are timed by wrapping work in `measure`; because MLX is lazy, the
/// closure must force materialization (`eval`) of its result before returning so
/// the elapsed time is attributed to the right stage rather than collapsing into
/// the next `eval` boundary. Totals accumulate across `generate` calls (e.g. all
/// segments of an SRT batch) so a single `report` summarizes the whole run.
public final class StageTimer: @unchecked Sendable {
    public static let shared = StageTimer()

    private var totals: [String: Double] = [:]
    private var counts: [String: Int] = [:]
    private var order: [String] = []
    private let lock = NSLock()

    /// Extra scalar tallies (e.g. mel-token count) reported alongside timings.
    private var tallies: [String: Double] = [:]
    private var tallyOrder: [String] = []

    public init() {}

    public func reset() {
        lock.lock(); defer { lock.unlock() }
        totals.removeAll(); counts.removeAll(); order.removeAll()
        tallies.removeAll(); tallyOrder.removeAll()
    }

    @discardableResult
    public func measure<T>(_ name: String, _ body: () -> T) -> T {
        let t0 = Date()
        let r = body()
        let dt = Date().timeIntervalSince(t0)
        lock.lock()
        if totals[name] == nil { order.append(name) }
        totals[name, default: 0] += dt
        counts[name, default: 0] += 1
        lock.unlock()
        return r
    }

    public func tally(_ name: String, _ v: Double) {
        lock.lock(); defer { lock.unlock() }
        if tallies[name] == nil { tallyOrder.append(name) }
        tallies[name, default: 0] += v
    }

    public func report(_ log: (String) -> Void) {
        lock.lock(); defer { lock.unlock() }
        let total = totals.values.reduce(0, +)
        log("──────────── stage timing ────────────")
        for k in order {
            let s = totals[k] ?? 0
            let n = counts[k] ?? 0
            let pct = total > 0 ? s / total * 100 : 0
            log(String(format: "  %-18@ %8.2fs  %5.1f%%  (n=%d)", k as NSString, s, pct, n))
        }
        log(String(format: "  %-18@ %8.2fs", "TOTAL" as NSString, total))
        for k in tallyOrder {
            let v = tallies[k] ?? 0
            log(String(format: "  · %-16@ %8.0f", k as NSString, v))
        }
        if let melT = tallies["mel_tokens"], let arS = totals["GPT AR"], arS > 0 {
            log(String(format: "  · %-16@ %8.1f tok/s", "AR throughput" as NSString, melT / arS))
        }
        log("───────────────────────────────────────")
    }
}
