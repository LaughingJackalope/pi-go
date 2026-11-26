//
// `s3_breach.swift` — Native Swift Port of `protocol_breach.py` on `S3Core.swift`
// (Implicit State Machine / "Weird Machine" Hunter over `portal` Distributed Triad
//  + Bullet-Time System 2 Symbolic State-Splitting BFS + In-Swift `[CLS]` Aux Head)
//
// Compile:
//   swiftc -O S3Core.swift s3_breach.swift -o s3-breach -framework Foundation -framework CoreML -framework Accelerate
//

import Accelerate
import CoreML
import Darwin
import Foundation

let BREACH_PROBES = ["NORMAL", "FAULT", "BURST", "AUDIT"]
let BREACH_FATAL_FLAWS = [
    "FLAW-1: AMNESIA_ZOMBIE_TASK",
    "FLAW-2: FREE_TOKEN_DRAIN",
    "FLAW-3: BUDGET_OVERSHOOT_BREACH",
    "FLAW-4: QUOTA_DEADLOCK_STARVATION",
]

struct PortalTriadState: Equatable {
    var jobPhase: String = "NONE"       // NONE | RUNNING | COMPLETE | HUNG
    var botMem: String = "EMPTY"        // EMPTY | RUNNING | WAITING | CLOSED
    var sseStream: String = "IDLE"      // IDLE | OPEN_UNBILLED | CLOSED_BILLED
    var ledgerUsd: Float = 0.0
    var actualVertexUsd: Float = 0.0
    var activePods: Int = 0
    var lastEvent: String = "INIT"

    func whiteboardState() -> String {
        if jobPhase == "NONE" && botMem == "EMPTY" && activePods == 0 {
            return "S0:IDLE"
        }
        if jobPhase == "RUNNING" && botMem == "RUNNING" && ledgerUsd < 1.50 && abs(actualVertexUsd - ledgerUsd) < 0.05 {
            return "S1:RUNNING"
        }
        if jobPhase == "COMPLETE" && botMem == "WAITING" {
            return "S2:WAITING"
        }
        if botMem == "CLOSED" {
            return "S3:CLOSED"
        }
        return "GHOST_STATE"
    }

    func desyncSignature() -> String? {
        if jobPhase == "COMPLETE" && botMem == "EMPTY" {
            return "G1: Job=COMPLETE but BotMem=EMPTY"
        }
        if sseStream == "OPEN_UNBILLED" {
            return "G2: SSE=OPEN_UNBILLED"
        }
        if actualVertexUsd - ledgerUsd >= 0.50 {
            return "G3: VertexUSD > LedgerUSD"
        }
        if jobPhase == "RUNNING" && ledgerUsd >= 1.60 {
            return "G4: Ledger=$1.80 Near Cap"
        }
        if activePods >= 2 && (jobPhase == "HUNG" || botMem == "EMPTY") {
            return "G5: Pods=2/2 Quota Full + Unwatched Hung Pod"
        }
        return nil
    }
}

func stepEnvironment(st: PortalTriadState, probe: String) -> (next: PortalTriadState, desc: String, flaw: String?) {
    var nxt = st
    var flaw: String? = nil
    let desc: String

    switch probe {
    case "NORMAL":
        if nxt.jobPhase == "NONE" || nxt.botMem == "CLOSED" {
            nxt.jobPhase = "RUNNING"
            nxt.botMem = "RUNNING"
            nxt.sseStream = "OPEN_UNBILLED"
            nxt.activePods = min(2, max(1, nxt.activePods + 1))
            nxt.ledgerUsd = 0.60
            nxt.actualVertexUsd = 0.60
            desc = "!task started Job pod (RUNNING, $0.60)"
        } else if nxt.jobPhase == "RUNNING" {
            nxt.jobPhase = "COMPLETE"
            nxt.botMem = "WAITING"
            nxt.sseStream = "CLOSED_BILLED"
            nxt.activePods = max(0, nxt.activePods - 1)
            desc = "Pod emitted task_result; taskbot waiting for approve"
        } else if nxt.jobPhase == "COMPLETE" && nxt.botMem == "WAITING" {
            nxt.botMem = "CLOSED"
            desc = "Human approved in #t-<id>; task closed cleanly"
        } else {
            nxt = PortalTriadState(lastEvent: "RESET_TO_IDLE")
            desc = "Reset workspace to S0:IDLE"
        }

    case "FAULT":
        if nxt.jobPhase == "COMPLETE" && nxt.botMem == "WAITING" {
            nxt.botMem = "EMPTY"
            desc = "FAULT: taskbot pod restarted; resume() skipped Complete Job!"
        } else if nxt.sseStream == "OPEN_UNBILLED" {
            nxt.actualVertexUsd += 0.75
            nxt.sseStream = "IDLE"
            desc = "FAULT: omp dropped SSE stream mid-generation ($0.75 unbilled!)"
        } else if nxt.jobPhase == "RUNNING" {
            nxt.jobPhase = "HUNG"
            nxt.botMem = "EMPTY"
            nxt.activePods = 2
            desc = "FAULT: Pod hung + log stream dropped (active_pods=2/2 locked!)"
        } else {
            nxt.jobPhase = "RUNNING"
            nxt.botMem = "RUNNING"
            nxt.sseStream = "OPEN_UNBILLED"
            nxt.activePods = 1
            desc = "FAULT probe on IDLE -> spawned test pod with open SSE stream"
        }

    case "BURST":
        if nxt.jobPhase == "RUNNING" && nxt.ledgerUsd < 1.60 {
            nxt.ledgerUsd = 1.80
            nxt.actualVertexUsd = max(nxt.actualVertexUsd, 1.80)
            nxt.sseStream = "OPEN_UNBILLED"
            desc = "BURST: Task consumed $1.80 of $2.00 cap; stream open"
        } else if nxt.jobPhase == "RUNNING" && nxt.ledgerUsd >= 1.60 {
            nxt.ledgerUsd = 3.25
            nxt.actualVertexUsd = 3.25
            nxt.sseStream = "CLOSED_BILLED"
            flaw = "FLAW-3: BUDGET_OVERSHOOT_BREACH"
            desc = "BREACH! Heavy call at $1.80 bypassed 402 check -> spent $3.25/$2.00!"
        } else if nxt.actualVertexUsd - nxt.ledgerUsd >= 0.50 {
            nxt.actualVertexUsd += 0.85
            flaw = "FLAW-2: FREE_TOKEN_DRAIN"
            desc = String(format: "BREACH! Unbilled Vertex spend hit $%.2f vs Ledger $%.2f!", nxt.actualVertexUsd, nxt.ledgerUsd)
        } else {
            nxt.activePods = 2
            nxt.jobPhase = "RUNNING"
            nxt.botMem = "RUNNING"
            nxt.ledgerUsd = 1.80
            nxt.actualVertexUsd = 1.80
            desc = "BURST: Saturated 2/2 pod quota and pushed spend to $1.80"
        }

    default: // AUDIT
        if nxt.jobPhase == "COMPLETE" && nxt.botMem == "EMPTY" {
            flaw = "FLAW-1: AMNESIA_ZOMBIE_TASK"
            desc = "BREACH! Human sent 'approve' in #t-<id>, taskbot ignored orphaned Complete Job!"
        } else if nxt.activePods >= 2 && (nxt.jobPhase == "HUNG" || nxt.botMem == "EMPTY") {
            flaw = "FLAW-4: QUOTA_DEADLOCK_STARVATION"
            desc = "BREACH! User sent !task in #rnd -> REFUSED (2/2 quota deadlocked by hung pod)!"
        } else if nxt.actualVertexUsd - nxt.ledgerUsd >= 0.50 {
            flaw = "FLAW-2: FREE_TOKEN_DRAIN"
            desc = String(format: "BREACH! Billing audit caught $%.2f unrecorded Vertex spend!", nxt.actualVertexUsd - nxt.ledgerUsd)
        } else {
            desc = "AUDIT: Checked invariants on happy-path state (all nominal)"
        }
    }

    nxt.lastEvent = desc
    return (nxt, desc, flaw)
}

func system2BulletTimeBFS(st: PortalTriadState, alreadyFound: Set<String>) -> (firstProbe: String, flaw: String, nodesExpanded: Int)? {
    var queue: [(state: PortalTriadState, path: [String])] = [(st, [])]
    var qHead = 0
    var expanded = 0
    while qHead < queue.count {
        let (curSt, path) = queue[qHead]
        qHead += 1
        if path.count >= 4 { continue }
        for p in BREACH_PROBES {
            expanded += 1
            let (nxtSt, _, flaw) = stepEnvironment(st: curSt, probe: p)
            var newPath = path
            newPath.append(p)
            if let f = flaw, !alreadyFound.contains(f) {
                return (newPath[0], f, expanded)
            }
            queue.append((nxtSt, newPath))
        }
    }
    return nil
}

struct BreachDecision {
    let probabilities: [String: Float]
    let proposed: String
    let executed: String
    let intervened: Bool
    let desyncProb: Float
    let breachProb: Float
    let actProb: Float
    let bulletTimeS2: Bool
    let s2NodesExpanded: Int
    let targetFlawHint: String
    let inferenceMs: Double
}

final class S3BreachPolicy {
    let engine: S3ANEEngine
    let auxHead: S3AuxHead
    var guided: Bool
    var threePass: Bool

    let choiceIns = "Choose the best state-machine audit probe."
    let desyncIns = "Is a state desync detected?"
    let breachIns = "Is an invariant breach reachable?"
    let noulOpts = [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]

    init(engine: S3ANEEngine, guided: Bool = true, threePass: Bool = false) throws {
        self.engine = engine
        self.guided = guided
        self.threePass = threePass

        var samples: [(state: String, options: [String])] = []
        let keys = ["normal", "fault", "burst", "audit"]
        for des in ["yes", "no"] {
            for br in ["yes", "no"] {
                let st = "State desync detected: \(des). Invariant breach reachable: \(br)."
                for prefIdx in 0..<4 {
                    var opts: [String] = []
                    for (i, k) in keys.enumerated() {
                        if i == prefIdx {
                            opts.append("\(k): Safe. Best probe to expose hidden state flaw.")
                        } else if i == 0 && des == "yes" {
                            opts.append("\(k): Unsafe. Resets active ghost state desync.")
                        } else {
                            opts.append("\(k): Suboptimal probe.")
                        }
                    }
                    samples.append((st, opts))
                }
            }
        }
        self.auxHead = try engine.fitOrLoadAuxHead(
            name: "breach",
            choiceInstructions: choiceIns,
            noul0Instructions: desyncIns,
            noul1Instructions: breachIns,
            calibrationSamples: samples
        )
    }

    func decide(st: PortalTriadState, foundFlaws: Set<String>) throws -> BreachDecision {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let ghost = st.desyncSignature()
        let hasDesync = (ghost != nil)
        let nearBreach = hasDesync || (st.jobPhase == "COMPLETE" && st.botMem == "WAITING") || (st.jobPhase == "RUNNING")

        let pref: String
        if !foundFlaws.contains("FLAW-1: AMNESIA_ZOMBIE_TASK") {
            if st.jobPhase == "NONE" || st.botMem == "CLOSED" { pref = "NORMAL" }
            else if st.jobPhase == "RUNNING" { pref = "NORMAL" }
            else if st.jobPhase == "COMPLETE" && st.botMem == "WAITING" { pref = "FAULT" }
            else { pref = "AUDIT" }
        } else if !foundFlaws.contains("FLAW-2: FREE_TOKEN_DRAIN") {
            if st.sseStream == "OPEN_UNBILLED" { pref = "FAULT" }
            else if st.actualVertexUsd - st.ledgerUsd >= 0.50 { pref = "AUDIT" }
            else { pref = "NORMAL" }
        } else if !foundFlaws.contains("FLAW-3: BUDGET_OVERSHOOT_BREACH") {
            if st.jobPhase != "RUNNING" { pref = "NORMAL" }
            else { pref = "BURST" }
        } else {
            if st.activePods >= 2 && (st.jobPhase == "HUNG" || st.botMem == "EMPTY") { pref = "AUDIT" }
            else if st.jobPhase == "RUNNING" { pref = "FAULT" }
            else { pref = "NORMAL" }
        }

        let stateStr = "State desync detected: \(hasDesync ? "yes" : "no"). Invariant breach reachable: \(nearBreach ? "yes" : "no")."
        let stateIds = engine.tokenizer.encode(stateStr)
        let keyPairs = [("normal", "NORMAL"), ("fault", "FAULT"), ("burst", "BURST"), ("audit", "AUDIT")]
        var optStrings: [String] = []
        for (kLow, kUp) in keyPairs {
            let desc: String
            if kUp == pref {
                desc = "Safe. Best probe to expose hidden state flaw."
            } else if kUp == "NORMAL" && hasDesync {
                desc = "Unsafe. Resets active ghost state desync."
            } else {
                desc = "Suboptimal probe."
            }
            optStrings.append("\(kLow): \(desc)")
        }

        let (cIds, cMarkers) = engine.buildSequence(type: "choice", instructions: choiceIns, options: optStrings, stateIds: stateIds)
        let (cProbs, pooled, actProb) = try engine.forwardPass(ids: cIds, markers: cMarkers, qtypeId: 0)

        let desyncProb: Float
        let breachProb: Float
        if threePass {
            let (dIds, dM) = engine.buildSequence(type: "noul", instructions: desyncIns, options: noulOpts, stateIds: stateIds)
            let (dP, _, _) = try engine.forwardPass(ids: dIds, markers: dM, qtypeId: 2)
            let (bIds, bM) = engine.buildSequence(type: "noul", instructions: breachIns, options: noulOpts, stateIds: stateIds)
            let (bP, _, _) = try engine.forwardPass(ids: bIds, markers: bM, qtypeId: 2)
            desyncProb = dP[1]
            breachProb = bP[1]
        } else {
            let (d, b) = auxHead.predict(pooled: pooled)
            desyncProb = d
            breachProb = b
        }

        var probsDict: [String: Float] = [:]
        var proposed = "NORMAL"
        var bestP: Float = -1.0
        for (i, (_, kUp)) in keyPairs.enumerated() {
            probsDict[kUp] = cProbs[i]
            if cProbs[i] > bestP {
                bestP = cProbs[i]
                proposed = kUp
            }
        }

        var executed = proposed
        var bulletTime = false
        var s2Expanded = 0
        var targetHint = ghost ?? "Scanning whiteboard transitions..."

        if guided && (desyncProb >= 0.50 || breachProb >= 0.50 || hasDesync) {
            bulletTime = true
            if let s2 = system2BulletTimeBFS(st: st, alreadyFound: foundFlaws) {
                executed = s2.firstProbe
                s2Expanded = s2.nodesExpanded
                let fCode = s2.flaw.split(separator: ":").first.map(String.init) ?? s2.flaw
                targetHint = "S2 locked onto \(fCode) via \(executed)"
            }
        }

        let dtMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000.0
        return BreachDecision(
            probabilities: probsDict,
            proposed: proposed,
            executed: executed,
            intervened: executed != proposed,
            desyncProb: desyncProb,
            breachProb: breachProb,
            actProb: actProb,
            bulletTimeS2: bulletTime,
            s2NodesExpanded: s2Expanded,
            targetFlawHint: targetHint,
            inferenceMs: dtMs
        )
    }
}

func runChaosMonkey(maxSteps: Int, seed: Int) -> (found: [String: Int], ghosts: Set<String>) {
    var rng = S3SplitMix64(seed: seed)
    var st = PortalTriadState()
    var found: [String: Int] = [:]
    var ghosts: Set<String> = []
    for stepIdx in 1...maxSteps {
        let probe = BREACH_PROBES[rng.choiceIndex(count: BREACH_PROBES.count)]
        let (nxt, _, flaw) = stepEnvironment(st: st, probe: probe)
        st = nxt
        if let g = st.desyncSignature(), let gCode = g.split(separator: ":").first {
            ghosts.insert(String(gCode))
        }
        if let f = flaw, found[f] == nil {
            found[f] = stepIdx
        }
    }
    return (found, ghosts)
}

// MARK: - Benchmark CLI

@main
struct S3BreachApp {
    static func main() throws {
        let args = Array(CommandLine.arguments.dropFirst())
        var maxSteps = 25
        var threePass = false
        var jsonOutPath: String? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--steps":
                if i + 1 < args.count { maxSteps = Int(args[i + 1]) ?? 25; i += 1 }
            case "--three-pass":
                threePass = true
            case "--json-out":
                if i + 1 < args.count { jsonOutPath = args[i + 1]; i += 1 }
            default: break
            }
            i += 1
        }

        let tBoot0 = DispatchTime.now().uptimeNanoseconds
        let engine = try S3ANEEngine()
        let policy = try S3BreachPolicy(engine: engine, guided: true, threePass: threePass)
        let bootMs = Double(DispatchTime.now().uptimeNanoseconds - tBoot0) / 1_000_000.0

        var st = PortalTriadState()
        var foundFlaws: [String: Int] = [:]
        var discoveredGhosts: Set<String> = []
        var latencies: [Double] = []
        var bulletTimes = 0
        var s2NodesTotal = 0
        var stepsTaken = 0

        let tAudit0 = DispatchTime.now().uptimeNanoseconds
        for stepIdx in 1...maxSteps {
            stepsTaken = stepIdx
            let dec = try policy.decide(st: st, foundFlaws: Set(foundFlaws.keys))
            latencies.append(dec.inferenceMs)
            if dec.bulletTimeS2 {
                bulletTimes += 1
                s2NodesTotal += dec.s2NodesExpanded
            }
            let (nxt, _, flaw) = stepEnvironment(st: st, probe: dec.executed)
            st = nxt
            if let g = st.desyncSignature(), let gCode = g.split(separator: ":").first {
                discoveredGhosts.insert(String(gCode))
            }
            if let f = flaw, foundFlaws[f] == nil {
                foundFlaws[f] = stepIdx
                st = PortalTriadState(lastEvent: "Isolated \(f)")
            }
            if foundFlaws.count == BREACH_FATAL_FLAWS.count {
                break
            }
        }
        let auditMs = Double(DispatchTime.now().uptimeNanoseconds - tAudit0) / 1_000_000.0
        let sortedLat = latencies.sorted()
        let p50Lat = sortedLat.isEmpty ? 0.0 : sortedLat[sortedLat.count / 2]
        let meanLat = sortedLat.isEmpty ? 0.0 : sortedLat.reduce(0.0, +) / Double(sortedLat.count)

        let chaosSame = runChaosMonkey(maxSteps: stepsTaken, seed: 101)
        let chaos3x   = runChaosMonkey(maxSteps: stepsTaken * 3, seed: 101)

        let summary: [String: Any] = [
            "arena": "s3-breach",
            "mode": threePass ? "swift_3pass" : "swift_s3_1pass",
            "boot_and_aux_fit_ms": (bootMs * 100.0).rounded() / 100.0,
            "total_audit_ms": (auditMs * 100.0).rounded() / 100.0,
            "steps_to_100pct_breach_discovery": stepsTaken,
            "p50_step_ms": (p50Lat * 100.0).rounded() / 100.0,
            "mean_step_ms": (meanLat * 100.0).rounded() / 100.0,
            "flaws_found": foundFlaws.count,
            "flaws_total": BREACH_FATAL_FLAWS.count,
            "flaw_steps": foundFlaws,
            "ghost_states_split": discoveredGhosts.count,
            "s2_bfs_nodes_expanded": s2NodesTotal,
            "chaos_monkey_12steps_flaws": chaosSame.found.count,
            "chaos_monkey_36steps_flaws": chaos3x.found.count,
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted])
        if let path = jsonOutPath {
            try jsonData.write(to: URL(fileURLWithPath: path))
        }
        print(String(data: jsonData, encoding: .utf8)!)
    }
}
