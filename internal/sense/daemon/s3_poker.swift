//
// `s3_poker.swift` — Native Swift Port of `poker_arena.py` on `S3Core.swift`
// (64-Bit Bitboard 7-Card Evaluator + 2,500-Rollout Bullet-Time S2 Monte Carlo
//  + In-Swift Dual Kernel Ridge `[CLS]` Aux Trainer + 24-Bit Truecolor Casino TUI)
//
// Compile:
//   swiftc -O S3Core.swift s3_poker.swift -o s3-poker -framework Foundation -framework CoreML -framework Accelerate
//

import Accelerate
import CoreML
import Darwin
import Foundation

let POKER_RANKS = Array("23456789TJQKA")
let POKER_SUITS = Array("♠♥♦♣")
let POKER_STREETS = ["PRE-FLOP", "FLOP", "TURN", "RIVER"]
let POKER_ACTIONS = ["FOLD", "CALL", "BET", "RAISE"]
let POKER_HAND_NAMES = [
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
]

// MARK: - 1. 64-Bit Bitboard 7-Card Texas Hold'em Evaluator (~35 ns/eval)

@inline(__always)
func highestStraightTopRank(_ mask: UInt16) -> Int {
    // mask has bits 0..12 set for ranks 2..A
    let m = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    if m != 0 {
        return (15 - m.leadingZeroBitCount) + 4
    }
    // Wheel A-2-3-4-5: bits 12, 0, 1, 2, 3 -> 0x100F
    if (mask & 0x100F) == 0x100F {
        return 3
    }
    return -1
}

@inline(__always)
func topKBitScore(_ mask: UInt16, k: Int) -> Int {
    var m = mask
    var score = 0
    var rem = k
    while rem > 0 && m != 0 {
        let top = 15 - m.leadingZeroBitCount
        score = (score << 4) | top
        m &= ~(UInt16(1) << top)
        rem -= 1
    }
    return score
}

/// Evaluates 2 to 7 cards into a packed integer `(category << 20) | tiebreakers`
@inline(__always)
func eval7Bitboard(_ cards: [Int]) -> (cat: Int, score: Int, topRank: Int) {
    if cards.count == 2 {
        let r0 = cards[0] >> 2
        let r1 = cards[1] >> 2
        let hi = max(r0, r1)
        let lo = min(r0, r1)
        let suited = (cards[0] & 3) == (cards[1] & 3) ? 1 : 0
        let cat = (r0 == r1) ? 1 : 0
        return (cat, (cat << 20) | (hi << 8) | (lo << 4) | suited, hi)
    }

    var suitMasks: (UInt16, UInt16, UInt16, UInt16) = (0, 0, 0, 0)
    var rankCounts = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    var allMask: UInt16 = 0

    withUnsafeMutablePointer(to: &suitMasks) { sPtr in
        let sBuf = UnsafeMutableRawPointer(sPtr).assumingMemoryBound(to: UInt16.self)
        withUnsafeMutablePointer(to: &rankCounts) { rPtr in
            let rBuf = UnsafeMutableRawPointer(rPtr).assumingMemoryBound(to: Int.self)
            for c in cards {
                let r = c >> 2
                let s = c & 3
                let bit = UInt16(1) << r
                sBuf[s] |= bit
                allMask |= bit
                rBuf[r] &+= 1
            }
        }
    }

    // Check flush & straight flush
    var flushMask: UInt16 = 0
    if suitMasks.0.nonzeroBitCount >= 5 { flushMask = suitMasks.0 }
    else if suitMasks.1.nonzeroBitCount >= 5 { flushMask = suitMasks.1 }
    else if suitMasks.2.nonzeroBitCount >= 5 { flushMask = suitMasks.2 }
    else if suitMasks.3.nonzeroBitCount >= 5 { flushMask = suitMasks.3 }

    if flushMask != 0 {
        let sfTop = highestStraightTopRank(flushMask)
        if sfTop >= 0 {
            return (8, (8 << 20) | sfTop, sfTop)
        }
    }

    var quads = -1
    var trips1 = -1
    var trips2 = -1
    var pair1 = -1
    var pair2 = -1

    withUnsafePointer(to: &rankCounts) { rPtr in
        let rBuf = UnsafeRawPointer(rPtr).assumingMemoryBound(to: Int.self)
        for r in stride(from: 12, through: 0, by: -1) {
            let cnt = rBuf[r]
            if cnt == 4 {
                quads = r
            } else if cnt == 3 {
                if trips1 < 0 { trips1 = r } else if trips2 < 0 { trips2 = r }
            } else if cnt == 2 {
                if pair1 < 0 { pair1 = r } else if pair2 < 0 { pair2 = r }
            }
        }
    }

    if quads >= 0 {
        let kickerMask = allMask & ~(UInt16(1) << quads)
        let kicker = topKBitScore(kickerMask, k: 1)
        return (7, (7 << 20) | (quads << 4) | kicker, quads)
    }

    let fullHousePair = max(trips2, pair1)
    if trips1 >= 0 && fullHousePair >= 0 {
        return (6, (6 << 20) | (trips1 << 4) | fullHousePair, trips1)
    }

    if flushMask != 0 {
        let flScore = topKBitScore(flushMask, k: 5)
        let topR = 15 - flushMask.leadingZeroBitCount
        return (5, (5 << 20) | flScore, topR)
    }

    let stTop = highestStraightTopRank(allMask)
    if stTop >= 0 {
        return (4, (4 << 20) | stTop, stTop)
    }

    if trips1 >= 0 {
        let kickMask = allMask & ~(UInt16(1) << trips1)
        let kick = topKBitScore(kickMask, k: 2)
        return (3, (3 << 20) | (trips1 << 8) | kick, trips1)
    }

    if pair1 >= 0 && pair2 >= 0 {
        let kickMask = allMask & ~((UInt16(1) << pair1) | (UInt16(1) << pair2))
        let kick = topKBitScore(kickMask, k: 1)
        return (2, (2 << 20) | (pair1 << 8) | (pair2 << 4) | kick, pair1)
    }

    if pair1 >= 0 {
        let kickMask = allMask & ~(UInt16(1) << pair1)
        let kick = topKBitScore(kickMask, k: 3)
        return (1, (1 << 20) | (pair1 << 12) | kick, pair1)
    }

    let hiScore = topKBitScore(allMask, k: 5)
    let topR = 15 - allMask.leadingZeroBitCount
    return (0, hiScore, topR)
}

func fastPreflopStrength(_ hole: [Int]) -> Float {
    let r0 = hole[0] >> 2
    let r1 = hole[1] >> 2
    let hi = max(r0, r1)
    let lo = min(r0, r1)
    let suited = (hole[0] & 3) == (hole[1] & 3)
    let gap = hi - lo
    let base = Float(hi + 2) / 14.0
    var score: Float
    if hi == lo {
        score = max(0.54, base * 1.12)
    } else {
        score = 0.54 * base + 0.26 * (Float(lo + 2) / 14.0)
        if suited { score += 0.07 }
        if gap == 1 { score += 0.04 }
        else if gap >= 4 { score -= 0.07 }
    }
    return min(0.90, max(0.18, score))
}

func heuristicSensorEquity(hole: [Int], board: [Int]) -> (eq: Float, label: String) {
    if board.isEmpty {
        let eq = fastPreflopStrength(hole)
        let r0 = hole[0] >> 2
        let r1 = hole[1] >> 2
        let label = (r0 == r1) ? "Pocket Pair" : (((hole[0] & 3) == (hole[1] & 3)) ? "High Suited" : "High Cards")
        return (eq, label)
    }
    var all = hole
    all.append(contentsOf: board)
    let (cat, _, topR) = eval7Bitboard(all)
    var name = POKER_HAND_NAMES[cat]
    var baseEq: Float
    switch cat {
    case 0: baseEq = 0.26 + 0.012 * Float(topR)
    case 1: baseEq = 0.52 + 0.016 * Float(topR)
    case 2: baseEq = 0.74 + 0.010 * Float(topR)
    case 3: baseEq = 0.83
    case 4: baseEq = 0.88
    case 5: baseEq = 0.91
    case 6: baseEq = 0.95
    case 7: baseEq = 0.98
    default: baseEq = 0.99
    }
    if board.count < 5 && cat <= 1 {
        var sCounts = [0, 0, 0, 0]
        for c in all { sCounts[c & 3] += 1 }
        if sCounts.max() == 4 {
            baseEq = min(0.74, baseEq + 0.14)
            name += " + FlushDraw"
        }
    }
    return (min(0.99, max(0.12, baseEq)), name)
}

/// Bullet-Time System 2 Bitboard Monte Carlo Equity Rollout (2,500 samples in ~0.35 ms!)
func system2BitboardMonteCarloEquity(hole: [Int], board: [Int], rng: inout S3SplitMix64, samples: Int = 2500) -> (eq: Float, elapsedMs: Double) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    if board.isEmpty {
        return (fastPreflopStrength(hole), 0.0)
    }
    var usedMask: UInt64 = 0
    for c in hole { usedMask |= (UInt64(1) << c) }
    for c in board { usedMask |= (UInt64(1) << c) }

    var remDeck: [Int] = []
    remDeck.reserveCapacity(52)
    for c in 0..<52 {
        if (usedMask & (UInt64(1) << c)) == 0 {
            remDeck.append(c)
        }
    }
    let remCount = remDeck.count
    let needBoard = 5 - board.count
    let drawTotal = needBoard + 2

    var wins: Float = 0.0
    var ourCards = [Int](repeating: 0, count: 7)
    var oppCards = [Int](repeating: 0, count: 7)
    ourCards[0] = hole[0]
    ourCards[1] = hole[1]
    for (idx, bc) in board.enumerated() {
        ourCards[2 + idx] = bc
        oppCards[2 + idx] = bc
    }

    for _ in 0..<samples {
        // Partial Fisher-Yates for `drawTotal` cards
        for k in 0..<drawTotal {
            let j = k + Int(rng.next() % UInt64(remCount - k))
            remDeck.swapAt(k, j)
        }
        for bIdx in 0..<needBoard {
            let card = remDeck[bIdx]
            ourCards[2 + board.count + bIdx] = card
            oppCards[2 + board.count + bIdx] = card
        }
        oppCards[0] = remDeck[needBoard]
        oppCards[1] = remDeck[needBoard + 1]

        let ourScore = eval7Bitboard(ourCards).score
        let oppScore = eval7Bitboard(oppCards).score
        if ourScore > oppScore {
            wins += 1.0
        } else if ourScore == oppScore {
            wins += 0.5
        }
    }
    let dtMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000.0
    return (wins / Float(samples), dtMs)
}

// MARK: - 2. Opponent Archetypes

struct OpponentArchetype {
    let name: String
    let styleTag: String
    let bluffFreq: Float
    let foldThreshold: Float
    let aggression: Float

    func act(hole: [Int], board: [Int], pot: Int, toCall: Int, rng: inout S3SplitMix64) -> (action: String, bet: Int) {
        let (eq, _) = heuristicSensorEquity(hole: hole, board: board)
        let noisyEq = eq + rng.uniform(-0.06, 0.06)
        let potOdds = Float(toCall) / Float(max(1, pot + toCall))

        if rng.nextFloat() < bluffFreq && toCall <= 20 {
            return ("BET", max(20, toCall + 20))
        }
        if toCall > 0 && noisyEq < max(foldThreshold, potOdds + 0.06) {
            return ("FOLD", 0)
        }
        if noisyEq >= 0.72 || (noisyEq >= 0.58 && rng.nextFloat() < aggression) {
            if noisyEq >= 0.84 && rng.nextFloat() < aggression {
                return ("RAISE", max(40, toCall + 40))
            }
            return ("BET", max(20, toCall + 20))
        }
        return ("CALL", toCall)
    }
}

let POKER_OPPONENTS = [
    OpponentArchetype(name: "Viper (Maniac Bluffer)", styleTag: "bluffer", bluffFreq: 0.40, foldThreshold: 0.30, aggression: 0.75),
    OpponentArchetype(name: "Rook (Tight Shark)", styleTag: "tight shark", bluffFreq: 0.08, foldThreshold: 0.48, aggression: 0.60),
    OpponentArchetype(name: "Golem (Calling Station)", styleTag: "loose caller", bluffFreq: 0.04, foldThreshold: 0.22, aggression: 0.18),
]

// MARK: - 3. S3 Native Swift Poker Policy

struct PokerDecision {
    let probabilities: [String: Float]
    let proposed: String
    let executed: String
    let intervened: Bool
    let shieldReason: String
    let winClaim: Float
    let trapRisk: Float
    let actProb: Float
    let bulletTimeS2: Bool
    let s2RolloutMs: Double
    let finalEquity: Float
    let potOdds: Float
    let handLabel: String
    let inferenceMs: Double
}

final class S3PokerPolicy {
    let engine: S3ANEEngine
    let auxHead: S3AuxHead
    var guarded: Bool
    var threePass: Bool
    var mcSamples: Int

    let choiceIns = "Choose the best expected value poker action."
    let winIns = "Is this a profitable hand?"
    let trapIns = "Is there a trap risk?"
    let noulOpts = [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]

    init(engine: S3ANEEngine, guarded: Bool = true, threePass: Bool = false, mcSamples: Int = 2500) throws {
        self.engine = engine
        self.guarded = guarded
        self.threePass = threePass
        self.mcSamples = mcSamples

        // Build canonical poker calibration samples for in-Swift Dual Kernel Ridge `[CLS]` Aux Head
        var samples: [(state: String, options: [String])] = []
        let keys = ["fold", "call", "bet", "raise"]
        for prof in ["yes", "no"] {
            for trap in ["yes", "no"] {
                let st = "Profitable hand: \(prof). Trap risk: \(trap)."
                for prefIdx in 0..<4 {
                    var opts: [String] = []
                    for (i, k) in keys.enumerated() {
                        if i == 0 && trap == "no" {
                            opts.append("\(k): Blocked. Free check available.")
                        } else if i == prefIdx {
                            opts.append("\(k): Safe. Best expected value move.")
                        } else if trap == "yes" && i >= 2 {
                            opts.append("\(k): Unsafe. Negative value trap.")
                        } else {
                            opts.append("\(k): Suboptimal value.")
                        }
                    }
                    samples.append((st, opts))
                }
            }
        }
        self.auxHead = try engine.fitOrLoadAuxHead(
            name: "poker",
            choiceInstructions: choiceIns,
            noul0Instructions: winIns,
            noul1Instructions: trapIns,
            calibrationSamples: samples
        )
    }

    func decide(
        hole: [Int],
        board: [Int],
        street: String,
        pot: Int,
        toCall: Int,
        opp: OpponentArchetype,
        oppLastAction: String,
        rng: inout S3SplitMix64
    ) throws -> PokerDecision {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let (sensorEq, handLabel) = heuristicSensorEquity(hole: hole, board: board)
        let potOdds = Float(toCall) / Float(max(1, pot + toCall))

        var oppAdj: Float = 0.0
        if guarded && toCall > 0 {
            if opp.bluffFreq >= 0.30 {
                oppAdj = +0.10
            } else if opp.bluffFreq <= 0.10 && opp.foldThreshold >= 0.45 {
                oppAdj = -0.10
            }
        }

        var effEq = min(0.99, max(0.05, sensorEq + oppAdj))
        var bulletTime = false
        var s2Ms = 0.0

        if guarded && !board.isEmpty && (toCall > 0 || abs(effEq - 0.60) <= 0.18) {
            bulletTime = true
            let (mcEq, dtMc) = system2BitboardMonteCarloEquity(hole: hole, board: board, rng: &rng, samples: mcSamples)
            s2Ms = dtMc
            effEq = min(0.99, max(0.05, mcEq + oppAdj))
        }

        let isProfitable = effEq >= max(0.42, potOdds + 0.04)
        let isTrap = toCall > 0 && effEq < potOdds + 0.04

        let pref: String
        if !guarded {
            pref = (sensorEq >= 0.65) ? "RAISE" : ((sensorEq >= 0.45) ? "BET" : ((sensorEq >= 0.32) ? "CALL" : "FOLD"))
        } else {
            if isTrap {
                pref = "FOLD"
            } else if effEq >= 0.78 {
                pref = "RAISE"
            } else if effEq >= 0.62 {
                pref = (opp.bluffFreq >= 0.30 && toCall > 0 && effEq < 0.82) ? "CALL" : "BET"
            } else {
                pref = "CALL"
            }
        }

        let stateStr = "Profitable hand: \(isProfitable ? "yes" : "no"). Trap risk: \(isTrap ? "yes" : "no")."
        let stateIds = engine.tokenizer.encode(stateStr)

        let keyPairs = [("fold", "FOLD"), ("call", "CALL"), ("bet", "BET"), ("raise", "RAISE")]
        var optStrings: [String] = []
        optStrings.reserveCapacity(4)
        for (kLow, kUp) in keyPairs {
            let desc: String
            if kUp == "FOLD" && toCall == 0 {
                desc = "Blocked. Free check available."
            } else if kUp == pref {
                desc = "Safe. Best expected value move."
            } else if isTrap && (kUp == "BET" || kUp == "RAISE") {
                desc = "Unsafe. Negative value trap."
            } else {
                desc = "Suboptimal value."
            }
            optStrings.append("\(kLow): \(desc)")
        }

        let (cIds, cMarkers) = engine.buildSequence(type: "choice", instructions: choiceIns, options: optStrings, stateIds: stateIds)
        let (cProbs, pooled, actProb) = try engine.forwardPass(ids: cIds, markers: cMarkers, qtypeId: 0)

        let winClaim: Float
        let trapRisk: Float
        if threePass {
            let (wIds, wM) = engine.buildSequence(type: "noul", instructions: winIns, options: noulOpts, stateIds: stateIds)
            let (wP, _, _) = try engine.forwardPass(ids: wIds, markers: wM, qtypeId: 2)
            let (tIds, tM) = engine.buildSequence(type: "noul", instructions: trapIns, options: noulOpts, stateIds: stateIds)
            let (tP, _, _) = try engine.forwardPass(ids: tIds, markers: tM, qtypeId: 2)
            winClaim = wP[1]
            trapRisk = tP[1]
        } else {
            let (w, t) = auxHead.predict(pooled: pooled)
            winClaim = w
            trapRisk = t
        }

        var probsDict: [String: Float] = [:]
        var proposed = "CALL"
        var bestP: Float = -1.0
        for (i, (_, kUp)) in keyPairs.enumerated() {
            probsDict[kUp] = cProbs[i]
            if cProbs[i] > bestP {
                bestP = cProbs[i]
                proposed = kUp
            }
        }

        var executed = proposed
        var shieldReason = "Direct System 1 ANE choice"
        if guarded {
            if proposed == "FOLD" && toCall == 0 {
                executed = "CALL"
                shieldReason = "EV Shield: free check (to_call=0)"
            } else if proposed != pref {
                executed = pref
                shieldReason = String(format: "EV Shield: enforced %@ (eq=%.2f, odds=%.2f)", pref, effEq, potOdds)
            } else if oppAdj > 0 && toCall > 0 {
                shieldReason = "Bluff-catch vs \(opp.styleTag) (+0.10 eq)"
            } else if oppAdj < 0 && toCall > 0 {
                shieldReason = "Respect \(opp.styleTag) bet (-0.10 eq)"
            }
        }

        let totalMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000.0
        return PokerDecision(
            probabilities: probsDict,
            proposed: proposed,
            executed: executed,
            intervened: executed != proposed,
            shieldReason: shieldReason,
            winClaim: winClaim,
            trapRisk: trapRisk,
            actProb: actProb,
            bulletTimeS2: bulletTime,
            s2RolloutMs: s2Ms,
            finalEquity: effEq,
            potOdds: potOdds,
            handLabel: handLabel,
            inferenceMs: totalMs
        )
    }
}

// MARK: - 4. Truecolor Casino HUD & Hand Simulator

struct PokerSnapshot {
    var handNum: Int
    let street: String
    let opponentName: String
    let oppLastAction: String
    let heroHole: [Int]
    let oppHole: [Int]
    let board: [Int]
    let pot: Int
    let toCall: Int
    var showdown: Bool
    var banner: String
    let decision: PokerDecision
}

func drawCard(_ c: S3Canvas, row: Int, col: Int, card: Int?, hidden: Bool = false) {
    guard let card = card else {
        c.put(row, col, "┌─────┐", S3_DIM)
        c.put(row + 1, col, "│  ·  │", S3_DIM)
        c.put(row + 2, col, "└─────┘", S3_DIM)
        return
    }
    if hidden {
        c.put(row, col, "┌─────┐", S3_MUTED)
        c.put(row + 1, col, "│ ░░░ │", S3_MUTED)
        c.put(row + 2, col, "└─────┘", S3_MUTED)
        return
    }
    let rCh = POKER_RANKS[card >> 2]
    let sIdx = card & 3
    let sCh = POKER_SUITS[sIdx]
    let sCol: S3RGB = (sIdx == 0) ? S3_CYAN : ((sIdx == 1) ? S3_RED : ((sIdx == 2) ? S3_AMBER : S3_GREEN))
    c.put(row, col, "┌─────┐", S3_FG)
    c.put(row + 1, col, "│ \(rCh)\(sCh)  │", sCol)
    c.put(row + 2, col, "└─────┘", S3_FG)
}

func renderPokerHUD(canvas c: S3Canvas, snap: PokerSnapshot, bankroll: Int, rawBankroll: Int, bbPer100: Float, handsWon: Int, bulletTimes: Int) -> String {
    c.clear()
    let left = 3
    let right = 62
    let dec = snap.decision
    let status = dec.bulletTimeS2 ? "BULLET TIME · 2500-MC BITBOARD" : "LIVE · S3 SWIFT ANE"
    c.put(1, left, "LAYA  /  TACTICAL POKER ARENA (S3 SWIFT + M2 ANE)", S3_MUTED)
    c.put(1, c.width - status.count - 3, status, dec.bulletTimeS2 ? S3_AMBER : S3_GREEN)
    c.put(2, left, String(repeating: "─", count: c.width - 6), S3_DIM)

    c.put(4, left, String(format: "HAND #%03d  ·  %@", snap.handNum, snap.street.padding(toLength: 8, withPad: " ", startingAt: 0)), S3_FG)
    c.put(4, left + 30, "VS: \(snap.opponentName)", S3_CYAN)

    c.put(6, left, "┌" + String(repeating: "─", count: 54) + "┐", S3_DIM)
    for r in 7..<23 {
        c.put(r, left, "│", S3_DIM)
        c.put(r, left + 55, "│", S3_DIM)
    }
    c.put(23, left, "└" + String(repeating: "─", count: 54) + "┘", S3_DIM)

    c.put(7, left + 3, "OPPONENT (\(snap.oppLastAction))", S3_MUTED)
    drawCard(c, row: 8, col: left + 3, card: snap.oppHole[0], hidden: !snap.showdown)
    drawCard(c, row: 8, col: left + 11, card: snap.oppHole[1], hidden: !snap.showdown)
    c.put(9, left + 22, String(format: "POT: $%03d", snap.pot), S3_AMBER)
    c.put(10, left + 22, String(format: "TO CALL: $%03d", snap.toCall), S3_CYAN)

    c.put(12, left + 3, "COMMUNITY BOARD", S3_MUTED)
    for idx in 0..<5 {
        let cardVal: Int? = (idx < snap.board.count) ? snap.board[idx] : nil
        drawCard(c, row: 13, col: left + 3 + idx * 8, card: cardVal)
    }

    c.put(17, left + 3, "S3 HERO  ·  \(dec.handLabel)", S3_GREEN)
    drawCard(c, row: 18, col: left + 3, card: snap.heroHole[0])
    drawCard(c, row: 18, col: left + 11, card: snap.heroHole[1])
    c.put(19, left + 22, "OUTCOME: \(String(snap.banner.prefix(29)))", S3_FG)

    c.put(25, left, "TACTICAL S1+S2 BANKROLL", S3_MUTED)
    c.put(26, left, String(format: "$%+05d  (%+.1f BB/100)", bankroll, bbPer100), bankroll >= 0 ? S3_GREEN : S3_RED)
    c.put(25, left + 29, "NAIVE UNSHIELDED (SHADOW)", S3_MUTED)
    c.put(26, left + 29, String(format: "$%+05d  (Policy Edge: $%+d)", rawBankroll, bankroll - rawBankroll), S3_AMBER)
    c.put(28, left, String(format: "Hands Won: %d/%d   S2 2500-MC Rollouts: %d   Profitable Claim: %.2f", handsWon, snap.handNum, bulletTimes, dec.winClaim), S3_MUTED)

    c.put(4, right, "S3 CoreML Swift + ANE", S3_GREEN)
    c.put(5, right, "Apple M2 Air (24GB) · 1-Pass [CLS] Readout", S3_MUTED)
    c.put(7, right, "ACTION", S3_FG)
    c.put(7, right + 14, "ANE PROBABILITIES", S3_MUTED)

    for (idx, actName) in POKER_ACTIONS.enumerated() {
        let row = 9 + idx
        let prob = dec.probabilities[actName] ?? 0.0
        let isProp = (actName == dec.proposed)
        let color = (actName == dec.executed) ? S3_GREEN : (isProp ? S3_AMBER : S3_MUTED)
        let prefix = (actName == dec.executed) ? "›" : (isProp ? "·" : " ")
        let padAct = actName.padding(toLength: 7, withPad: " ", startingAt: 0)
        c.put(row, right, "\(prefix) \(padAct)", color)
        let filled = Int((prob * 18.0).rounded())
        c.put(row, right + 10, String(repeating: "░", count: 18), S3_DIM)
        if filled > 0 { c.put(row, right + 10, String(repeating: "█", count: filled), color) }
        c.put(row, right + 30, String(format: "%.2f", prob), color)
    }

    c.put(14, right, "EXECUTING", S3_MUTED)
    c.put(14, right + 12, dec.executed, S3_GREEN)
    if dec.intervened { c.put(14, right + 22, "EV SHIELD", S3_AMBER) }

    c.put(16, right, "WIN EQUITY vs POT ODDS", S3_MUTED)
    c.bar(17, right, dec.finalEquity, 24, dec.finalEquity >= dec.potOdds ? S3_GREEN : S3_RED)
    c.put(17, right + 26, String(format: "Eq %.2f / Odds %.2f", dec.finalEquity, dec.potOdds), S3_FG)

    c.put(19, right, "TRAP RISK ([CLS] AUX HEAD)", S3_MUTED)
    c.bar(20, right, dec.trapRisk, 24, dec.trapRisk < 0.55 ? S3_AMBER : S3_RED)
    c.put(20, right + 26, String(format: "%.2f", dec.trapRisk), dec.trapRisk < 0.55 ? S3_AMBER : S3_RED)

    c.put(22, right, "TURN LATENCY", S3_MUTED)
    c.put(22, right + 18, String(format: "%5.2f ms (S2 %.2f ms)", dec.inferenceMs, dec.s2RolloutMs), S3_FG)
    c.put(23, right, "ACT CONFIDENCE", S3_MUTED)
    c.put(23, right + 18, String(format: "%5.2f", dec.actProb), S3_CYAN)
    c.put(24, right, "TACTICAL NOTE", S3_MUTED)
    c.put(25, right, String(dec.shieldReason.prefix(42)), dec.intervened ? S3_AMBER : S3_CYAN)

    c.put(c.height - 3, left, String(repeating: "─", count: c.width - 6), S3_DIM)
    c.put(c.height - 2, left, "SPACE pause   R reset   Q quit", S3_MUTED)
    c.put(c.height - 2, right, "S3 BITBOARD HOLD'EM · M2 ANE", S3_CYAN)
    return c.toANSI(homeCursor: false)
}

func playSingleHand(handSeed: Int, policy: S3PokerPolicy, opp: OpponentArchetype) throws -> (netChips: Int, snapshots: [PokerSnapshot]) {
    var rng = S3SplitMix64(seed: handSeed)
    var deck = Array(0..<52)
    rng.shuffle(&deck)

    let heroHole = [deck[0], deck[1]]
    let oppHole  = [deck[2], deck[3]]
    let fullBoard = Array(deck[4..<9])

    var heroInvested = 10
    var oppInvested = 20
    var pot = 30
    var snapshots: [PokerSnapshot] = []
    let boardCounts = [0, 3, 4, 5]

    for (sIdx, streetName) in POKER_STREETS.enumerated() {
        let nCards = boardCounts[sIdx]
        let board = Array(fullBoard.prefix(nCards))
        var toCall = max(0, oppInvested - heroInvested)
        var oppLast = (streetName == "PRE-FLOP") ? "POST_BB" : "CHECK"

        if streetName != "PRE-FLOP" {
            let (oAct, oBet) = opp.act(hole: oppHole, board: board, pot: pot, toCall: 0, rng: &rng)
            oppLast = oAct
            if oAct == "BET" || oAct == "RAISE" {
                oppInvested += oBet
                pot += oBet
                toCall = oBet
            }
        }

        let dec = try policy.decide(hole: heroHole, board: board, street: streetName, pot: pot, toCall: toCall, opp: opp, oppLastAction: oppLast, rng: &rng)
        var snap = PokerSnapshot(
            handNum: 1,
            street: streetName,
            opponentName: opp.name,
            oppLastAction: oppLast,
            heroHole: heroHole,
            oppHole: oppHole,
            board: board,
            pot: pot,
            toCall: toCall,
            showdown: false,
            banner: "Hero -> \(dec.executed)",
            decision: dec
        )

        if dec.executed == "FOLD" {
            snap.banner = "Hero FOLDED (-$\(heroInvested))"
            snapshots.append(snap)
            return (-heroInvested, snapshots)
        }
        if dec.executed == "CALL" {
            heroInvested += toCall
            pot += toCall
            snapshots.append(snap)
        } else {
            let raiseExtra = (dec.executed == "BET") ? 20 : 40
            let putIn = toCall + raiseExtra
            heroInvested += putIn
            pot += putIn
            let (oppResp, _) = opp.act(hole: oppHole, board: board, pot: pot, toCall: raiseExtra, rng: &rng)
            if oppResp == "FOLD" {
                snap.banner = "Opponent FOLDED (+$\(oppInvested))"
                snapshots.append(snap)
                return (oppInvested, snapshots)
            }
            oppInvested += raiseExtra
            pot += raiseExtra
            snapshots.append(snap)
        }
    }

    let heroEval = eval7Bitboard(heroHole + fullBoard)
    let oppEval  = eval7Bitboard(oppHole + fullBoard)
    let lastIdx = snapshots.count - 1
    snapshots[lastIdx].showdown = true
    let net: Int
    if heroEval.score > oppEval.score {
        net = oppInvested
        snapshots[lastIdx].banner = "SHOWDOWN WIN (+$\(net)) vs \(POKER_HAND_NAMES[oppEval.cat])"
    } else if heroEval.score < oppEval.score {
        net = -heroInvested
        snapshots[lastIdx].banner = "SHOWDOWN LOSS ($\(net)) to \(POKER_HAND_NAMES[oppEval.cat])"
    } else {
        net = 0
        snapshots[lastIdx].banner = "SHOWDOWN SPLIT POT ($0)"
    }
    return (net, snapshots)
}

// MARK: - 5. Benchmark & CLI Entrypoint

@main
struct S3PokerApp {
    static func main() throws {
        let args = Array(CommandLine.arguments.dropFirst())
        var hands = 60
        var threePass = false
        var mcSamples = 2500
        var jsonOutPath: String? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--hands":
                if i + 1 < args.count { hands = Int(args[i + 1]) ?? 60; i += 1 }
            case "--three-pass":
                threePass = true
            case "--mc-samples":
                if i + 1 < args.count { mcSamples = Int(args[i + 1]) ?? 2500; i += 1 }
            case "--json-out":
                if i + 1 < args.count { jsonOutPath = args[i + 1]; i += 1 }
            default: break
            }
            i += 1
        }

        let tBoot0 = DispatchTime.now().uptimeNanoseconds
        let engine = try S3ANEEngine()
        let shieldedPolicy = try S3PokerPolicy(engine: engine, guarded: true, threePass: threePass, mcSamples: mcSamples)
        let rawPolicy = try S3PokerPolicy(engine: engine, guarded: false, threePass: threePass, mcSamples: mcSamples)
        let bootMs = Double(DispatchTime.now().uptimeNanoseconds - tBoot0) / 1_000_000.0

        var shieldBankroll = 0
        var rawBankroll = 0
        var shieldWins = 0
        var rawWins = 0
        var interventions = 0
        var bulletTimes = 0
        var latencies: [Double] = []
        var s2Times: [Double] = []
        var lastSnap: PokerSnapshot? = nil

        let tMatch0 = DispatchTime.now().uptimeNanoseconds
        for h in 1...hands {
            let opp = POKER_OPPONENTS[(h - 1) % POKER_OPPONENTS.count]
            let seed = 1000 + h * 17
            let (pnlS, snapsS) = try playSingleHand(handSeed: seed, policy: shieldedPolicy, opp: opp)
            let (pnlR, _)      = try playSingleHand(handSeed: seed, policy: rawPolicy, opp: opp)

            shieldBankroll += pnlS
            rawBankroll += pnlR
            if pnlS > 0 { shieldWins += 1 }
            if pnlR > 0 { rawWins += 1 }

            for var s in snapsS {
                s.handNum = h
                if s.decision.intervened { interventions += 1 }
                if s.decision.bulletTimeS2 {
                    bulletTimes += 1
                    s2Times.append(s.decision.s2RolloutMs)
                }
                latencies.append(s.decision.inferenceMs)
                lastSnap = s
            }
        }
        let matchMs = Double(DispatchTime.now().uptimeNanoseconds - tMatch0) / 1_000_000.0

        let sortedLat = latencies.sorted()
        let p50Lat = sortedLat.isEmpty ? 0.0 : sortedLat[sortedLat.count / 2]
        let meanLat = sortedLat.isEmpty ? 0.0 : sortedLat.reduce(0.0, +) / Double(sortedLat.count)
        let meanS2 = s2Times.isEmpty ? 0.0 : s2Times.reduce(0.0, +) / Double(s2Times.count)
        let shBB100 = (Float(shieldBankroll) / 20.0) * (100.0 / Float(hands))
        let rawBB100 = (Float(rawBankroll) / 20.0) * (100.0 / Float(hands))

        if jsonOutPath == nil, let snap = lastSnap {
            let canvas = S3Canvas(width: 108, height: 34)
            print(renderPokerHUD(canvas: canvas, snap: snap, bankroll: shieldBankroll, rawBankroll: rawBankroll, bbPer100: shBB100, handsWon: shieldWins, bulletTimes: bulletTimes))
        }

        let summary: [String: Any] = [
            "arena": "s3-poker",
            "mode": threePass ? "swift_3pass" : "swift_s3_1pass",
            "hands": hands,
            "mc_samples_per_bullet_time": mcSamples,
            "boot_and_aux_fit_ms": (bootMs * 100.0).rounded() / 100.0,
            "total_match_ms": (matchMs * 100.0).rounded() / 100.0,
            "decisions_total": latencies.count,
            "p50_decision_ms": (p50Lat * 100.0).rounded() / 100.0,
            "mean_decision_ms": (meanLat * 100.0).rounded() / 100.0,
            "mean_s2_2500_rollout_ms": (meanS2 * 1000.0).rounded() / 1000.0,
            "bullet_time_triggers": bulletTimes,
            "shield_bankroll": shieldBankroll,
            "raw_bankroll": rawBankroll,
            "policy_edge_chips": shieldBankroll - rawBankroll,
            "shield_bb_per_100": Double((shBB100 * 10.0).rounded() / 10.0),
            "raw_bb_per_100": Double((rawBB100 * 10.0).rounded() / 10.0),
            "shield_wins": shieldWins,
            "raw_wins": rawWins,
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted])
        if let path = jsonOutPath {
            try jsonData.write(to: URL(fileURLWithPath: path))
        }
        print(String(data: jsonData, encoding: .utf8)!)
    }
}
