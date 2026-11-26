//
// `s3_tactics.swift` — Native Swift Port of `tactics_arena.py` on `S3Core.swift`
// (Final Fantasy Tactics 8x6 Elevation Grid + CT Turn Clock + Directional Backstab/Counter Math
//  + Value-Type Multi-Ply Bullet-Time CT Timeline Lookahead + In-Swift `[CLS]` Aux Head)
//
// Compile:
//   swiftc -O S3Core.swift s3_tactics.swift -o s3-tactics -framework Foundation -framework CoreML -framework Accelerate
//

import Accelerate
import CoreML
import Darwin
import Foundation

let TACTICS_W = 8
let TACTICS_H = 6
let TACTICS_POSTURES = ["LETHAL", "FLANK", "SUPPORT", "KITE"]

struct TUnit {
    let uid: String
    let name: String
    let team: String // "BLUE" | "RED"
    let role: String // "Knight" | "Archer" | "Mage"
    let glyph: String
    var x: Int
    var y: Int
    var facing: String // "N" | "S" | "E" | "W"
    var hp: Int
    let maxHp: Int
    var mp: Int
    let maxMp: Int
    let speed: Int
    let moveRange: Int
    var ct: Int
    var alive: Bool = true
}

struct TManeuver {
    var posture: String
    let destX: Int
    let destY: Int
    let newFacing: String
    let ability: String // "Slash" | "AimBow" | "Fireball" | "Cure" | "Wait"
    let targetUid: String?
    let expectedDmg: Float
    let expectedHeal: Float
    let hitProb: Float
    let angle: String // "FRONT" | "SIDE" | "BACK" | "NONE"
    let triggersCounter: Bool
    let friendlyFireDmg: Int
    let exposureCount: Int
    let ctCost: Int
    var score: Float
    let summary: String
}

@inline(__always)
func manhattan(_ x1: Int, _ y1: Int, _ x2: Int, _ y2: Int) -> Int {
    return abs(x1 - x2) + abs(y1 - y2)
}

@inline(__always)
func facingVec(_ f: String) -> (Int, Int) {
    switch f {
    case "N": return (0, -1)
    case "S": return (0, 1)
    case "E": return (1, 0)
    default:  return (-1, 0)
    }
}

@inline(__always)
func facingArrow(_ f: String) -> String {
    switch f {
    case "N": return "▲"
    case "S": return "▼"
    case "E": return "►"
    default:  return "◄"
    }
}

func bestFacingToward(_ fx: Int, _ fy: Int, _ tx: Int, _ ty: Int) -> String {
    let dx = tx - fx
    let dy = ty - fy
    if abs(dx) >= abs(dy) { return dx >= 0 ? "E" : "W" }
    return dy >= 0 ? "S" : "N"
}

func bestFacingTowardNearestEnemy(_ x: Int, _ y: Int, _ enemies: [TUnit]) -> String {
    let living = enemies.filter { $0.alive }
    guard let nearest = living.min(by: { manhattan(x, y, $0.x, $0.y) < manhattan(x, y, $1.x, $1.y) }) else {
        return "E"
    }
    return bestFacingToward(x, y, nearest.x, nearest.y)
}

func attackAngle(attackerX: Int, attackerY: Int, target: TUnit) -> String {
    let (fx, fy) = facingVec(target.facing)
    let vx = attackerX - target.x
    let vy = attackerY - target.y
    let dot = fx * vx + fy * vy
    if dot > 0 { return "FRONT" }
    if dot < 0 { return "BACK" }
    return "SIDE"
}

func buildBattlefield(seed: Int) -> (elev: [[Int]], units: [TUnit]) {
    var rng = S3SplitMix64(seed: seed)
    var elev = [[Int]](repeating: [Int](repeating: 0, count: TACTICS_W), count: TACTICS_H)
    for y in 0..<TACTICS_H {
        for x in 0..<TACTICS_W {
            if (x == 3 || x == 4) && (1...4).contains(y) {
                elev[y][x] = ((x + y + seed) % 3 == 0) ? 2 : 1
            } else if (x == 2 || x == 5) && ((y + seed) % 2 == 0) {
                elev[y][x] = 1
            }
        }
    }
    let units = [
        TUnit(uid: "B_K", name: "Ramza (Knight)", team: "BLUE", role: "Knight", glyph: "K", x: 1, y: 2, facing: "E", hp: 125, maxHp: 125, mp: 0, maxMp: 0, speed: 11, moveRange: 3, ct: rng.randint(20, 45)),
        TUnit(uid: "B_A", name: "Mustadio (Archer)", team: "BLUE", role: "Archer", glyph: "A", x: 0, y: 1, facing: "E", hp: 85, maxHp: 85, mp: 0, maxMp: 0, speed: 12, moveRange: 3, ct: rng.randint(25, 50)),
        TUnit(uid: "B_M", name: "Agrias (Mage)", team: "BLUE", role: "Mage", glyph: "M", x: 0, y: 4, facing: "E", hp: 75, maxHp: 75, mp: 36, maxMp: 36, speed: 10, moveRange: 3, ct: rng.randint(15, 40)),
        TUnit(uid: "R_K", name: "Gafgarion (Knight)", team: "RED", role: "Knight", glyph: "K", x: 6, y: 3, facing: "W", hp: 125, maxHp: 125, mp: 0, maxMp: 0, speed: 11, moveRange: 3, ct: rng.randint(20, 45)),
        TUnit(uid: "R_A", name: "Brigand (Archer)", team: "RED", role: "Archer", glyph: "A", x: 7, y: 1, facing: "W", hp: 85, maxHp: 85, mp: 0, maxMp: 0, speed: 12, moveRange: 3, ct: rng.randint(25, 50)),
        TUnit(uid: "R_M", name: "Summoner (Mage)", team: "RED", role: "Mage", glyph: "M", x: 7, y: 4, facing: "W", hp: 75, maxHp: 75, mp: 36, maxMp: 36, speed: 10, moveRange: 3, ct: rng.randint(15, 40)),
    ]
    return (elev, units)
}

func reachableTiles(unit: TUnit, units: [TUnit], elev: [[Int]]) -> [(Int, Int)] {
    var tiles: [(Int, Int)] = [(unit.x, unit.y)]
    for y in 0..<TACTICS_H {
        for x in 0..<TACTICS_W {
            if x == unit.x && y == unit.y { continue }
            var occ = false
            for u in units where u.alive && u.uid != unit.uid {
                if u.x == x && u.y == y { occ = true; break }
            }
            if occ { continue }
            let dist = manhattan(unit.x, unit.y, x, y)
            if dist <= unit.moveRange && abs(elev[y][x] - elev[unit.y][unit.x]) <= 1 {
                tiles.append((x, y))
            }
        }
    }
    return tiles
}

func countEnemyThreats(x: Int, y: Int, enemies: [TUnit]) -> Int {
    var threats = 0
    for e in enemies where e.alive {
        let reach = e.moveRange + ((e.role == "Knight") ? 1 : ((e.role == "Archer") ? 4 : 3))
        if manhattan(e.x, e.y, x, y) <= reach { threats += 1 }
    }
    return threats
}

/// Deep Bullet-Time System 2 CT Timeline Lookahead:
/// Simulates the actual CT Turn-Order Clock across the next 5 unit activations in value-type structs!
func system2DeepCTTimelineLookahead(actor: TUnit, cand: TManeuver, units: [TUnit]) -> (survives: Bool, statesEvaluated: Int) {
    var simUnits = units
    var statesVisited = 1
    guard let aIdx = simUnits.firstIndex(where: { $0.uid == actor.uid }) else { return (true, 1) }

    simUnits[aIdx].x = cand.destX
    simUnits[aIdx].y = cand.destY
    simUnits[aIdx].facing = cand.newFacing
    simUnits[aIdx].ct = max(0, simUnits[aIdx].ct - (50 + cand.ctCost))

    if let tUid = cand.targetUid, let tIdx = simUnits.firstIndex(where: { $0.uid == tUid && $0.alive }) {
        if cand.ability == "Cure" {
            simUnits[tIdx].hp = min(simUnits[tIdx].maxHp, simUnits[tIdx].hp + Int(cand.expectedHeal))
        } else if cand.expectedDmg > 0 {
            let dmg = Int(cand.expectedDmg)
            simUnits[tIdx].hp = max(0, simUnits[tIdx].hp - dmg)
            if simUnits[tIdx].hp == 0 {
                simUnits[tIdx].alive = false
            } else if cand.triggersCounter {
                simUnits[aIdx].hp = max(0, simUnits[aIdx].hp - 22)
            }
        }
    }

    // Roll the CT clock forward up to 5 activations or until `actor`'s next turn
    for _ in 0..<5 {
        statesVisited += 1
        while !simUnits.contains(where: { $0.alive && $0.ct >= 100 }) {
            for i in 0..<simUnits.count where simUnits[i].alive {
                simUnits[i].ct += simUnits[i].speed
            }
        }
        guard let nextIdx = simUnits.indices.filter({ simUnits[$0].alive }).max(by: {
            (simUnits[$0].ct, simUnits[$0].speed) < (simUnits[$1].ct, simUnits[$1].speed)
        }) else { break }

        if simUnits[nextIdx].uid == actor.uid {
            break
        }
        simUnits[nextIdx].ct = max(0, simUnits[nextIdx].ct - 70)
        if simUnits[nextIdx].team != actor.team {
            let e = simUnits[nextIdx]
            let dist = manhattan(e.x, e.y, simUnits[aIdx].x, simUnits[aIdx].y)
            let reach = e.moveRange + ((e.role == "Knight") ? 1 : ((e.role == "Archer") ? 4 : 3))
            if dist <= reach {
                let dmg = (e.role == "Knight") ? 34 : ((e.role == "Archer") ? 26 : 30)
                simUnits[aIdx].hp -= dmg
                if simUnits[aIdx].hp <= 0 {
                    return (false, statesVisited)
                }
            }
        }
    }
    return (simUnits[aIdx].hp > 0, statesVisited)
}

func generatePostureManeuvers(actor: TUnit, units: [TUnit], elev: [[Int]]) -> [String: TManeuver] {
    let allies = units.filter { $0.alive && $0.team == actor.team }
    let enemies = units.filter { $0.alive && $0.team != actor.team }
    let tiles = reachableTiles(unit: actor, units: units, elev: elev)
    var allCandidates: [TManeuver] = []

    for (tx, ty) in tiles {
        let hHere = elev[ty][tx]
        let exposure = countEnemyThreats(x: tx, y: ty, enemies: enemies)
        let moveCt = (tx == actor.x && ty == actor.y) ? 0 : 20

        for e in enemies {
            let dist = manhattan(tx, ty, e.x, e.y)
            let angle = attackAngle(attackerX: tx, attackerY: ty, target: e)
            let angleHit: Float = (angle == "FRONT") ? 0.65 : ((angle == "SIDE") ? 0.85 : 0.98)
            let angleMult: Float = (angle == "FRONT") ? 1.0 : ((angle == "SIDE") ? 1.10 : 1.30)
            let heightBonus: Float = (hHere > elev[e.y][e.x]) ? 0.20 : 0.0
            let face = bestFacingToward(tx, ty, e.x, e.y)
            let eShort = e.name.split(separator: " ").first.map(String.init) ?? e.uid

            if actor.role == "Knight" && dist == 1 {
                let rawDmg = Int(38.0 * angleMult * (1.0 + 0.5 * heightBonus))
                let triggersCounter = (e.role == "Knight" && angle == "FRONT" && rawDmg < e.hp)
                let isKill = rawDmg >= e.hp
                let posture = (isKill || angle == "BACK") ? "LETHAL" : "FLANK"
                let score = Float(rawDmg) * angleHit + (isKill ? 45.0 : 0.0) - (triggersCounter ? 24.0 : 0.0)
                allCandidates.append(TManeuver(posture: posture, destX: tx, destY: ty, newFacing: face, ability: "Slash", targetUid: e.uid, expectedDmg: Float(rawDmg), expectedHeal: 0, hitProb: angleHit, angle: angle, triggersCounter: triggersCounter, friendlyFireDmg: 0, exposureCount: exposure, ctCost: moveCt + 30, score: score, summary: "Slash \(eShort) (\(angle), \(rawDmg) dmg)"))
            } else if actor.role == "Archer" && (2...4).contains(dist) {
                let hitP = min(0.99, angleHit + ((hHere > elev[e.y][e.x]) ? 0.10 : 0.0))
                let rawDmg = Int(30.0 * angleMult * (1.0 + heightBonus))
                let isKill = rawDmg >= e.hp
                let posture = isKill ? "LETHAL" : "FLANK"
                let score = Float(rawDmg) * hitP + (isKill ? 45.0 : 0.0) + 8.0 * Float(hHere) - 10.0 * Float(exposure)
                allCandidates.append(TManeuver(posture: posture, destX: tx, destY: ty, newFacing: face, ability: "AimBow", targetUid: e.uid, expectedDmg: Float(rawDmg), expectedHeal: 0, hitProb: hitP, angle: angle, triggersCounter: false, friendlyFireDmg: 0, exposureCount: exposure, ctCost: moveCt + 30, score: score, summary: "AimBow \(eShort) (h=\(hHere), \(rawDmg) dmg)"))
            } else if actor.role == "Mage" && (1...3).contains(dist) && actor.mp >= 12 {
                let aoeEnemies = enemies.filter { manhattan($0.x, $0.y, e.x, e.y) <= 1 }
                let aoeAllies = allies.filter { al in
                    let (ax, ay) = (al.uid == actor.uid) ? (tx, ty) : (al.x, al.y)
                    return manhattan(ax, ay, e.x, e.y) <= 1
                }
                let totalDmg = 34 * aoeEnemies.count
                let ffDmg = 34 * aoeAllies.count
                let isKill = aoeEnemies.contains { 34 >= $0.hp }
                let posture = isKill ? "LETHAL" : "FLANK"
                let score = Float(totalDmg) + (isKill ? 45.0 : 0.0) - 65.0 * Float(aoeAllies.count) - 12.0 * Float(exposure)
                allCandidates.append(TManeuver(posture: posture, destX: tx, destY: ty, newFacing: face, ability: "Fireball", targetUid: e.uid, expectedDmg: Float(totalDmg), expectedHeal: 0, hitProb: 0.95, angle: angle, triggersCounter: false, friendlyFireDmg: ffDmg, exposureCount: exposure, ctCost: moveCt + 30, score: score, summary: "Fireball \(eShort) (\(aoeEnemies.count) hit, FF=\(ffDmg))"))
            }
        }

        if actor.role == "Mage" && actor.mp >= 10 {
            for al in allies {
                let missing = al.maxHp - al.hp
                let (ax, ay) = (al.uid == actor.uid) ? (tx, ty) : (al.x, al.y)
                if missing >= 25 && manhattan(tx, ty, ax, ay) <= 3 {
                    let healAmt = min(missing, 45)
                    let urgent = al.hp <= Int(0.45 * Float(al.maxHp))
                    let score = Float(healAmt) * (urgent ? 1.8 : 1.1) - 8.0 * Float(exposure)
                    let aShort = al.name.split(separator: " ").first.map(String.init) ?? al.uid
                    allCandidates.append(TManeuver(posture: "SUPPORT", destX: tx, destY: ty, newFacing: bestFacingTowardNearestEnemy(tx, ty, enemies), ability: "Cure", targetUid: al.uid, expectedDmg: 0, expectedHeal: Float(healAmt), hitProb: 1.0, angle: "NONE", triggersCounter: false, friendlyFireDmg: 0, exposureCount: exposure, ctCost: moveCt + 25, score: score, summary: "Cure \(aShort) (+\(healAmt) HP)"))
                }
            }
        }

        let nearestDist = enemies.map { manhattan(tx, ty, $0.x, $0.y) }.min() ?? 4
        let idealDist = (actor.role == "Knight") ? 1 : 3
        let distScore = -Float(abs(nearestDist - idealDist)) * 6.0
        let kiteScore = 18.0 + 6.0 * Float(hHere) + distScore - 9.0 * Float(exposure)
        allCandidates.append(TManeuver(posture: "KITE", destX: tx, destY: ty, newFacing: bestFacingTowardNearestEnemy(tx, ty, enemies), ability: "Wait", targetUid: nil, expectedDmg: 0, expectedHeal: 0, hitProb: 1.0, angle: "NONE", triggersCounter: false, friendlyFireDmg: 0, exposureCount: exposure, ctCost: moveCt, score: kiteScore, summary: "Move (\(tx),\(ty)) h=\(hHere) & Wait (+30 CT)"))
    }

    var byPosture: [String: TManeuver] = [:]
    for pName in TACTICS_POSTURES {
        let matching = allCandidates.filter { $0.posture == pName }
        if let best = matching.max(by: { $0.score < $1.score }) {
            byPosture[pName] = best
        }
    }
    let fallback = byPosture["KITE"] ?? allCandidates.max(by: { $0.score < $1.score })!
    for pName in TACTICS_POSTURES where byPosture[pName] == nil {
        var cp = fallback
        cp.posture = pName
        cp.score -= 25.0
        byPosture[pName] = cp
    }
    return byPosture
}

// MARK: - S3 Native Tactics Policy

struct TacticsDecision {
    let probabilities: [String: Float]
    let proposed: String
    let executed: String
    let maneuver: TManeuver
    let intervened: Bool
    let shieldReason: String
    let winExchange: Float
    let trapRisk: Float
    let actProb: Float
    let bulletTimeS2: Bool
    let s2StatesEvaluated: Int
    let inferenceMs: Double
}

final class S3TacticsPolicy {
    let engine: S3ANEEngine
    let auxHead: S3AuxHead
    var guarded: Bool
    var threePass: Bool

    let choiceIns = "Choose the best tactical squad maneuver."
    let winIns = "Is there an advantageous trade?"
    let trapIns = "Is there a counter or trap risk?"
    let noulOpts = [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]

    init(engine: S3ANEEngine, guarded: Bool = true, threePass: Bool = false) throws {
        self.engine = engine
        self.guarded = guarded
        self.threePass = threePass

        var samples: [(state: String, options: [String])] = []
        let keys = ["lethal", "flank", "support", "kite"]
        for good in ["yes", "no"] {
            for trap in ["yes", "no"] {
                let st = "Advantageous trade: \(good). Counter or trap risk: \(trap)."
                for prefIdx in 0..<4 {
                    var opts: [String] = []
                    for (i, k) in keys.enumerated() {
                        if i == prefIdx {
                            opts.append("\(k): Safe. Best tactical value move.")
                        } else if trap == "yes" && i == 0 {
                            opts.append("\(k): Unsafe. Friendly fire or fatal counter trap.")
                        } else if i == 2 {
                            opts.append("\(k): Blocked. Out of range.")
                        } else {
                            opts.append("\(k): Suboptimal tactical trade.")
                        }
                    }
                    samples.append((st, opts))
                }
            }
        }
        self.auxHead = try engine.fitOrLoadAuxHead(
            name: "tactics",
            choiceInstructions: choiceIns,
            noul0Instructions: winIns,
            noul1Instructions: trapIns,
            calibrationSamples: samples
        )
    }

    func decide(actor: TUnit, units: [TUnit], elev: [[Int]]) throws -> TacticsDecision {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let options = generatePostureManeuvers(actor: actor, units: units, elev: elev)

        let pref: String
        if !guarded {
            pref = options.values.max(by: {
                ($0.expectedDmg + 0.1 * $0.score + ($0.triggersCounter ? 20.0 : 0.0) + ($0.friendlyFireDmg > 0 ? 30.0 : 0.0)) <
                ($1.expectedDmg + 0.1 * $1.score + ($1.triggersCounter ? 20.0 : 0.0) + ($1.friendlyFireDmg > 0 ? 30.0 : 0.0))
            })!.posture
        } else {
            pref = TACTICS_POSTURES.max(by: { options[$0]!.score < options[$1]!.score })!
        }

        let bestM = options[pref]!
        let hasGoodTrade = bestM.score >= 22.0
        let hasTrap = options.values.contains { $0.triggersCounter || $0.friendlyFireDmg > 0 || $0.exposureCount >= 2 }

        let stateStr = "Advantageous trade: \(hasGoodTrade ? "yes" : "no"). Counter or trap risk: \(hasTrap ? "yes" : "no")."
        let stateIds = engine.tokenizer.encode(stateStr)

        let keyPairs = [("lethal", "LETHAL"), ("flank", "FLANK"), ("support", "SUPPORT"), ("kite", "KITE")]
        var optStrings: [String] = []
        for (kLow, kUp) in keyPairs {
            let m = options[kUp]!
            let desc: String
            if guarded && (m.friendlyFireDmg > 0 || (m.triggersCounter && actor.hp <= 45)) {
                desc = "Unsafe. Friendly fire or fatal counter trap."
            } else if kUp == pref {
                desc = "Safe. Best tactical value move."
            } else if m.expectedDmg == 0 && m.expectedHeal == 0 && kUp != "KITE" {
                desc = "Blocked. Out of range."
            } else {
                desc = "Suboptimal tactical trade."
            }
            optStrings.append("\(kLow): \(desc)")
        }

        let (cIds, cMarkers) = engine.buildSequence(type: "choice", instructions: choiceIns, options: optStrings, stateIds: stateIds)
        let (cProbs, pooled, actProb) = try engine.forwardPass(ids: cIds, markers: cMarkers, qtypeId: 0)

        let winExchange: Float
        let trapRisk: Float
        if threePass {
            let (wIds, wM) = engine.buildSequence(type: "noul", instructions: winIns, options: noulOpts, stateIds: stateIds)
            let (wP, _, _) = try engine.forwardPass(ids: wIds, markers: wM, qtypeId: 2)
            let (tIds, tM) = engine.buildSequence(type: "noul", instructions: trapIns, options: noulOpts, stateIds: stateIds)
            let (tP, _, _) = try engine.forwardPass(ids: tIds, markers: tM, qtypeId: 2)
            winExchange = wP[1]
            trapRisk = tP[1]
        } else {
            let (w, t) = auxHead.predict(pooled: pooled)
            winExchange = w
            trapRisk = t
        }

        var probsDict: [String: Float] = [:]
        var proposed = "KITE"
        var bestP: Float = -1.0
        for (i, (_, kUp)) in keyPairs.enumerated() {
            probsDict[kUp] = cProbs[i]
            if cProbs[i] > bestP {
                bestP = cProbs[i]
                proposed = kUp
            }
        }

        var executed = proposed
        var chosenM = options[executed]!
        var shieldReason = "Direct ANE -> \(chosenM.summary)"
        var bulletTime = false
        var s2States = 0

        if guarded {
            var safePostures = TACTICS_POSTURES.filter { p in
                let m = options[p]!
                return m.friendlyFireDmg == 0 && !(m.triggersCounter && actor.hp <= 60)
            }
            if safePostures.isEmpty { safePostures = ["KITE"] }

            if chosenM.exposureCount >= 2 || actor.hp <= 65 || actProb < 0.60 {
                bulletTime = true
                var surviving: [String] = []
                for p in safePostures {
                    let (ok, visited) = system2DeepCTTimelineLookahead(actor: actor, cand: options[p]!, units: units)
                    s2States += visited
                    if ok { surviving.append(p) }
                }
                if !surviving.isEmpty { safePostures = surviving }
            }

            let bestSafe = safePostures.max(by: { options[$0]!.score < options[$1]!.score })!
            if !safePostures.contains(executed) || options[bestSafe]!.score > chosenM.score + 8.0 {
                executed = bestSafe
                chosenM = options[executed]!
                shieldReason = "Tactical Shield -> \(chosenM.summary)"
            }
        }

        let dtMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000.0
        return TacticsDecision(
            probabilities: probsDict,
            proposed: proposed,
            executed: executed,
            maneuver: chosenM,
            intervened: executed != proposed,
            shieldReason: shieldReason,
            winExchange: winExchange,
            trapRisk: trapRisk,
            actProb: actProb,
            bulletTimeS2: bulletTime,
            s2StatesEvaluated: s2States,
            inferenceMs: dtMs
        )
    }
}

func applyManeuver(actorIdx: Int, m: TManeuver, units: inout [TUnit], rng: inout S3SplitMix64, backstabs: inout Int, counters: inout Int, ffHits: inout Int) -> String {
    units[actorIdx].x = m.destX
    units[actorIdx].y = m.destY
    units[actorIdx].facing = m.newFacing
    units[actorIdx].ct = max(0, units[actorIdx].ct - (50 + m.ctCost))
    let aShort = units[actorIdx].name.split(separator: " ").first.map(String.init) ?? units[actorIdx].uid
    let aTeam = units[actorIdx].team

    guard m.ability != "Wait", let tUid = m.targetUid, let tIdx = units.firstIndex(where: { $0.uid == tUid && $0.alive }) else {
        return "\(aShort) repositions to (\(m.destX),\(m.destY)) facing \(m.newFacing) & waits (+CT)"
    }
    let tShort = units[tIdx].name.split(separator: " ").first.map(String.init) ?? units[tIdx].uid

    if m.ability == "Cure" {
        units[actorIdx].mp = max(0, units[actorIdx].mp - 10)
        let heal = Int(m.expectedHeal)
        units[tIdx].hp = min(units[tIdx].maxHp, units[tIdx].hp + heal)
        return "\(aShort) casts Cure on \(tShort) (+\(heal) HP)"
    }
    if m.ability == "Fireball" {
        units[actorIdx].mp = max(0, units[actorIdx].mp - 12)
        let tx = units[tIdx].x
        let ty = units[tIdx].y
        var hits: [String] = []
        for i in 0..<units.count where units[i].alive {
            if manhattan(units[i].x, units[i].y, tx, ty) <= 1 {
                units[i].hp = max(0, units[i].hp - 34)
                if units[i].team == aTeam { ffHits += 1 }
                let uShort = units[i].name.split(separator: " ").first.map(String.init) ?? units[i].uid
                if units[i].hp == 0 {
                    units[i].alive = false
                    hits.append("\(uShort) KO!")
                } else {
                    hits.append("\(uShort)-34")
                }
            }
        }
        return "\(aShort) casts AoE Fireball -> \(hits.joined(separator: ", "))"
    }
    if rng.nextFloat() <= m.hitProb {
        let dmg = Int(m.expectedDmg)
        units[tIdx].hp = max(0, units[tIdx].hp - dmg)
        var note = "\(aShort) \(m.ability) (\(m.angle)) -> \(tShort) -\(dmg) HP"
        if m.angle == "BACK" { backstabs += 1 }
        if units[tIdx].hp == 0 {
            units[tIdx].alive = false
            note += " (KO!)"
        } else if m.triggersCounter && units[tIdx].alive {
            units[actorIdx].hp = max(0, units[actorIdx].hp - 22)
            counters += 1
            note += " | Countered -22!"
            if units[actorIdx].hp == 0 { units[actorIdx].alive = false }
        }
        return note
    }
    return "\(aShort) \(m.ability) (\(m.angle)) EVADED by \(tShort)!"
}

func playBattle(seed: Int, bluePolicy: S3TacticsPolicy, redPolicy: S3TacticsPolicy, maxTurns: Int = 36) throws -> (winner: String, blueSurvivors: Int, backstabs: Int, counters: Int, ffHits: Int, bulletTimes: Int, s2States: Int, latencies: [Double]) {
    var rng = S3SplitMix64(seed: seed)
    let (elev, initUnits) = buildBattlefield(seed: seed)
    var units = initUnits
    var backstabs = 0
    var counters = 0
    var ffHits = 0
    var bulletTimes = 0
    var s2States = 0
    var latencies: [Double] = []

    for _ in 0..<maxTurns {
        let blueAlive = units.contains { $0.alive && $0.team == "BLUE" }
        let redAlive  = units.contains { $0.alive && $0.team == "RED" }
        if !blueAlive || !redAlive { break }

        while !units.contains(where: { $0.alive && $0.ct >= 100 }) {
            for i in 0..<units.count where units[i].alive {
                units[i].ct += units[i].speed
            }
        }
        guard let activeIdx = units.indices.filter({ units[$0].alive }).max(by: {
            (units[$0].ct, units[$0].speed) < (units[$1].ct, units[$1].speed)
        }) else { break }

        let active = units[activeIdx]
        let pol = (active.team == "BLUE") ? bluePolicy : redPolicy
        let dec = try pol.decide(actor: active, units: units, elev: elev)

        var stepBS = 0, stepCT = 0, stepFF = 0
        _ = applyManeuver(actorIdx: activeIdx, m: dec.maneuver, units: &units, rng: &rng, backstabs: &stepBS, counters: &stepCT, ffHits: &stepFF)
        if active.team == "BLUE" {
            latencies.append(dec.inferenceMs)
            if dec.bulletTimeS2 { bulletTimes += 1; s2States += dec.s2StatesEvaluated }
            backstabs += stepBS
            counters += stepCT
            ffHits += stepFF
        }
    }
    let blueHp = units.filter { $0.alive && $0.team == "BLUE" }.reduce(0) { $0 + $1.hp }
    let redHp  = units.filter { $0.alive && $0.team == "RED" }.reduce(0) { $0 + $1.hp }
    let blueSurv = units.filter { $0.alive && $0.team == "BLUE" }.count
    return (blueHp > redHp ? "BLUE" : "RED", blueSurv, backstabs, counters, ffHits, bulletTimes, s2States, latencies)
}

// MARK: - Benchmark CLI

@main
struct S3TacticsApp {
    static func main() throws {
        let args = Array(CommandLine.arguments.dropFirst())
        var battles = 16
        var threePass = false
        var jsonOutPath: String? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--battles":
                if i + 1 < args.count { battles = Int(args[i + 1]) ?? 16; i += 1 }
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
        let tacticalPolicy = try S3TacticsPolicy(engine: engine, guarded: true, threePass: threePass)
        let naivePolicy    = try S3TacticsPolicy(engine: engine, guarded: false, threePass: threePass)
        let bootMs = Double(DispatchTime.now().uptimeNanoseconds - tBoot0) / 1_000_000.0

        var shieldWins = 0, rawWins = 0
        var shieldSurvivors = 0, rawSurvivors = 0
        var shieldBS = 0, rawBS = 0
        var shieldCT = 0, rawCT = 0
        var shieldFF = 0, rawFF = 0
        var bulletTimes = 0, s2StatesTotal = 0
        var latencies: [Double] = []

        let tCampaign0 = DispatchTime.now().uptimeNanoseconds
        for b in 1...battles {
            let seed = 500 + b * 23
            let resS = try playBattle(seed: seed, bluePolicy: tacticalPolicy, redPolicy: naivePolicy)
            let resR = try playBattle(seed: seed, bluePolicy: naivePolicy, redPolicy: naivePolicy)

            if resS.winner == "BLUE" { shieldWins += 1 }
            if resR.winner == "BLUE" { rawWins += 1 }
            shieldSurvivors += resS.blueSurvivors
            rawSurvivors += resR.blueSurvivors
            shieldBS += resS.backstabs; rawBS += resR.backstabs
            shieldCT += resS.counters;  rawCT += resR.counters
            shieldFF += resS.ffHits;    rawFF += resR.ffHits
            bulletTimes += resS.bulletTimes
            s2StatesTotal += resS.s2States
            latencies.append(contentsOf: resS.latencies)
        }
        let campaignMs = Double(DispatchTime.now().uptimeNanoseconds - tCampaign0) / 1_000_000.0
        let sortedLat = latencies.sorted()
        let p50Lat = sortedLat.isEmpty ? 0.0 : sortedLat[sortedLat.count / 2]
        let meanLat = sortedLat.isEmpty ? 0.0 : sortedLat.reduce(0.0, +) / Double(sortedLat.count)

        let summary: [String: Any] = [
            "arena": "s3-tactics",
            "mode": threePass ? "swift_3pass" : "swift_s3_1pass",
            "battles": battles,
            "boot_and_aux_fit_ms": (bootMs * 100.0).rounded() / 100.0,
            "total_campaign_ms": (campaignMs * 100.0).rounded() / 100.0,
            "decisions_total": latencies.count,
            "p50_turn_ms": (p50Lat * 100.0).rounded() / 100.0,
            "mean_turn_ms": (meanLat * 100.0).rounded() / 100.0,
            "shield_wins": shieldWins,
            "raw_wins": rawWins,
            "shield_win_rate_pct": Double(shieldWins) * 100.0 / Double(battles),
            "raw_win_rate_pct": Double(rawWins) * 100.0 / Double(battles),
            "shield_survivors": shieldSurvivors,
            "raw_survivors": rawSurvivors,
            "shield_friendly_fire": shieldFF,
            "raw_friendly_fire": rawFF,
            "shield_counters_taken": shieldCT,
            "raw_counters_taken": rawCT,
            "shield_backstabs": shieldBS,
            "raw_backstabs": rawBS,
            "bullet_time_triggers": bulletTimes,
            "s2_ct_timeline_states_evaluated": s2StatesTotal,
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted])
        if let path = jsonOutPath {
            try jsonData.write(to: URL(fileURLWithPath: path))
        }
        print(String(data: jsonData, encoding: .utf8)!)
    }
}
