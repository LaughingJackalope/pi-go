//
// `s3_snake.swift` — Full Native Swift Port of `laya-coreml-snake`
// (Game Engine + Hamiltonian Cycle Safety Shield + Truecolor 24-Bit ANSI Terminal UI
//  + Zero-Copy `mmap` BPE Tokenizer + CoreML `.mlmodelc` M2 Neural Engine Policy
//  + S3 Single-Pass `[CLS]` Multi-Head Readout)
//
// Compile:
//   swiftc -O s3_snake.swift -o s3-snake -framework Foundation -framework CoreML -framework Accelerate
//
// Modes:
//   ./s3-snake                           # Interactive Truecolor TUI (S3 1-Pass Tactical Mode)
//   ./s3-snake --three-pass              # Interactive Truecolor TUI (Legacy 3-Pass Parity Mode)
//   ./s3-snake --max-speed --steps 300   # Uncapped live TUI run
//   ./s3-snake benchmark                 # Full benchmark (3-Pass Parity vs S3 1-Pass across 10..120 FPS)
//

import Accelerate
import CoreML
import Darwin
import Foundation

// MARK: - Zero-Copy Memory-Mapped File & BPE Tokenizer

final class MMapFile {
    let path: String
    let size: Int
    let rawPtr: UnsafeRawPointer

    init(path: String) throws {
        self.path = path
        let fd = Darwin.open(path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "S3Snake", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot open \(path)"])
        }
        defer { Darwin.close(fd) }
        var st = Darwin.stat()
        guard Darwin.fstat(fd, &st) == 0 else {
            throw NSError(domain: "S3Snake", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot stat \(path)"])
        }
        self.size = Int(st.st_size)
        let mapped = Darwin.mmap(nil, self.size, PROT_READ, MAP_SHARED, fd, 0)
        guard mapped != MAP_FAILED, let validPtr = mapped else {
            throw NSError(domain: "S3Snake", code: 3, userInfo: [NSLocalizedDescriptionKey: "mmap failed for \(path)"])
        }
        self.rawPtr = UnsafeRawPointer(validPtr)
    }

    deinit {
        Darwin.munmap(UnsafeMutableRawPointer(mutating: rawPtr), size)
    }
}

struct MergeSlot {
    let key: UInt64
    let rank: UInt32
    let mergedId: UInt32
}

final class S3Tokenizer {
    private let mmapFile: MMapFile
    private let charCap: Int
    private let charMask: UInt32
    private let mergeCap: Int
    private let mergeMask: UInt64
    private let byteFallbackPtr: UnsafePointer<UInt32>
    private let charTablePtr: UnsafePointer<UInt32>
    private let mergeTablePtr: UnsafePointer<MergeSlot>
    private var tokenCache: [String: [Int]] = [:]

    let padId: Int = 0
    let sepId: Int = 1
    let clsId: Int = 2
    let unkId: Int = 3
    let maskId: Int = 4

    init(mmapPath: String) throws {
        self.mmapFile = try MMapFile(path: mmapPath)
        let base = mmapFile.rawPtr
        let headerU32 = base.assumingMemoryBound(to: UInt32.self)
        self.charCap = Int(headerU32[2])
        self.charMask = UInt32(charCap - 1)
        self.mergeCap = Int(headerU32[3])
        self.mergeMask = UInt64(mergeCap - 1)

        var offset = 32
        self.byteFallbackPtr = base.advanced(by: offset).assumingMemoryBound(to: UInt32.self)
        offset += 256 * 4
        self.charTablePtr = base.advanced(by: offset).assumingMemoryBound(to: UInt32.self)
        offset += charCap * 8
        self.mergeTablePtr = base.advanced(by: offset).assumingMemoryBound(to: MergeSlot.self)
    }

    @inline(__always)
    private func lookupChar(_ cp: UInt32) -> Int? {
        let key = cp &+ 1
        var h = (key &* 2654435761) & charMask
        while true {
            let slotKey = charTablePtr[Int(h) &* 2]
            if slotKey == 0 { return nil }
            if slotKey == key { return Int(charTablePtr[Int(h) &* 2 &+ 1]) }
            h = (h &+ 1) & charMask
        }
    }

    @inline(__always)
    private func lookupMerge(_ leftId: Int, _ rightId: Int) -> (rank: UInt32, mergedId: Int)? {
        let key = (UInt64(UInt32(leftId)) << 32) | UInt64(UInt32(rightId))
        var h = ((key &* 11400714819323198485) >> 43) & mergeMask
        while true {
            let slot = mergeTablePtr[Int(h)]
            if slot.key == 0 { return nil }
            if slot.key == key { return (slot.rank, Int(slot.mergedId)) }
            h = (h &+ 1) & mergeMask
        }
    }

    private func bpeWord(_ piece: String) -> [Int] {
        if piece.isEmpty { return [] }
        var syms: [Int] = []
        syms.reserveCapacity(piece.utf8.count)
        for scalar in piece.unicodeScalars {
            if let tid = lookupChar(scalar.value) {
                syms.append(tid)
            } else {
                for b in String(scalar).utf8 {
                    syms.append(Int(byteFallbackPtr[Int(b)]))
                }
            }
        }
        while syms.count >= 2 {
            var bestRank: UInt32 = UInt32.max
            var bestIdx: Int = -1
            var bestMerged: Int = -1
            for i in 0..<(syms.count - 1) {
                if let info = lookupMerge(syms[i], syms[i + 1]), info.rank < bestRank {
                    bestRank = info.rank
                    bestIdx = i
                    bestMerged = info.mergedId
                }
            }
            if bestIdx < 0 { break }
            syms[bestIdx] = bestMerged
            syms.remove(at: bestIdx + 1)
        }
        return syms
    }

    func encode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        if let cached = tokenCache[text] { return cached }
        var norm = text.replacingOccurrences(of: " ", with: "▁")
        if !norm.hasPrefix("▁") { norm = "▁" + norm }
        var out: [Int] = []
        var cur = ""
        for (idx, ch) in norm.enumerated() {
            if idx > 0 && ch == "▁" {
                out.append(contentsOf: bpeWord(cur))
                cur = "▁"
            } else {
                cur.append(ch)
            }
        }
        if !cur.isEmpty { out.append(contentsOf: bpeWord(cur)) }
        if tokenCache.count < 4096 {
            tokenCache[text] = out
        }
        return out
    }
}

// MARK: - Deterministic Snake Game & Hamiltonian Cycle Safety Planner (`game.py` exact port)

let DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

@inline(__always)
func dirVector(_ dir: String) -> (Int, Int) {
    switch dir {
    case "UP":    return (0, -1)
    case "DOWN":  return (0, 1)
    case "LEFT":  return (-1, 0)
    default:      return (1, 0)
    }
}

struct Cell: Hashable, Equatable {
    let x: Int
    let y: Int
}

struct MoveInfo {
    let direction: String
    let legal: Bool
    let safe: Bool
    let advance: Int
    let reason: String
    let eats: Bool
}

struct SplitMix64 {
    var state: UInt64
    init(seed: Int) {
        self.state = UInt64(bitPattern: Int64(seed)) &+ 0x9E3779B97F4A7C15
    }
    mutating func next() -> UInt64 {
        var z = state
        state = state &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func choiceIndex(count: Int) -> Int {
        return Int(next() % UInt64(count))
    }
}

func buildHamiltonianCycle(width: Int, height: Int) -> [Cell] {
    if height % 2 != 0 {
        return buildHamiltonianCycle(width: height, height: width).map { Cell(x: $0.y, y: $0.x) }
    }
    var path: [Cell] = []
    path.reserveCapacity(width * height)
    path.append(Cell(x: 0, y: 0))
    for y in 0..<height {
        if y % 2 == 0 {
            for x in 1..<width { path.append(Cell(x: x, y: y)) }
        } else {
            for x in stride(from: width - 1, through: 1, by: -1) { path.append(Cell(x: x, y: y)) }
        }
    }
    for y in stride(from: height - 1, through: 1, by: -1) {
        path.append(Cell(x: 0, y: y))
    }
    return path
}

final class SnakeGame {
    let width: Int
    let height: Int
    let seed: Int
    let initialLength: Int
    let capacity: Int
    let cycle: [Cell]
    let cycleIndexGrid: [Int] // y * width + x -> cycle index

    var rng: SplitMix64
    var body: [Cell] // index 0 is head, last is tail
    var occupiedGrid: [Bool]
    var score: Int = 0
    var ticks: Int = 0
    var alive: Bool = true
    var won: Bool = false
    var deathReason: String? = nil
    var food: Cell? = nil

    init(width: Int = 24, height: Int = 16, seed: Int = 7, initialLength: Int = 6) {
        self.width = width
        self.height = height
        self.seed = seed
        self.initialLength = initialLength
        self.capacity = width * height
        self.cycle = buildHamiltonianCycle(width: width, height: height)
        var grid = [Int](repeating: 0, count: width * height)
        for (idx, c) in cycle.enumerated() {
            grid[c.y * width + c.x] = idx
        }
        self.cycleIndexGrid = grid
        self.rng = SplitMix64(seed: seed)

        let startIdx = grid[(height / 2) * width + (width / 2)]
        var initBody: [Cell] = []
        initBody.reserveCapacity(capacity)
        var occ = [Bool](repeating: false, count: width * height)
        for i in 0..<initialLength {
            let cIdx = (startIdx - i + capacity) % capacity
            let cell = cycle[cIdx]
            initBody.append(cell)
            occ[cell.y * width + cell.x] = true
        }
        self.body = initBody
        self.occupiedGrid = occ
        self.food = spawnFood()
    }

    var head: Cell { body[0] }

    @inline(__always)
    func cycleIndex(_ c: Cell) -> Int {
        return cycleIndexGrid[c.y * width + c.x]
    }

    private func spawnFood() -> Cell? {
        var empty: [Cell] = []
        empty.reserveCapacity(capacity - body.count)
        for cell in cycle {
            if !occupiedGrid[cell.y * width + cell.x] {
                empty.append(cell)
            }
        }
        if empty.isEmpty { return nil }
        return empty[rng.choiceIndex(count: empty.count)]
    }

    @inline(__always)
    func target(_ dir: String) -> Cell {
        let (dx, dy) = dirVector(dir)
        return Cell(x: head.x + dx, y: head.y + dy)
    }

    func legalReason(_ dir: String) -> String {
        let cell = target(dir)
        if cell.x < 0 || cell.x >= width || cell.y < 0 || cell.y >= height {
            return "wall"
        }
        if body.count > 1 && cell == body[1] {
            return "reverse"
        }
        let tail = body[body.count - 1]
        if cell != food && cell == tail {
            return "legal"
        }
        return occupiedGrid[cell.y * width + cell.x] ? "body" : "legal"
    }

    func moves() -> [MoveInfo] {
        if !alive || won { return [] }
        let headIdx = cycleIndex(head)
        let tailIdx = cycleIndex(body[body.count - 1])
        let tailDist = (tailIdx - headIdx + capacity) % capacity
        let foodDist: Int
        if let f = food {
            foodDist = (cycleIndex(f) - headIdx + capacity) % capacity
        } else {
            foodDist = capacity
        }

        var result: [MoveInfo] = []
        result.reserveCapacity(4)
        for dir in DIRECTIONS {
            var reason = legalReason(dir)
            let legal = (reason == "legal")
            let t = target(dir)
            let tIdx: Int
            if t.x >= 0 && t.x < width && t.y >= 0 && t.y < height {
                tIdx = cycleIndex(t)
            } else {
                tIdx = headIdx
            }
            let advance = (tIdx - headIdx + capacity) % capacity
            let eats = (t == food)
            var safe = legal
            if safe && (advance > tailDist || (advance == tailDist && eats)) {
                safe = false
                reason = "would cross the tail"
            }
            if safe && (advance == 0 || advance > foodDist) {
                safe = false
                reason = "would skip the food on the safe route"
            }
            result.append(MoveInfo(direction: dir, legal: legal, safe: safe, advance: advance, reason: reason, eats: eats))
        }
        return result
    }

    func foodReachability() -> (reachable: Bool, space: Int) {
        var visited = [Bool](repeating: false, count: capacity)
        let hFlat = head.y * width + head.x
        visited[hFlat] = true
        var queue = [Int](repeating: 0, count: capacity)
        var qHead = 0
        var qTail = 1
        queue[0] = hFlat

        while qHead < qTail {
            let cur = queue[qHead]
            qHead += 1
            let cx = cur % width
            let cy = cur / width

            // UP, DOWN, LEFT, RIGHT
            let neighbors = [
                (cx, cy - 1),
                (cx, cy + 1),
                (cx - 1, cy),
                (cx + 1, cy),
            ]
            for (nx, ny) in neighbors {
                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    let nFlat = ny * width + nx
                    if !visited[nFlat] && !occupiedGrid[nFlat] {
                        visited[nFlat] = true
                        queue[qTail] = nFlat
                        qTail += 1
                    }
                }
            }
        }
        let reach: Bool
        if let f = food {
            reach = visited[f.y * width + f.x]
        } else {
            reach = false
        }
        return (reach, qTail)
    }

    @discardableResult
    func step(_ dir: String) -> Bool {
        ticks += 1
        let reason = legalReason(dir)
        if reason != "legal" {
            alive = false
            deathReason = reason
            return false
        }
        let t = target(dir)
        if t == food {
            body.insert(t, at: 0)
            occupiedGrid[t.y * width + t.x] = true
            score += 1
            if body.count == capacity {
                won = true
                food = nil
            } else {
                food = spawnFood()
            }
            return true
        } else {
            let oldTail = body.removeLast()
            occupiedGrid[oldTail.y * width + oldTail.x] = false
            body.insert(t, at: 0)
            occupiedGrid[t.y * width + t.x] = true
            return false
        }
    }

    func cycleOrderValid() -> Bool {
        if body.count < 2 { return true }
        var total = 0
        for i in stride(from: body.count - 1, through: 1, by: -1) {
            let a = cycleIndex(body[i])
            let b = cycleIndex(body[i - 1])
            let d = (b - a + capacity) % capacity
            if d <= 0 { return false }
            total += d
        }
        return total < capacity
    }
}

// MARK: - Truecolor 24-Bit ANSI Terminal UI Composer (`ui.py` exact port)

struct RGB: Equatable {
    let r: UInt8
    let g: UInt8
    let b: UInt8
}

let COLOR_BG    = RGB(r: 0x09, g: 0x0F, b: 0x13)
let COLOR_FG    = RGB(r: 0xE3, g: 0xF3, b: 0xEF)
let COLOR_MUTED = RGB(r: 0x68, g: 0x86, b: 0x8C)
let COLOR_DIM   = RGB(r: 0x20, g: 0x35, b: 0x3C)
let COLOR_DOT   = RGB(r: 0x13, g: 0x27, b: 0x2E)
let COLOR_GREEN = RGB(r: 0x62, g: 0xF5, b: 0xB5)
let COLOR_AMBER = RGB(r: 0xFF, g: 0xCE, b: 0x73)
let COLOR_RED   = RGB(r: 0xFF, g: 0x7C, b: 0x8C)
let COLOR_CYAN  = RGB(r: 0x8A, g: 0xD8, b: 0xE9)
let COLOR_HEAD  = RGB(r: 0xDC, g: 0xFF, b: 0xF0)

let DIGITS_GLYPHS: [[String]] = [
    ["█▀█", "█ █", "▀▀▀"], // 0
    ["▄█ ", " █ ", "▀▀▀"], // 1
    ["▀▀█", "█▀▀", "▀▀▀"], // 2
    ["▀▀█", "▀▀█", "▀▀▀"], // 3
    ["█ █", "▀▀█", "  ▀"], // 4
    ["█▀▀", "▀▀█", "▀▀▀"], // 5
    ["█▀▀", "█▀█", "▀▀▀"], // 6
    ["▀▀█", "  █", "  ▀"], // 7
    ["█▀█", "█▀█", "▀▀▀"], // 8
    ["█▀█", "▀▀█", "▀▀▀"], // 9
]

final class SnakeCanvas {
    let width: Int
    let height: Int
    var chars: [Character]
    var colors: [RGB]

    init(gameWidth: Int, gameHeight: Int) {
        self.width = max(104, gameWidth * 2 + 50)
        self.height = max(35, gameHeight + 19)
        self.chars = [Character](repeating: " ", count: width * height)
        self.colors = [RGB](repeating: COLOR_FG, count: width * height)
    }

    func clear() {
        for i in 0..<chars.count {
            chars[i] = " "
            colors[i] = COLOR_FG
        }
    }

    @inline(__always)
    func put(_ row: Int, _ col: Int, _ text: String, _ color: RGB = COLOR_FG) {
        if row < 0 || row >= height { return }
        let rowBase = row * width
        var x = col
        for ch in text {
            if x >= 0 && x < width {
                chars[rowBase + x] = ch
                colors[rowBase + x] = color
            }
            x += 1
        }
    }

    func bar(_ row: Int, _ col: Int, _ value: Float, _ length: Int = 20, _ color: RGB = COLOR_GREEN) {
        let clamped = max(0.0, min(1.0, value))
        let count = Int((clamped * Float(length)).rounded())
        put(row, col, String(repeating: "━", count: length), COLOR_DIM)
        if count > 0 {
            put(row, col, String(repeating: "━", count: count), color)
        }
    }

    func number(_ row: Int, _ col: Int, _ value: Int, _ color: RGB = COLOR_GREEN) {
        let clamped = max(0, min(999, value))
        let d0 = (clamped / 100) % 10
        let d1 = (clamped / 10) % 10
        let d2 = clamped % 10
        for (idx, d) in [d0, d1, d2].enumerated() {
            for (lineIdx, glyph) in DIGITS_GLYPHS[d].enumerated() {
                put(row + lineIdx, col + idx * 4, glyph, color)
            }
        }
    }

    func renderANSI(game: SnakeGame, decision: SnakeDecision, stats: SnakeStats) -> String {
        clear()
        let left = 3
        let right = max(58, game.width * 2 + 10)
        let top = 6
        let side = width - right - 4
        let bottom = top + game.height + 1

        let stateStr = stats.paused ? "PAUSED" : (game.won ? "BOARD CLEAR" : (!game.alive ? "GAME OVER" : "LIVE"))
        put(1, left, "LAYA  /  LOCAL INTELLIGENCE (S3 SWIFT NATIVE)", COLOR_MUTED)
        put(1, width - stateStr.count - 3, stateStr, game.alive ? COLOR_GREEN : COLOR_RED)
        put(2, left, String(repeating: "─", count: width - 6), COLOR_DIM)
        put(4, left, "S N A K E", COLOR_FG)
        put(4, left + 31, String(format: "ROUND %02d", stats.round), COLOR_MUTED)
        put(top, left, "┌" + String(repeating: "─", count: game.width * 2) + "┐", COLOR_DIM)
        put(bottom, left, "└" + String(repeating: "─", count: game.width * 2) + "┘", COLOR_DIM)

        let dotRow = String(repeating: "· ", count: game.width)
        for y in 0..<game.height {
            put(top + 1 + y, left, "│", COLOR_DIM)
            put(top + 1 + y, left + game.width * 2 + 1, "│", COLOR_DIM)
            put(top + 1 + y, left + 1, dotRow, COLOR_DOT)
        }

        let bLen = max(1, game.body.count)
        for index in stride(from: game.body.count - 1, through: 0, by: -1) {
            let cell = game.body[index]
            let color: RGB
            if index == 0 {
                color = COLOR_HEAD
            } else {
                let frac = 1.0 - Float(index) / Float(bLen)
                color = RGB(
                    r: UInt8(18 + Int(64.0 * frac)),
                    g: UInt8(73 + Int(150.0 * frac)),
                    b: UInt8(57 + Int(102.0 * frac))
                )
            }
            put(top + cell.y + 1, left + 1 + 2 * cell.x, "██", color)
        }
        if let f = game.food {
            put(top + f.y + 1, left + 1 + 2 * f.x, "● ", COLOR_AMBER)
        }

        put(bottom + 2, left + 0, "SCORE", COLOR_MUTED)
        number(bottom + 3, left + 0, game.score, COLOR_GREEN)
        put(bottom + 2, left + 18, "LENGTH", COLOR_MUTED)
        number(bottom + 3, left + 18, game.body.count, COLOR_FG)
        put(bottom + 2, left + 36, "BEST", COLOR_MUTED)
        number(bottom + 3, left + 36, stats.best, COLOR_MUTED)

        let fillFrac = Float(game.body.count) / Float(game.capacity)
        bar(bottom + 7, left, fillFrac, 41, COLOR_GREEN)
        put(bottom + 7, left + 43, String(format: "%4.1f%%", fillFrac * 100.0), COLOR_MUTED)

        put(4, right, "S3 CoreML Swift", COLOR_GREEN)
        put(5, right, "\(stats.hardware) · Local ANE", COLOR_MUTED)
        put(7, right, "NEXT MOVE", COLOR_FG)
        put(7, right + 15, "MODEL PROBABILITIES", COLOR_MUTED)

        for (idx, dir) in DIRECTIONS.enumerated() {
            let row = 9 + idx
            let prob = decision.probabilities[idx]
            let selected = (dir == decision.proposed)
            let color = selected ? COLOR_GREEN : COLOR_MUTED
            let prefix = selected ? "›" : " "
            let padded = dir.padding(toLength: 5, withPad: " ", startingAt: 0)
            put(row, right, "\(prefix) \(padded)", color)
            let count = Int((prob * 18.0).rounded())
            put(row, right + 9, String(repeating: "░", count: 18), COLOR_DIM)
            if count > 0 {
                put(row, right + 9, String(repeating: "█", count: count), color)
            }
            put(row, right + 29, String(format: "%.2f", prob), color)
        }

        put(14, right, "EXECUTING", COLOR_MUTED)
        put(14, right + 12, decision.executed, COLOR_GREEN)
        if decision.intervened {
            put(14, right + 20, "SHIELD", COLOR_AMBER)
        }
        put(16, right, "DEAD-END RISK", COLOR_MUTED)
        let riskColor = (decision.deadEndRisk < 0.5) ? COLOR_AMBER : COLOR_RED
        bar(17, right, decision.deadEndRisk, min(24, side - 9), riskColor)
        put(17, right + 29, String(format: "%.2f", decision.deadEndRisk), riskColor)

        put(19, right, "FOOD REACHABLE", COLOR_MUTED)
        bar(20, right, decision.foodReachable, min(24, side - 9), COLOR_CYAN)
        put(20, right + 29, String(format: "%.2f", decision.foodReachable), COLOR_CYAN)

        put(22, right, "INFERENCE", COLOR_MUTED)
        put(22, right + 18, String(format: "%5.1f ms", decision.inferenceMs), COLOR_FG)
        put(23, right, "DECISIONS", COLOR_MUTED)
        put(23, right + 18, String(format: "%5.1f /s", stats.stepsPerSecond), COLOR_FG)
        put(24, right, "OUTPUT TOKENS", COLOR_MUTED)
        put(24, right + 18, "0", COLOR_FG)
        put(25, right, "NETWORK", COLOR_MUTED)
        put(25, right + 18, "OFFLINE", COLOR_GREEN)
        put(26, right, "ENGINE", COLOR_MUTED)
        put(26, right + 18, stats.engine, COLOR_MUTED)
        put(28, right, stats.guarded ? "Laya + cycle safety" : "Laya · shield OFF", COLOR_MUTED)
        put(29, right, String(format: "Shield interventions  %04d", stats.interventions), COLOR_AMBER)

        put(height - 3, left, String(repeating: "─", count: width - 6), COLOR_DIM)
        put(height - 2, left, "SPACE pause   ↑/↓ speed   R reset   Q quit", COLOR_MUTED)
        let mins = Int(stats.elapsed) / 60
        let secs = Int(stats.elapsed) % 60
        put(height - 2, right, String(format: "ESTIMATES BY LAYA            %02d:%02d", mins, secs), COLOR_MUTED)

        // Serialize to 24-bit Truecolor ANSI string with style run-length encoding
        var out = "\u{1B}[H\u{1B}[48;2;\(COLOR_BG.r);\(COLOR_BG.g);\(COLOR_BG.b)m"
        out.reserveCapacity(width * height * 2)
        for r in 0..<height {
            let rBase = r * width
            var curColor = colors[rBase]
            out.append("\u{1B}[38;2;\(curColor.r);\(curColor.g);\(curColor.b)m")
            for c in 0..<width {
                let col = colors[rBase + c]
                if col != curColor {
                    curColor = col
                    out.append("\u{1B}[38;2;\(curColor.r);\(curColor.g);\(curColor.b)m")
                }
                out.append(chars[rBase + c])
            }
            if r + 1 != height {
                out.append("\n")
            }
        }
        out.append("\u{1B}[0m")
        return out
    }
}

// MARK: - S3 Native Snake Policy Engine (Supports 3-Pass Parity & S3 1-Pass Tactical Mode)

struct SnakeDecision {
    let probabilities: [Float] // [UP, DOWN, LEFT, RIGHT]
    let proposed: String
    let executed: String
    let safeDirections: [String]
    let intervened: Bool
    let deadEndRisk: Float
    let foodReachable: Float
    let inferenceMs: Double
    let decisionMs: Double
    let inputTokens: Int
}

struct SnakeStats {
    var hardware: String = "M2"
    var engine: String = "CPU+ANE · FP16"
    var guarded: Bool = true
    var interventions: Int = 0
    var best: Int = 0
    var round: Int = 1
    var paused: Bool = false
    var elapsed: Double = 0
    var stepsPerSecond: Double = 0
}

final class S3SnakePolicy {
    let bundleDir: URL
    let tokenizer: S3Tokenizer
    let tokEmbMMap: MMapFile
    let typeEmbMMap: MMapFile
    let auxHeadMMap: MMapFile
    let model: MLModel

    let maxLen: Int = 96
    let maxOptions: Int = 32
    let hiddenSize: Int = 768
    let localWindowHalf: Int = 64

    var guarded: Bool
    var threePass: Bool

    let embeddingsArr: MLMultiArray
    let fullMaskArr: MLMultiArray
    let localMaskArr: MLMultiArray
    let typeVectorsArr: MLMultiArray
    let markerMapArr: MLMultiArray

    // Pointers to snake_aux_head.fp32.bin ([W_risk (768), W_food (768), b (2)])
    let auxWRisk: UnsafePointer<Float>
    let auxWFood: UnsafePointer<Float>
    let auxBias: UnsafePointer<Float>

    // Pre-tokenized static prefixes for `risk` and `food` (`noul` questions in compact mode)
    let riskPrefixIds: [Int]
    let riskMarkers: [Int]
    let foodPrefixIds: [Int]
    let foodMarkers: [Int]
    let moveHeadPrefix: [Int]

    init(bundlePath: String, guarded: Bool = true, threePass: Bool = false) throws {
        self.bundleDir = URL(fileURLWithPath: bundlePath)
        self.guarded = guarded
        self.threePass = threePass
        self.tokenizer = try S3Tokenizer(mmapPath: bundleDir.appendingPathComponent("tokenizer_mmap.bin").path)
        self.tokEmbMMap = try MMapFile(path: bundleDir.appendingPathComponent("tok_embeddings.fp16.bin").path)
        self.typeEmbMMap = try MMapFile(path: bundleDir.appendingPathComponent("type_emb.fp16.bin").path)
        self.auxHeadMMap = try MMapFile(path: bundleDir.appendingPathComponent("snake_aux_head.fp32.bin").path)

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: bundleDir.appendingPathComponent("model.mlmodelc"), configuration: mlConfig)

        self.embeddingsArr = try MLMultiArray(shape: [1, 768, 1, 96], dataType: .float16)
        self.fullMaskArr = try MLMultiArray(shape: [1, 96, 1, 96], dataType: .float16)
        self.localMaskArr = try MLMultiArray(shape: [1, 96, 1, 96], dataType: .float16)
        self.typeVectorsArr = try MLMultiArray(shape: [1, 768, 1, 1], dataType: .float16)
        self.markerMapArr = try MLMultiArray(shape: [1, 96, 1, 32], dataType: .float16)

        let auxBase = auxHeadMMap.rawPtr.assumingMemoryBound(to: Float.self)
        self.auxWRisk = auxBase
        self.auxWFood = auxBase.advanced(by: 768)
        self.auxBias  = auxBase.advanced(by: 1536)

        // Pre-build static `move` instruction prefix and `risk` / `food` noul question heads
        self.moveHeadPrefix = [tokenizer.clsId] + tokenizer.encode("choice question: Choose the best safe move toward food.") + [tokenizer.sepId]

        let noulFalseOpt = [tokenizer.maskId] + tokenizer.encode(" false: no, the statement does not hold")
        let noulTrueOpt  = [tokenizer.maskId] + tokenizer.encode(" true: yes, the statement holds")

        var rIds = [tokenizer.clsId] + tokenizer.encode("noul question: Is a safe route available?") + [tokenizer.sepId]
        let rM0 = rIds.count
        rIds.append(contentsOf: noulFalseOpt)
        let rM1 = rIds.count
        rIds.append(contentsOf: noulTrueOpt)
        rIds.append(tokenizer.sepId)
        self.riskPrefixIds = rIds
        self.riskMarkers = [rM0, rM1]

        var fIds = [tokenizer.clsId] + tokenizer.encode("noul question: Is food reachable through empty cells?") + [tokenizer.sepId]
        let fM0 = fIds.count
        fIds.append(contentsOf: noulFalseOpt)
        let fM1 = fIds.count
        fIds.append(contentsOf: noulTrueOpt)
        fIds.append(tokenizer.sepId)
        self.foodPrefixIds = fIds
        self.foodMarkers = [fM0, fM1]
    }

    func forwardPass(ids: [Int], markers: [Int], qtypeId: Int) throws -> (probs: [Float], pooled: [Float]) {
        let validCount = min(ids.count, maxLen)
        let tokEmbBase = tokEmbMMap.rawPtr.assumingMemoryBound(to: Float16.self)
        let typeEmbBase = typeEmbMMap.rawPtr.assumingMemoryBound(to: Float16.self)

        let embPtr = embeddingsArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let embStrides = embeddingsArr.strides.map { $0.intValue }
        let eStrideC = embStrides[1]
        let eStrideS = embStrides[3]
        for s in 0..<maxLen {
            let tid = (s < validCount) ? ids[s] : tokenizer.padId
            let srcRow = tokEmbBase.advanced(by: tid * hiddenSize)
            for c in 0..<hiddenSize {
                embPtr[c * eStrideC + s * eStrideS] = srcRow[c]
            }
        }

        let fullPtr = fullMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let localPtr = localMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let maskStrides = fullMaskArr.strides.map { $0.intValue }
        let mStrideK = maskStrides[1]
        let mStrideQ = maskStrides[3]
        let zero16: Float16 = 0.0
        let negInf16: Float16 = -10000.0

        for k in 0..<maxLen {
            let validK = k < validCount
            let rowKOffset = k * mStrideK
            if !validK {
                for q in 0..<maxLen {
                    let idx = rowKOffset + q * mStrideQ
                    fullPtr[idx] = negInf16
                    localPtr[idx] = negInf16
                }
            } else {
                for q in 0..<maxLen {
                    let idx = rowKOffset + q * mStrideQ
                    fullPtr[idx] = zero16
                    let validQ = q < validCount
                    let inWin = abs(q - k) <= localWindowHalf
                    localPtr[idx] = (inWin || !validQ) ? zero16 : negInf16
                }
            }
        }

        let typePtr = typeVectorsArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let tStrideC = typeVectorsArr.strides[1].intValue
        let srcType = typeEmbBase.advanced(by: qtypeId * hiddenSize)
        for c in 0..<hiddenSize {
            typePtr[c * tStrideC] = srcType[c]
        }

        let markerPtr = markerMapArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let mmStrides = markerMapArr.strides.map { $0.intValue }
        let mmStrideS = mmStrides[1]
        let mmStrideOpt = mmStrides[3]
        for s in 0..<maxLen {
            let rowS = s * mmStrideS
            for opt in 0..<maxOptions {
                markerPtr[rowS + opt * mmStrideOpt] = zero16
            }
        }
        let one16: Float16 = 1.0
        for opt in 0..<maxOptions {
            let pos = (opt < markers.count) ? markers[opt] : 0
            markerPtr[pos * mmStrideS + opt * mmStrideOpt] = one16
        }

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "embeddings": MLFeatureValue(multiArray: embeddingsArr),
            "full_mask": MLFeatureValue(multiArray: fullMaskArr),
            "local_mask": MLFeatureValue(multiArray: localMaskArr),
            "type_vectors": MLFeatureValue(multiArray: typeVectorsArr),
            "marker_map": MLFeatureValue(multiArray: markerMapArr),
        ])

        let outProvider = try model.prediction(from: inputFeatures)
        let k = markers.count
        var rawLogits = [Float](repeating: -10000.0, count: k)
        var pooled = [Float](repeating: 0.0, count: hiddenSize)

        for name in outProvider.featureNames {
            guard let arr = outProvider.featureValue(for: name)?.multiArrayValue else { continue }
            let shape = arr.shape.map { $0.intValue }
            let strides = arr.strides.map { $0.intValue }
            if shape.count >= 4 && shape[1] == 1 {
                let sOpt = strides[3]
                if arr.dataType == .float16 {
                    let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                    for i in 0..<k { rawLogits[i] = Float(ptr[i * sOpt]) }
                } else {
                    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
                    for i in 0..<k { rawLogits[i] = ptr[i * sOpt] }
                }
            } else {
                let sC = strides[1]
                if arr.dataType == .float16 {
                    let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                    for c in 0..<hiddenSize { pooled[c] = Float(ptr[c * sC]) }
                } else {
                    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
                    for c in 0..<hiddenSize { pooled[c] = ptr[c * sC] }
                }
            }
        }

        var maxL: Float = -Float.infinity
        for v in rawLogits { if v > maxL { maxL = v } }
        var probs = [Float](repeating: 0.0, count: k)
        var sumP: Float = 0.0
        for i in 0..<k {
            let e = expf(rawLogits[i] - maxL)
            probs[i] = e
            sumP += e
        }
        for i in 0..<k { probs[i] /= sumP }
        return (probs, pooled)
    }

    func decide(game: SnakeGame) throws -> SnakeDecision {
        let tStart = DispatchTime.now().uptimeNanoseconds
        let moves = game.moves()
        let safeMoves = moves.filter { $0.safe }
        if safeMoves.isEmpty && guarded {
            throw NSError(domain: "S3Snake", code: 100, userInfo: [NSLocalizedDescriptionKey: "Cycle safety invariant violated: no safe action"])
        }
        let preferred = safeMoves.max(by: { $0.advance < $1.advance })?.direction ?? "NONE"
        let (reachable, _) = game.foodReachability()

        let stateStr = "Safe route: \(!safeMoves.isEmpty ? "yes" : "no"). Food reachable through empty cells: \(reachable ? "yes" : "no")."
        let stateIds = tokenizer.encode(stateStr)

        // Build `move` sequence in exact (UP, DOWN, LEFT, RIGHT) slot order matching `policy.py`
        var moveIds = moveHeadPrefix
        moveIds.reserveCapacity(maxLen)
        var moveMarkers: [Int] = []
        moveMarkers.reserveCapacity(4)

        for m in moves {
            let desc: String
            if !m.legal {
                desc = "Blocked. Collision."
            } else if !m.safe {
                desc = "Unsafe. Traps the snake."
            } else if m.eats {
                desc = "Safe. Eat food now. Best."
            } else if m.direction == preferred {
                desc = "Safe. Best route to food."
            } else {
                desc = "Safe. Slower route."
            }
            moveMarkers.append(moveIds.count)
            moveIds.append(tokenizer.maskId)
            moveIds.append(contentsOf: tokenizer.encode(" \(m.direction): \(desc)"))
        }
        moveIds.append(tokenizer.sepId)
        let roomMove = max(0, maxLen - moveIds.count - 1)
        moveIds.append(contentsOf: stateIds.prefix(roomMove))
        moveIds.append(tokenizer.sepId)

        let tInf0 = DispatchTime.now().uptimeNanoseconds
        let (moveProbs, movePooled) = try forwardPass(ids: moveIds, markers: moveMarkers, qtypeId: 0)

        let pRiskSafe: Float
        let pFoodReach: Float
        var totalTokens = moveIds.count

        if threePass {
            var rIds = riskPrefixIds
            let rRoom = max(0, maxLen - rIds.count - 1)
            rIds.append(contentsOf: stateIds.prefix(rRoom))
            rIds.append(tokenizer.sepId)
            let (rProbs, _) = try forwardPass(ids: rIds, markers: riskMarkers, qtypeId: 2)
            pRiskSafe = rProbs[1]
            totalTokens += rIds.count

            var fIds = foodPrefixIds
            let fRoom = max(0, maxLen - fIds.count - 1)
            fIds.append(contentsOf: stateIds.prefix(fRoom))
            fIds.append(tokenizer.sepId)
            let (fProbs, _) = try forwardPass(ids: fIds, markers: foodMarkers, qtypeId: 2)
            pFoodReach = fProbs[1]
            totalTokens += fIds.count
        } else {
            // S3 Single-Pass Tactical Mode: Read `risk` and `food` directly from `movePooled` via vDSP
            var logitRisk: Float = 0.0
            var logitFood: Float = 0.0
            vDSP_dotpr(movePooled, 1, auxWRisk, 1, &logitRisk, vDSP_Length(768))
            vDSP_dotpr(movePooled, 1, auxWFood, 1, &logitFood, vDSP_Length(768))
            logitRisk += auxBias[0]
            logitFood += auxBias[1]
            pRiskSafe = 1.0 / (1.0 + expf(-max(-30.0, min(30.0, logitRisk))))
            pFoodReach = 1.0 / (1.0 + expf(-max(-30.0, min(30.0, logitFood))))
        }

        let infMs = Double(DispatchTime.now().uptimeNanoseconds - tInf0) / 1_000_000.0

        var bestPropIdx = 0
        for i in 1..<4 {
            if moveProbs[i] > moveProbs[bestPropIdx] { bestPropIdx = i }
        }
        let proposed = DIRECTIONS[bestPropIdx]
        let allowed = safeMoves.map { $0.direction }

        let executed: String
        if guarded && !allowed.contains(proposed) {
            var bestAllowed = allowed[0]
            var bestAllowedProb: Float = -1.0
            for (idx, dir) in DIRECTIONS.enumerated() {
                if allowed.contains(dir) && moveProbs[idx] > bestAllowedProb {
                    bestAllowedProb = moveProbs[idx]
                    bestAllowed = dir
                }
            }
            executed = bestAllowed
        } else {
            executed = proposed
        }

        let decMs = Double(DispatchTime.now().uptimeNanoseconds - tStart) / 1_000_000.0
        return SnakeDecision(
            probabilities: moveProbs,
            proposed: proposed,
            executed: executed,
            safeDirections: allowed,
            intervened: proposed != executed,
            deadEndRisk: 1.0 - pRiskSafe,
            foodReachable: pFoodReach,
            inferenceMs: infMs,
            decisionMs: decMs,
            inputTokens: totalTokens
        )
    }
}

// MARK: - Episode Runner & Benchmark Harness (`benchmark.py` port)

func percentiles(_ vals: [Double]) -> [String: Double] {
    if vals.isEmpty { return ["mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0] }
    let s = vals.sorted()
    let n = s.count
    let mean = s.reduce(0.0, +) / Double(n)
    let p50 = s[n / 2]
    let p95 = s[min(n - 1, Int(Double(n) * 0.95))]
    let p99 = s[min(n - 1, Int(Double(n) * 0.99))]
    return [
        "mean_ms": (mean * 100.0).rounded() / 100.0,
        "p50_ms": (p50 * 100.0).rounded() / 100.0,
        "p95_ms": (p95 * 100.0).rounded() / 100.0,
        "p99_ms": (p99 * 100.0).rounded() / 100.0,
        "max_ms": (s[n - 1] * 100.0).rounded() / 100.0,
    ]
}

func runBenchmarkEpisode(policy: S3SnakePolicy, seed: Int, steps: Int, width: Int = 24, height: Int = 16, fps: Double? = nil, render: Bool = true) throws -> [String: Any] {
    let game = SnakeGame(width: width, height: height, seed: seed, initialLength: 6)
    let canvas = SnakeCanvas(gameWidth: width, gameHeight: height)
    var stats = SnakeStats(
        hardware: "M2",
        engine: policy.threePass ? "Swift 3-Pass ANE · FP16" : "Swift S3 1-Pass ANE · FP16",
        guarded: policy.guarded,
        interventions: 0,
        best: 0,
        round: 1,
        paused: false,
        elapsed: 0,
        stepsPerSecond: fps ?? 0
    )

    var infTimes: [Double] = []
    var renderTimes: [Double] = []
    var activeTickTimes: [Double] = []
    var deadlineMisses = 0
    var sinkBytes = 0

    let tEpisodeStart = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<steps {
        let tTick0 = DispatchTime.now().uptimeNanoseconds
        let decision = try policy.decide(game: game)
        if decision.intervened { stats.interventions += 1 }

        let tRen0 = DispatchTime.now().uptimeNanoseconds
        if render {
            stats.elapsed = Double(tTick0 - tEpisodeStart) / 1_000_000_000.0
            let frameStr = canvas.renderANSI(game: game, decision: decision, stats: stats)
            sinkBytes &+= frameStr.utf8.count
        }
        let renMs = Double(DispatchTime.now().uptimeNanoseconds - tRen0) / 1_000_000.0

        game.step(decision.executed)
        let activeMs = Double(DispatchTime.now().uptimeNanoseconds - tTick0) / 1_000_000.0

        infTimes.append(decision.inferenceMs)
        renderTimes.append(renMs)
        activeTickTimes.append(activeMs)

        if let targetFps = fps {
            let budgetMs = 1000.0 / targetFps
            if activeMs > budgetMs {
                deadlineMisses += 1
            }
            let remSec = (budgetMs - activeMs) / 1000.0
            if remSec > 0 {
                Thread.sleep(forTimeInterval: remSec)
            }
        }
        if !game.alive || game.won { break }
    }

    let elapsedSec = Double(DispatchTime.now().uptimeNanoseconds - tEpisodeStart) / 1_000_000_000.0
    let doneSteps = activeTickTimes.count
    let achievedFps = Double(doneSteps) / max(elapsedSec, 1e-6)
    let missFrac = Double(deadlineMisses) / Double(max(doneSteps, 1))
    let passes = game.alive && missFrac <= 0.01

    return [
        "mode": policy.threePass ? "swift_3pass" : "swift_s3_1pass",
        "seed": seed,
        "target_fps": fps as Any,
        "steps": doneSteps,
        "seconds": (elapsedSec * 1000.0).rounded() / 1000.0,
        "achieved_steps_per_second": (achievedFps * 10.0).rounded() / 10.0,
        "score": game.score,
        "length": game.body.count,
        "alive": game.alive,
        "interventions": stats.interventions,
        "inference": percentiles(infTimes),
        "render": percentiles(renderTimes),
        "active_tick": percentiles(activeTickTimes),
        "deadline_misses": deadlineMisses,
        "miss_fraction": (missFrac * 10000.0).rounded() / 10000.0,
        "passes": passes,
        "ansi_bytes_rendered": sinkBytes,
    ]
}

// MARK: - Raw Terminal Keyboard Helper for Interactive Play

final class RawTerminal {
    var saved = termios()
    var active = false

    init() {
        if isatty(STDIN_FILENO) != 0 {
            tcgetattr(STDIN_FILENO, &saved)
            var raw = saved
            raw.c_lflag &= ~tcflag_t(ICANON | ECHO)
            raw.c_cc.16 = 0 // VMIN
            raw.c_cc.17 = 0 // VTIME
            tcsetattr(STDIN_FILENO, TCSANOW, &raw)
            active = true
        }
    }

    func readKey() -> String {
        guard active else { return "" }
        var buf = [UInt8](repeating: 0, count: 32)
        let n = Darwin.read(STDIN_FILENO, &buf, buf.count)
        if n > 0 {
            return String(bytes: buf[0..<n], encoding: .utf8) ?? ""
        }
        return ""
    }

    func restore() {
        if active {
            tcsetattr(STDIN_FILENO, TCSADRAIN, &saved)
            active = false
        }
    }

    deinit { restore() }
}

// MARK: - CLI Main

let bundlePath = "/Users/mp/decision/models/snake/swift_bundle"
let args = Array(CommandLine.arguments.dropFirst())

if args.first == "benchmark" {
    var sweepSteps = 120
    var rates: [Double] = [10, 20, 30, 40, 50, 60, 90, 120]
    var i = 1
    while i < args.count {
        if args[i] == "--sweep-steps", i + 1 < args.count {
            sweepSteps = Int(args[i + 1]) ?? 120
            i += 1
        } else if args[i] == "--rates", i + 1 < args.count {
            rates = args[i + 1].split(separator: ",").compactMap { Double($0) }
            i += 1
        }
        i += 1
    }

    let tBoot0 = DispatchTime.now().uptimeNanoseconds
    let policy3 = try S3SnakePolicy(bundlePath: bundlePath, guarded: true, threePass: true)
    let warmGame = SnakeGame(width: 24, height: 16, seed: 9001, initialLength: 6)
    for _ in 0..<6 {
        let d = try policy3.decide(game: warmGame)
        warmGame.step(d.executed)
    }
    let bootMs = Double(DispatchTime.now().uptimeNanoseconds - tBoot0) / 1_000_000.0

    // 1. Uncapped 3-Pass Parity Mode
    let uncapped3 = try runBenchmarkEpisode(policy: policy3, seed: 7, steps: sweepSteps, fps: nil, render: true)

    // 2. Paced rates for 3-Pass Parity Mode
    var sweep3: [[String: Any]] = []
    var fastest3: Double = 0
    for r in rates {
        let ep = try runBenchmarkEpisode(policy: policy3, seed: 7, steps: min(sweepSteps, 60), fps: r, render: true)
        sweep3.append(ep)
        if (ep["passes"] as? Bool) == true { fastest3 = r }
    }

    // 3. Uncapped S3 1-Pass Tactical Mode
    policy3.threePass = false
    let uncapped1 = try runBenchmarkEpisode(policy: policy3, seed: 7, steps: sweepSteps, fps: nil, render: true)

    // 4. Paced rates for S3 1-Pass Tactical Mode
    var sweep1: [[String: Any]] = []
    var fastest1: Double = 0
    for r in rates {
        let ep = try runBenchmarkEpisode(policy: policy3, seed: 7, steps: min(sweepSteps, 60), fps: r, render: true)
        sweep1.append(ep)
        if (ep["passes"] as? Bool) == true { fastest1 = r }
    }

    let out: [String: Any] = [
        "boot_and_warmup_ms": (bootMs * 100.0).rounded() / 100.0,
        "swift_3pass_parity": [
            "uncapped": uncapped3,
            "sweep": sweep3,
            "fastest_tested_stable_fps": fastest3,
        ],
        "swift_s3_1pass_tactical": [
            "uncapped": uncapped1,
            "sweep": sweep1,
            "fastest_tested_stable_fps": fastest1,
        ],
    ]
    let jsonData = try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted])
    print(String(data: jsonData, encoding: .utf8)!)
    exit(0)
}

// Interactive or Headless Play Mode
var threePass = false
var maxSpeed = false
var headless = false
var unassisted = false
var fps: Double = 12.0
var maxSteps: Int? = nil
var seed = 7

var idx = 0
while idx < args.count {
    switch args[idx] {
    case "--three-pass": threePass = true
    case "--max-speed":  maxSpeed = true
    case "--headless":   headless = true
    case "--unassisted": unassisted = true
    case "--fps":
        if idx + 1 < args.count { fps = Double(args[idx + 1]) ?? 12.0; idx += 1 }
    case "--steps":
        if idx + 1 < args.count { maxSteps = Int(args[idx + 1]); idx += 1 }
    case "--seed":
        if idx + 1 < args.count { seed = Int(args[idx + 1]) ?? 7; idx += 1 }
    default: break
    }
    idx += 1
}

let policy = try S3SnakePolicy(bundlePath: bundlePath, guarded: !unassisted, threePass: threePass)
var game = SnakeGame(width: 24, height: 16, seed: seed, initialLength: 6)
let canvas = SnakeCanvas(gameWidth: 24, gameHeight: 16)
var stats = SnakeStats(
    hardware: "M2",
    engine: threePass ? "Swift 3-Pass ANE · FP16" : "Swift S3 1-Pass ANE · FP16",
    guarded: !unassisted
)

// Warmup
let warm = SnakeGame(width: 24, height: 16, seed: seed + 10000, initialLength: 6)
for _ in 0..<6 {
    let d = try policy.decide(game: warm)
    warm.step(d.executed)
}

let term = headless ? nil : RawTerminal()
if !headless {
    print("\u{1B}[?1049h\u{1B}[?25l", terminator: "")
    fflush(stdout)
}
defer {
    term?.restore()
    if !headless {
        print("\u{1B}[?25h\u{1B}[?1049l", terminator: "")
        fflush(stdout)
    }
}

let tStart = DispatchTime.now().uptimeNanoseconds
var totalSteps = 0
var infSum = 0.0
var recentTimestamps: [Double] = []

while true {
    if let limit = maxSteps, totalSteps >= limit { break }
    let tNow = DispatchTime.now().uptimeNanoseconds
    if let k = term?.readKey().lowercased(), !k.isEmpty {
        if k.contains("q") || k.contains("\u{03}") { break }
        if k.contains(" ") { stats.paused.toggle() }
        if k.contains("+") || k.contains("\u{1B}[a") { fps = min(240.0, fps + 4.0) }
        if k.contains("-") || k.contains("\u{1B}[b") { fps = max(1.0, fps - 4.0) }
        if k.contains("r") {
            stats.round += 1
            game = SnakeGame(width: 24, height: 16, seed: seed + stats.round - 1, initialLength: 6)
        }
    }
    if stats.paused {
        Thread.sleep(forTimeInterval: 0.03)
        continue
    }

    let decision = try policy.decide(game: game)
    infSum += decision.inferenceMs
    if decision.intervened { stats.interventions += 1 }
    let nowSec = Double(DispatchTime.now().uptimeNanoseconds - tStart) / 1_000_000_000.0
    recentTimestamps.append(nowSec)
    if recentTimestamps.count > 60 { recentTimestamps.removeFirst() }
    stats.elapsed = nowSec
    if recentTimestamps.count > 1 {
        stats.stepsPerSecond = Double(recentTimestamps.count - 1) / max(1e-6, recentTimestamps.last! - recentTimestamps.first!)
    }
    stats.best = max(stats.best, game.score)

    if !headless {
        let ansi = canvas.renderANSI(game: game, decision: decision, stats: stats)
        FileHandle.standardOutput.write(Data(ansi.utf8))
    }

    if !maxSpeed {
        let tickElapsed = Double(DispatchTime.now().uptimeNanoseconds - tNow) / 1_000_000_000.0
        let rem = (1.0 / fps) - tickElapsed
        if rem > 0 { Thread.sleep(forTimeInterval: rem) }
    }

    game.step(decision.executed)
    totalSteps += 1
    stats.best = max(stats.best, game.score)
    if !game.alive || game.won {
        if unassisted { break }
        stats.round += 1
        game = SnakeGame(width: 24, height: 16, seed: seed + stats.round - 1, initialLength: 6)
    }
}

let totalSec = Double(DispatchTime.now().uptimeNanoseconds - tStart) / 1_000_000_000.0
let summary: [String: Any] = [
    "steps": totalSteps,
    "seconds": (totalSec * 1000.0).rounded() / 1000.0,
    "steps_per_second": (Double(totalSteps) / max(totalSec, 1e-6) * 10.0).rounded() / 10.0,
    "score": game.score,
    "best_score": stats.best,
    "interventions": stats.interventions,
    "engine": stats.engine,
    "mean_inference_ms": totalSteps > 0 ? ((infSum / Double(totalSteps)) * 100.0).rounded() / 100.0 : 0.0,
]
let sData = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted])
print(String(data: sData, encoding: .utf8)!)
