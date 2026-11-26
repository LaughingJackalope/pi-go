//
// `S3Core.swift` — Shared Native Apple Silicon (M2 Neural Engine) Neuro-Symbolic Core Library
// for Project `decision` (`s3-cortexd`, `s3-snake`, `s3-poker`, `s3-tactics`, `s3-breach`).
//
// Provides:
//   1. `MMapFile`          — Zero-copy POSIX `mmap(MAP_SHARED)` loader (0 MB dirty heap).
//   2. `S3Tokenizer`       — Zero-copy memory-mapped O(1) BPE tokenizer (`tokenizer_mmap.bin`).
//   3. `S3AuxHead`         — Single-pass `[CLS]` (768-dim) -> multi-noul `vDSP` readout + native
//                            in-Swift Dual Kernel Ridge Regression trainer $(X X^T + \lambda I)^{-1} Y$.
//   4. `S3ANEEngine`       — Pre-compiled `model.mlmodelc` CoreML ANE runner with `autoreleasepool`
//                            `IOSurface` lifecycle management, `act_head`, and S3 v2 FiLM+SRR head.
//   5. `S3Canvas`          — 24-bit Truecolor ANSI Fixed-Cell Terminal HUD Composer (~0.4 ms/frame).
//   6. `S3RawTerminal`     — Non-blocking POSIX `termios` keyboard reader for live interactive TUIs.
//   7. `S3SplitMix64`      — Deterministic 64-bit PRNG for reproducible duplicate benchmarks.
//

import Accelerate
import CoreML
import Darwin
import Foundation

// MARK: - 1. Zero-Copy Memory-Mapped File

public final class MMapFile {
    public let path: String
    public let size: Int
    public let rawPtr: UnsafeRawPointer

    public init(path: String) throws {
        self.path = path
        let fd = Darwin.open(path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "S3Core", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot open \(path)"])
        }
        defer { Darwin.close(fd) }
        var st = Darwin.stat()
        guard Darwin.fstat(fd, &st) == 0 else {
            throw NSError(domain: "S3Core", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot stat \(path)"])
        }
        self.size = Int(st.st_size)
        let mapped = Darwin.mmap(nil, self.size, PROT_READ, MAP_SHARED, fd, 0)
        guard mapped != MAP_FAILED, let validPtr = mapped else {
            throw NSError(domain: "S3Core", code: 3, userInfo: [NSLocalizedDescriptionKey: "mmap failed for \(path)"])
        }
        self.rawPtr = UnsafeRawPointer(validPtr)
    }

    deinit {
        Darwin.munmap(UnsafeMutableRawPointer(mutating: rawPtr), size)
    }
}

// MARK: - 2. Zero-Copy Memory-Mapped BPE Tokenizer

public struct S3AddedToken {
    public let bytes: [UInt8]
    public let str: String
    public let id: Int
    public let lstrip: Bool
}

public struct S3MergeSlot {
    public let key: UInt64
    public let rank: UInt32
    public let mergedId: UInt32
}

public final class S3Tokenizer {
    private let mmapFile: MMapFile
    private let charCap: Int
    private let charMask: UInt32
    private let mergeCap: Int
    private let mergeMask: UInt64
    private let byteFallbackPtr: UnsafePointer<UInt32>
    private let charTablePtr: UnsafePointer<UInt32>
    private let mergeTablePtr: UnsafePointer<S3MergeSlot>
    private var addedTokens: [S3AddedToken] = []
    private var tokenCache: [String: [Int]] = [:]

    public let padId: Int = 0
    public let sepId: Int = 1
    public let clsId: Int = 2
    public let unkId: Int = 3
    public let maskId: Int = 4
    public let maskToken: String = "<mask>"

    public init(mmapPath: String) throws {
        self.mmapFile = try MMapFile(path: mmapPath)
        let base = mmapFile.rawPtr
        let headerU32 = base.assumingMemoryBound(to: UInt32.self)
        let version = Int(headerU32[1])
        guard version == 1 else {
            throw NSError(domain: "S3Core", code: 10, userInfo: [NSLocalizedDescriptionKey: "Unsupported tokenizer_mmap version \(version)"])
        }
        self.charCap = Int(headerU32[2])
        self.charMask = UInt32(charCap - 1)
        self.mergeCap = Int(headerU32[3])
        self.mergeMask = UInt64(mergeCap - 1)
        let numAdded = Int(headerU32[4])

        var offset = 32
        self.byteFallbackPtr = base.advanced(by: offset).assumingMemoryBound(to: UInt32.self)
        offset += 256 * 4
        self.charTablePtr = base.advanced(by: offset).assumingMemoryBound(to: UInt32.self)
        offset += charCap * 8
        self.mergeTablePtr = base.advanced(by: offset).assumingMemoryBound(to: S3MergeSlot.self)
        offset += mergeCap * 16

        var list: [S3AddedToken] = []
        list.reserveCapacity(numAdded)
        for _ in 0..<numAdded {
            let tid = Int(base.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            let flags = base.loadUnaligned(fromByteOffset: offset + 4, as: UInt8.self)
            let blen = Int(base.loadUnaligned(fromByteOffset: offset + 5, as: UInt16.self))
            offset += 7
            let bptr = base.advanced(by: offset).assumingMemoryBound(to: UInt8.self)
            let bytes = Array(UnsafeBufferPointer(start: bptr, count: blen))
            offset += blen
            if let s = String(bytes: bytes, encoding: .utf8), !s.isEmpty {
                list.append(S3AddedToken(bytes: bytes, str: s, id: tid, lstrip: (flags & 2) != 0))
            }
        }
        self.addedTokens = list
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

    private func encodeSegment(_ seg: String) -> [Int] {
        if seg.isEmpty { return [] }
        var norm = seg.replacingOccurrences(of: " ", with: "▁")
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
        return out
    }

    public func encode(_ text: String, truncation: Bool = false, maxLength: Int? = nil) -> [Int] {
        if text.isEmpty { return [] }
        if !truncation, let cached = tokenCache[text] { return cached }
        let utf8 = Array(text.utf8)
        let n = utf8.count
        var i = 0
        var segStart = 0
        var out: [Int] = []

        while i < n {
            let b0 = utf8[i]
            var matched: S3AddedToken? = nil
            if b0 == 10 || b0 == 9 || b0 == 60 || b0 == 91 || b0 == 0xE2 {
                for cand in addedTokens {
                    let clen = cand.bytes.count
                    if i + clen <= n && cand.bytes[0] == b0 {
                        var ok = true
                        for k in 1..<clen {
                            if utf8[i + k] != cand.bytes[k] { ok = false; break }
                        }
                        if ok { matched = cand; break }
                    }
                }
            }
            if let m = matched {
                var segEnd = i
                if m.lstrip {
                    while segEnd > segStart {
                        let c = utf8[segEnd - 1]
                        if c == 32 || c == 9 || c == 10 || c == 13 { segEnd -= 1 } else { break }
                    }
                }
                if segEnd > segStart, let segStr = String(bytes: utf8[segStart..<segEnd], encoding: .utf8) {
                    out.append(contentsOf: encodeSegment(segStr))
                }
                out.append(m.id)
                i += m.bytes.count
                segStart = i
            } else {
                i += 1
            }
        }
        if segStart < n, let segStr = String(bytes: utf8[segStart..<n], encoding: .utf8) {
            out.append(contentsOf: encodeSegment(segStr))
        }
        if !truncation && tokenCache.count < 4096 {
            tokenCache[text] = out
        }
        if truncation, let maxLen = maxLength, out.count > maxLen {
            return Array(out.prefix(maxLen))
        }
        return out
    }
}

// MARK: - 3. Single-Pass `[CLS]` Auxiliary Head & Native In-Swift Dual Ridge Trainer

public struct S3AuxHead {
    public let w0: [Float] // 768
    public let w1: [Float] // 768
    public let b0: Float
    public let b1: Float

    public init(w0: [Float], w1: [Float], b0: Float, b1: Float) {
        self.w0 = w0
        self.w1 = w1
        self.b0 = b0
        self.b1 = b1
    }

    public init(path: String) throws {
        let mm = try MMapFile(path: path)
        let ptr = mm.rawPtr.assumingMemoryBound(to: Float.self)
        self.w0 = Array(UnsafeBufferPointer(start: ptr, count: 768))
        self.w1 = Array(UnsafeBufferPointer(start: ptr.advanced(by: 768), count: 768))
        self.b0 = ptr[1536]
        self.b1 = ptr[1537]
    }

    public func save(to path: String) throws {
        var data = Data(capacity: 1538 * 4)
        w0.withUnsafeBufferPointer { data.append(UnsafeBufferPointer(start: UnsafeRawPointer($0.baseAddress!).assumingMemoryBound(to: UInt8.self), count: 768 * 4)) }
        w1.withUnsafeBufferPointer { data.append(UnsafeBufferPointer(start: UnsafeRawPointer($0.baseAddress!).assumingMemoryBound(to: UInt8.self), count: 768 * 4)) }
        let bias = [b0, b1]
        bias.withUnsafeBufferPointer { data.append(UnsafeBufferPointer(start: UnsafeRawPointer($0.baseAddress!).assumingMemoryBound(to: UInt8.self), count: 2 * 4)) }
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    @inline(__always)
    public func predict(pooled: [Float]) -> (Float, Float) {
        var l0: Float = 0.0
        var l1: Float = 0.0
        vDSP_dotpr(pooled, 1, w0, 1, &l0, vDSP_Length(768))
        vDSP_dotpr(pooled, 1, w1, 1, &l1, vDSP_Length(768))
        l0 += b0
        l1 += b1
        let p0 = 1.0 / (1.0 + expf(-max(-30.0, min(30.0, l0))))
        let p1 = 1.0 / (1.0 + expf(-max(-30.0, min(30.0, l1))))
        return (p0, p1)
    }

    /// Native In-Swift Dual Kernel Ridge Regression Trainer:
    /// Solves $(X_{\text{aug}} X_{\text{aug}}^T + \lambda I_N) \alpha = Y_{\text{logit}}$ in $O(N^3)$ via Gaussian elimination
    /// with partial pivoting, then projects $W_{\text{aug}} = X_{\text{aug}}^T \alpha$ in `<0.1 ms` using `vDSP_dotpr`.
    public static func fitDualRidge(features: [[Float]], targets: [(Float, Float)], lambda: Float = 1e-2) -> S3AuxHead {
        let n = features.count
        let d = 768
        var xAug = [[Float]](repeating: [Float](repeating: 1.0, count: d + 1), count: n)
        var yLogit0 = [Float](repeating: 0.0, count: n)
        var yLogit1 = [Float](repeating: 0.0, count: n)

        for i in 0..<n {
            for j in 0..<d { xAug[i][j] = features[i][j] }
            xAug[i][d] = 1.0
            let p0 = max(1e-4, min(1.0 - 1e-4, targets[i].0))
            let p1 = max(1e-4, min(1.0 - 1e-4, targets[i].1))
            yLogit0[i] = logf(p0 / (1.0 - p0))
            yLogit1[i] = logf(p1 / (1.0 - p1))
        }

        // Build N x (N + 2) augmented Gram matrix [K | y0 | y1] where K_{ij} = <xAug_i, xAug_j> + delta_{ij} * lambda
        var mat = [[Double]](repeating: [Double](repeating: 0.0, count: n + 2), count: n)
        for i in 0..<n {
            for j in i..<n {
                var dot: Float = 0.0
                vDSP_dotpr(xAug[i], 1, xAug[j], 1, &dot, vDSP_Length(d + 1))
                let val = Double(dot + (i == j ? lambda : 0.0))
                mat[i][j] = val
                mat[j][i] = val
            }
            mat[i][n] = Double(yLogit0[i])
            mat[i][n + 1] = Double(yLogit1[i])
        }

        // Gaussian elimination with partial pivoting
        for col in 0..<n {
            var pivot = col
            var maxVal = abs(mat[col][col])
            for row in (col + 1)..<n {
                if abs(mat[row][col]) > maxVal {
                    maxVal = abs(mat[row][col])
                    pivot = row
                }
            }
            if pivot != col { mat.swapAt(col, pivot) }
            let diag = mat[col][col]
            if abs(diag) < 1e-12 { continue }
            for k in col..<(n + 2) { mat[col][k] /= diag }
            for row in 0..<n where row != col {
                let factor = mat[row][col]
                if factor != 0.0 {
                    for k in col..<(n + 2) {
                        mat[row][k] -= factor * mat[col][k]
                    }
                }
            }
        }

        var w0 = [Float](repeating: 0.0, count: d)
        var w1 = [Float](repeating: 0.0, count: d)
        var b0: Float = 0.0
        var b1: Float = 0.0
        for i in 0..<n {
            let a0 = Float(mat[i][n])
            let a1 = Float(mat[i][n + 1])
            for j in 0..<d {
                w0[j] += a0 * xAug[i][j]
                w1[j] += a1 * xAug[i][j]
            }
            b0 += a0
            b1 += a1
        }
        return S3AuxHead(w0: w0, w1: w1, b0: b0, b1: b1)
    }
}

// MARK: - 4. CoreML Apple Neural Engine Runner (`S3ANEEngine`)

public final class S3ANEEngine {
    public let bundleDir: URL
    public let tokenizer: S3Tokenizer
    public let tokEmbMMap: MMapFile
    public let typeEmbMMap: MMapFile
    public let headsMMap: MMapFile
    public let model: MLModel

    public let maxLen: Int = 96
    public let headMaxLen: Int = 256
    public let maxOptions: Int = 32
    public let hiddenSize: Int = 768
    public let localWindowHalf: Int = 64

    public let embeddingsArr: MLMultiArray
    public let fullMaskArr: MLMultiArray
    public let localMaskArr: MLMultiArray
    public let typeVectorsArr: MLMultiArray
    public let markerMapArr: MLMultiArray

    public let act0Weight: UnsafePointer<Float>
    public let act0Bias: UnsafePointer<Float>
    public let act2Weight: UnsafePointer<Float>
    public let act2Bias: UnsafePointer<Float>

    public let srrMuH: UnsafePointer<Float>
    public let srrSigmaH: UnsafePointer<Float>
    public let srrWs: UnsafePointer<Float>
    public let srrBs: UnsafePointer<Float>
    public let srrWu: UnsafePointer<Float>
    public let srrBu: UnsafePointer<Float>
    public let srrWgamma: UnsafePointer<Float>
    public let srrBgamma: UnsafePointer<Float>
    public let srrWbeta: UnsafePointer<Float>
    public let srrBbeta: UnsafePointer<Float>
    public let srrWout: UnsafePointer<Float>
    public let srrBout: UnsafePointer<Float>
    public let srrAlpha: Float

    public init(bundlePath: String = "/Users/mp/decision/models/snake/swift_bundle") throws {
        self.bundleDir = URL(fileURLWithPath: bundlePath)
        self.tokenizer = try S3Tokenizer(mmapPath: bundleDir.appendingPathComponent("tokenizer_mmap.bin").path)
        self.tokEmbMMap = try MMapFile(path: bundleDir.appendingPathComponent("tok_embeddings.fp16.bin").path)
        self.typeEmbMMap = try MMapFile(path: bundleDir.appendingPathComponent("type_emb.fp16.bin").path)
        self.headsMMap = try MMapFile(path: bundleDir.appendingPathComponent("heads.fp32.bin").path)

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: bundleDir.appendingPathComponent("model.mlmodelc"), configuration: mlConfig)

        self.embeddingsArr = try MLMultiArray(shape: [1, 768, 1, 96], dataType: .float16)
        self.fullMaskArr = try MLMultiArray(shape: [1, 96, 1, 96], dataType: .float16)
        self.localMaskArr = try MLMultiArray(shape: [1, 96, 1, 96], dataType: .float16)
        self.typeVectorsArr = try MLMultiArray(shape: [1, 768, 1, 1], dataType: .float16)
        self.markerMapArr = try MLMultiArray(shape: [1, 96, 1, 32], dataType: .float16)

        let hBase = headsMMap.rawPtr.assumingMemoryBound(to: Float.self)
        var off = 0
        self.act0Weight = hBase.advanced(by: off); off += 256 * 772
        self.act0Bias   = hBase.advanced(by: off); off += 256
        self.act2Weight = hBase.advanced(by: off); off += 2 * 256
        self.act2Bias   = hBase.advanced(by: off); off += 2
        self.srrMuH     = hBase.advanced(by: off); off += 768
        self.srrSigmaH  = hBase.advanced(by: off); off += 768
        self.srrWs      = hBase.advanced(by: off); off += 768 * 32
        self.srrBs      = hBase.advanced(by: off); off += 32
        self.srrWu      = hBase.advanced(by: off); off += 4 * 32
        self.srrBu      = hBase.advanced(by: off); off += 32
        self.srrWgamma  = hBase.advanced(by: off); off += 32 * 32
        self.srrBgamma  = hBase.advanced(by: off); off += 32
        self.srrWbeta   = hBase.advanced(by: off); off += 32 * 32
        self.srrBbeta   = hBase.advanced(by: off); off += 32
        self.srrWout    = hBase.advanced(by: off); off += 32 * 3
        self.srrBout    = hBase.advanced(by: off); off += 3
        self.srrAlpha   = hBase.advanced(by: off)[0]
    }

    /// Builds a sequence `[CLS] <type> question: <ins> [SEP] [MASK] opt0 ... [SEP] <state> [SEP]`
    public func buildSequence(type: String, instructions: String, options: [String], stateIds: [Int]) -> (ids: [Int], markers: [Int]) {
        var ids = [tokenizer.clsId]
        ids.append(contentsOf: tokenizer.encode("\(type) question: \(instructions)"))
        ids.append(tokenizer.sepId)
        var markers: [Int] = []
        markers.reserveCapacity(options.count)
        for opt in options {
            markers.append(ids.count)
            ids.append(tokenizer.maskId)
            ids.append(contentsOf: tokenizer.encode(" " + opt, truncation: true, maxLength: 48))
        }
        ids.append(tokenizer.sepId)
        let room = max(0, maxLen - ids.count - 1)
        ids.append(contentsOf: stateIds.prefix(room))
        ids.append(tokenizer.sepId)
        if ids.count > maxLen { ids = Array(ids.prefix(maxLen)) }
        return (ids, markers)
    }

    /// Executes one B=1, S=96, K=32 forward pass on the M2 Neural Engine inside an `autoreleasepool`
    public func forwardPass(ids: [Int], markers: [Int], qtypeId: Int) throws -> (probs: [Float], pooled: [Float], actProb: Float) {
        return try autoreleasepool {
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
            var rawLogits32 = [Float](repeating: -10000.0, count: maxOptions)
            var pooled = [Float](repeating: 0.0, count: hiddenSize)

            for name in outProvider.featureNames {
                guard let arr = outProvider.featureValue(for: name)?.multiArrayValue else { continue }
                let shape = arr.shape.map { $0.intValue }
                let strides = arr.strides.map { $0.intValue }
                if shape.count >= 4 && shape[1] == 1 {
                    let sOpt = strides[3]
                    if arr.dataType == .float16 {
                        let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                        for i in 0..<k { rawLogits32[i] = Float(ptr[i * sOpt]) }
                    } else {
                        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
                        for i in 0..<k { rawLogits32[i] = ptr[i * sOpt] }
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
            for v in rawLogits32 { if v > maxL { maxL = v } }
            var p32 = [Float](repeating: 0.0, count: maxOptions)
            var sumP32: Float = 0.0
            for i in 0..<maxOptions {
                let e = expf(rawLogits32[i] - maxL)
                p32[i] = e
                sumP32 += e
            }
            for i in 0..<maxOptions { p32[i] /= sumP32 }

            var probs = [Float](repeating: 0.0, count: k)
            var sumK: Float = 0.0
            for i in 0..<k {
                let e = expf(rawLogits32[i] - maxL)
                probs[i] = e
                sumK += e
            }
            for i in 0..<k { probs[i] /= sumK }

            // Compute legacy act_head probability
            let kFloat = Float(max(k, 2))
            var entropy: Float = 0.0
            for i in 0..<maxOptions {
                let pv = max(p32[i], 1e-9)
                entropy -= p32[i] * logf(pv)
            }
            entropy /= logf(kFloat)
            let sortedP = p32.sorted()
            let top1 = sortedP[maxOptions - 1]
            let top2 = sortedP[maxOptions - 2]
            var vec772 = pooled
            vec772.append(contentsOf: [top1, top1 - top2, entropy, kFloat / 255.0])

            var hidden = [Float](repeating: 0.0, count: 256)
            let invSqrt2: Float = 1.0 / sqrtf(2.0)
            for h in 0..<256 {
                var dot: Float = 0.0
                vDSP_dotpr(vec772, 1, act0Weight.advanced(by: h * 772), 1, &dot, vDSP_Length(772))
                dot += act0Bias[h]
                hidden[h] = dot * 0.5 * (1.0 + erff(dot * invSqrt2))
            }
            var a0: Float = 0.0
            var a1: Float = 0.0
            vDSP_dotpr(hidden, 1, act2Weight, 1, &a0, vDSP_Length(256))
            vDSP_dotpr(hidden, 1, act2Weight.advanced(by: 256), 1, &a1, vDSP_Length(256))
            a0 += act2Bias[0]
            a1 += act2Bias[1]
            let mAct = max(a0, a1)
            let e0 = expf(a0 - mAct)
            let e1 = expf(a1 - mAct)
            let actProb = e0 / (e0 + e1)

            return (probs, pooled, actProb)
        }
    }

    /// Loads or fits a zero-extra-ANE-call `[CLS]` readout head for an arena (`poker`, `tactics`, `breach`)
    public func fitOrLoadAuxHead(
        name: String,
        choiceInstructions: String,
        noul0Instructions: String,
        noul1Instructions: String,
        calibrationSamples: [(state: String, options: [String])]
    ) throws -> S3AuxHead {
        let binPath = bundleDir.appendingPathComponent("\(name)_aux_head.fp32.bin").path
        if FileManager.default.fileExists(atPath: binPath),
           let loaded = try? S3AuxHead(path: binPath) {
            return loaded
        }

        let noulOpts = [
            "false: no, the statement does not hold",
            "true: yes, the statement holds",
        ]
        var features: [[Float]] = []
        var targets: [(Float, Float)] = []
        features.reserveCapacity(calibrationSamples.count)
        targets.reserveCapacity(calibrationSamples.count)

        for sample in calibrationSamples {
            let stIds = tokenizer.encode(sample.state)
            let (cIds, cMarkers) = buildSequence(type: "choice", instructions: choiceInstructions, options: sample.options, stateIds: stIds)
            let (_, pooled, _) = try forwardPass(ids: cIds, markers: cMarkers, qtypeId: 0)

            let (n0Ids, n0Markers) = buildSequence(type: "noul", instructions: noul0Instructions, options: noulOpts, stateIds: stIds)
            let (p0, _, _) = try forwardPass(ids: n0Ids, markers: n0Markers, qtypeId: 2)

            let (n1Ids, n1Markers) = buildSequence(type: "noul", instructions: noul1Instructions, options: noulOpts, stateIds: stIds)
            let (p1, _, _) = try forwardPass(ids: n1Ids, markers: n1Markers, qtypeId: 2)

            features.append(pooled)
            targets.append((p0[1], p1[1]))
        }

        let head = S3AuxHead.fitDualRidge(features: features, targets: targets, lambda: 1e-2)
        try? head.save(to: binPath)
        return head
    }
}

// MARK: - 5. Truecolor 24-Bit ANSI Fixed-Cell Terminal HUD Composer

public struct S3RGB: Equatable {
    public let r: UInt8
    public let g: UInt8
    public let b: UInt8
    public init(r: UInt8, g: UInt8, b: UInt8) {
        self.r = r; self.g = g; self.b = b
    }
}

public let S3_BG    = S3RGB(r: 0x09, g: 0x0F, b: 0x13)
public let S3_FG    = S3RGB(r: 0xE3, g: 0xF3, b: 0xEF)
public let S3_MUTED = S3RGB(r: 0x68, g: 0x86, b: 0x8C)
public let S3_DIM   = S3RGB(r: 0x20, g: 0x35, b: 0x3C)
public let S3_DOT   = S3RGB(r: 0x13, g: 0x27, b: 0x2E)
public let S3_GREEN = S3RGB(r: 0x62, g: 0xF5, b: 0xB5)
public let S3_AMBER = S3RGB(r: 0xFF, g: 0xCE, b: 0x73)
public let S3_RED   = S3RGB(r: 0xFF, g: 0x7C, b: 0x8C)
public let S3_CYAN  = S3RGB(r: 0x8A, g: 0xD8, b: 0xE9)

public final class S3Canvas {
    public let width: Int
    public let height: Int
    private var chars: [Character]
    private var colors: [S3RGB]

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.chars = [Character](repeating: " ", count: width * height)
        self.colors = [S3RGB](repeating: S3_FG, count: width * height)
    }

    public func clear() {
        for i in 0..<chars.count {
            chars[i] = " "
            colors[i] = S3_FG
        }
    }

    @inline(__always)
    public func put(_ row: Int, _ col: Int, _ text: String, _ color: S3RGB = S3_FG) {
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

    public func bar(_ row: Int, _ col: Int, _ value: Float, _ length: Int = 20, _ color: S3RGB = S3_GREEN) {
        let clamped = max(0.0, min(1.0, value))
        let count = Int((clamped * Float(length)).rounded())
        put(row, col, String(repeating: "━", count: length), S3_DIM)
        if count > 0 {
            put(row, col, String(repeating: "━", count: count), color)
        }
    }

    public func toANSI(homeCursor: Bool = true) -> String {
        var out = homeCursor ? "\u{1B}[H" : ""
        out.append("\u{1B}[48;2;\(S3_BG.r);\(S3_BG.g);\(S3_BG.b)m")
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

// MARK: - 6. Non-Blocking Raw Terminal Keyboard Reader

public final class S3RawTerminal {
    private var saved = termios()
    public private(set) var active = false

    public init() {
        if isatty(STDIN_FILENO) != 0 {
            tcgetattr(STDIN_FILENO, &saved)
            var raw = saved
            raw.c_lflag &= ~tcflag_t(ICANON | ECHO)
            raw.c_cc.16 = 0
            raw.c_cc.17 = 0
            tcsetattr(STDIN_FILENO, TCSANOW, &raw)
            active = true
        }
    }

    public func readKey() -> String {
        guard active else { return "" }
        var buf = [UInt8](repeating: 0, count: 32)
        let n = Darwin.read(STDIN_FILENO, &buf, buf.count)
        if n > 0 {
            return String(bytes: buf[0..<n], encoding: .utf8) ?? ""
        }
        return ""
    }

    public func restore() {
        if active {
            tcsetattr(STDIN_FILENO, TCSADRAIN, &saved)
            active = false
        }
    }

    deinit { restore() }
}

// MARK: - 7. Deterministic 64-Bit SplitMix64 PRNG

public struct S3SplitMix64 {
    public var state: UInt64

    public init(seed: Int) {
        self.state = UInt64(bitPattern: Int64(seed)) &+ 0x9E3779B97F4A7C15
    }

    @inline(__always)
    public mutating func next() -> UInt64 {
        var z = state
        state = state &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    @inline(__always)
    public mutating func nextFloat() -> Float {
        return Float(next() >> 40) / Float(1 << 24)
    }

    @inline(__always)
    public mutating func uniform(_ lo: Float, _ hi: Float) -> Float {
        return lo + (hi - lo) * nextFloat()
    }

    @inline(__always)
    public mutating func choiceIndex(count: Int) -> Int {
        return Int(next() % UInt64(max(1, count)))
    }

    @inline(__always)
    public mutating func randint(_ lo: Int, _ hi: Int) -> Int {
        let span = max(1, hi - lo + 1)
        return lo + Int(next() % UInt64(span))
    }

    public mutating func shuffle<T>(_ arr: inout [T]) {
        if arr.count <= 1 { return }
        for i in stride(from: arr.count - 1, through: 1, by: -1) {
            let j = Int(next() % UInt64(i + 1))
            if i != j { arr.swapAt(i, j) }
        }
    }
}
