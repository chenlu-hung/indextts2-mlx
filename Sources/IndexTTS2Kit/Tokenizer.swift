import Foundation

/// Pure-Swift SentencePiece **Unigram** tokenizer.
///
/// Parses `tokenizer.model` (SentencePiece ModelProto), applies the active
/// normalization (NFKC + whitespace escaping with ▁), and runs Viterbi over the
/// unigram language model. No PyTorch / C++ dependency.
///
/// Note: the SentencePiece `nmt_nfkc` precompiled charmap is approximated by
/// Foundation NFKC. For the already-normalized CN/EN input IndexTTS feeds in
/// (CJK split, upper-cased, ASCII punctuation), this matches in practice.
public final class UnigramTokenizer {
    private struct Piece { let score: Double }
    private var vocab: [String: (id: Int, score: Double)] = [:]
    private var maxPieceScalars = 1
    private let unkId: Int
    private let unkScore: Double

    private static let whitespacePiece = "\u{2581}"  // ▁

    public init(modelPath: URL) throws {
        let data = try Data(contentsOf: modelPath)
        var pieces: [(String, Double)] = []
        var cursor = 0
        let bytes = [UInt8](data)

        func readVarint() -> Int {
            var shift = 0
            var result = 0
            while cursor < bytes.count {
                let b = bytes[cursor]
                cursor += 1
                result |= Int(b & 0x7F) << shift
                if b & 0x80 == 0 { break }
                shift += 7
            }
            return result
        }

        // top-level scan: field 1 = pieces (LEN)
        while cursor < bytes.count {
            let tag = readVarint()
            let field = tag >> 3
            let wire = tag & 7
            if wire == 2 {
                let len = readVarint()
                let end = cursor + len
                if field == 1 {
                    // SentencePiece sub-message: f1 piece(str), f2 score(float32)
                    var piece = ""
                    var score = 0.0
                    var c2 = cursor
                    func rv2() -> Int {
                        var s = 0
                        var r = 0
                        while c2 < end {
                            let b = bytes[c2]
                            c2 += 1
                            r |= Int(b & 0x7F) << s
                            if b & 0x80 == 0 { break }
                            s += 7
                        }
                        return r
                    }
                    while c2 < end {
                        let t2 = rv2()
                        let f2 = t2 >> 3
                        let w2 = t2 & 7
                        if w2 == 2 {
                            let l2 = rv2()
                            if f2 == 1 {
                                piece = String(decoding: bytes[c2 ..< c2 + l2], as: UTF8.self)
                            }
                            c2 += l2
                        } else if w2 == 5 {
                            let raw = UInt32(bytes[c2]) | UInt32(bytes[c2 + 1]) << 8
                                | UInt32(bytes[c2 + 2]) << 16 | UInt32(bytes[c2 + 3]) << 24
                            if f2 == 2 { score = Double(Float(bitPattern: raw)) }
                            c2 += 4
                        } else if w2 == 0 {
                            _ = rv2()
                        } else if w2 == 1 {
                            c2 += 8
                        } else {
                            break
                        }
                    }
                    pieces.append((piece, score))
                }
                cursor = end
            } else if wire == 0 {
                _ = readVarint()
            } else if wire == 1 {
                cursor += 8
            } else if wire == 5 {
                cursor += 4
            } else {
                break
            }
        }

        var unk = 0
        var minScore = Double.greatestFiniteMagnitude
        for (i, (piece, score)) in pieces.enumerated() {
            vocab[piece] = (i, score)
            maxPieceScalars = max(maxPieceScalars, piece.unicodeScalars.count)
            if piece == "<unk>" { unk = i }
            if score < minScore { minScore = score }
        }
        self.unkId = unk
        self.unkScore = minScore - 10.0
    }

    public var vocabSize: Int { vocab.count }

    /// Piece-string id lookup (SentencePiece `PieceToId`), falling back to `<unk>`.
    public func pieceToId(_ piece: String) -> Int { vocab[piece]?.id ?? unkId }

    /// SentencePiece normalization (approximated): NFKC, collapse whitespace,
    /// dummy prefix, escape spaces with ▁.
    private func normalize(_ text: String) -> [Unicode.Scalar] {
        let nfkc = text.precomposedStringWithCompatibilityMapping
        let collapsed = nfkc.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" })
            .joined(separator: " ")
        let prefixed = " " + collapsed
        let escaped = prefixed.replacingOccurrences(of: " ", with: Self.whitespacePiece)
        return Array(escaped.unicodeScalars)
    }

    /// Encode text into unigram token ids (no BOS/EOS).
    public func encode(_ text: String) -> [Int] {
        encodePieces(text).map(pieceToId)
    }

    /// Encode text into unigram **piece strings** (SentencePiece `EncodeAsPieces`),
    /// with the active normalization applied. Pieces carry the ▁ whitespace marker.
    public func encodePieces(_ text: String) -> [String] {
        let scalars = normalize(text)
        let n = scalars.count
        if n == 0 { return [] }

        let negInf = -Double.greatestFiniteMagnitude
        var best = [Double](repeating: negInf, count: n + 1)
        var backStart = [Int](repeating: -1, count: n + 1)
        best[0] = 0

        for i in 1 ... n {
            let lo = max(0, i - maxPieceScalars)
            var j = i - 1
            while j >= lo {
                if best[j] != negInf {
                    let sub = String(String.UnicodeScalarView(scalars[j ..< i]))
                    if let entry = vocab[sub] {
                        let cand = best[j] + entry.score
                        if cand > best[i] {
                            best[i] = cand
                            backStart[i] = j
                        }
                    }
                }
                j -= 1
            }
            if best[i] == negInf {
                // single-scalar unknown fallback
                best[i] = best[i - 1] + unkScore
                backStart[i] = i - 1
            }
        }

        var pieces: [String] = []
        var pos = n
        while pos > 0 {
            let start = backStart[pos]
            pieces.append(String(String.UnicodeScalarView(scalars[start ..< pos])))
            pos = start
        }
        return pieces.reversed()
    }
}
