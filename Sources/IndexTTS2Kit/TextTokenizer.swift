import Foundation

/// High-level text tokenizer for IndexTTS-2: normalization + CJK splitting +
/// SentencePiece (unigram) pieces, plus sentence segmentation for long text.
/// Port of `tokenizer.py::TextTokenizer`.
public final class TextTokenizer {
    let sp: UnigramTokenizer

    /// Tokens that mark a sentence boundary (used by `splitSegments`).
    static let punctuationMarks: Set<String> = [
        ".", "!", "?", "\u{2581}.", "\u{2581}?", "...", "\u{2581}...",
    ]

    public init(modelPath: URL) throws {
        self.sp = try UnigramTokenizer(modelPath: modelPath)
    }

    /// Full normalization then CJK-char splitting, matching `tokenizer.py::tokenize`.
    public func tokenize(_ text: String, normalize: Bool = true) -> [String] {
        var t = text
        if normalize { t = TextNormalize.normalize(t) }
        t = TextNormalize.tokenizeByCJKChar(t)
        return sp.encodePieces(t)
    }

    public func convertTokensToIds(_ tokens: [String]) -> [Int] {
        tokens.map { sp.pieceToId($0) }
    }

    /// Split a token list into segments at natural sentence boundaries.
    public func splitSegments(_ tokens: [String], maxTokensPerSegment: Int = 120) -> [[String]] {
        Self.splitByToken(
            tokens, splitTokens: Self.punctuationMarks,
            maxTokensPerSegment: maxTokensPerSegment)
    }

    /// Port of `_split_segments_by_token`: recursive split on punctuation, then
    /// comma, then hyphen, force-splitting overly long runs and merging short ones.
    static func splitByToken(
        _ tokenized: [String], splitTokens: Set<String>, maxTokensPerSegment: Int
    ) -> [[String]] {
        if tokenized.isEmpty { return [] }

        var segments: [[String]] = []
        var current: [String] = []

        let hasComma = splitTokens.contains(",") || splitTokens.contains("\u{2581},")
        let hasHyphen = splitTokens.contains("-")

        var i = 0
        while i < tokenized.count {
            let token = tokenized[i]
            current.append(token)

            if !hasComma && (current.contains(",") || current.contains("\u{2581},")) {
                segments.append(contentsOf: splitByToken(
                    current, splitTokens: [",", "\u{2581},"],
                    maxTokensPerSegment: maxTokensPerSegment))
                current = []
            } else if !hasHyphen && current.contains("-") {
                segments.append(contentsOf: splitByToken(
                    current, splitTokens: ["-"], maxTokensPerSegment: maxTokensPerSegment))
                current = []
            } else if current.count <= maxTokensPerSegment {
                if splitTokens.contains(token) && current.count > 2 {
                    // Don't split before a closing quote.
                    if i < tokenized.count - 1 {
                        let next = tokenized[i + 1]
                        if next == "'" || next == "\u{2581}'" {
                            current.append(next)
                            i += 1
                        }
                    }
                    segments.append(current)
                    current = []
                }
            } else {
                // Exceeded max length -> force split.
                var j = 0
                while j < current.count {
                    let end = min(j + maxTokensPerSegment, current.count)
                    segments.append(Array(current[j ..< end]))
                    j += maxTokensPerSegment
                }
                current = []
            }
            i += 1
        }
        if !current.isEmpty { segments.append(current) }

        // Merge short adjacent segments.
        var merged: [[String]] = []
        for seg in segments where !seg.isEmpty {
            if merged.isEmpty {
                merged.append(seg)
            } else if merged[merged.count - 1].count + seg.count <= maxTokensPerSegment {
                merged[merged.count - 1].append(contentsOf: seg)
            } else {
                merged.append(seg)
            }
        }
        return merged
    }
}
