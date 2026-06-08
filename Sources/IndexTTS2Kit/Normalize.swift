import Foundation

/// Text normalization + CJK tokenization, ported from the Python reference.
/// The heavy wetext (zh/en) normalizers are disabled there, so this mirrors
/// the active code path: punctuation mapping, contraction expansion,
/// English number-to-words, and CJK-char splitting.
enum TextNormalize {

    static let charMap: [String: String] = [
        "：": ",", "；": ",", ";": ",", "，": ",", "。": ".", "！": "!", "？": "?",
        "\n": " ", "·": "-", "、": ",", "...": "…", ",,,": "…", "，，，": "…", "……": "…",
        "\u{201C}": "'", "\u{201D}": "'", "\"": "'", "'": "'", "（": "'", "）": "'",
        "(": "'", ")": "'", "《": "'", "》": "'", "【": "'", "】": "'", "[": "'", "]": "'",
        "—": "-", "～": "-", "~": "-", "「": "'", "」": "'", ":": ",",
    ]
    static var zhCharMap: [String: String] {
        var m = charMap
        m["$"] = "."
        return m
    }

    static func hasChinese(_ s: String) -> Bool {
        s.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) }
    }
    static func hasAlpha(_ s: String) -> Bool {
        s.unicodeScalars.contains {
            (0x41...0x5A).contains($0.value) || (0x61...0x7A).contains($0.value)
        }
    }
    static func isEmail(_ s: String) -> Bool {
        s.range(of: #"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"#, options: .regularExpression) != nil
    }
    static func useChinese(_ s: String) -> Bool {
        hasChinese(s) || !hasAlpha(s) || isEmail(s)
    }

    static func replaceChars(_ text: String, _ map: [String: String]) -> String {
        // Apply multi-char keys first (longest match) to mirror the regex alternation.
        var result = text
        for key in map.keys.sorted(by: { $0.count > $1.count }) {
            result = result.replacingOccurrences(of: key, with: map[key]!)
        }
        return result
    }

    static func expandContractions(_ text: String) -> String {
        let pattern = #"(?i)(what|where|who|which|how|t?here|it|s?he|that|this)'s"#
        return regexReplace(text, pattern, template: "$1 is")
    }

    static func numberToWords(_ n: Int) -> String {
        let ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        let teens = [
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen",
        ]
        let tens = [
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        ]
        let thousands = ["", "thousand", "million", "billion", "trillion"]

        func convertHundreds(_ num: Int) -> String {
            if num == 0 { return "" }
            if num < 10 { return ones[num] }
            if num < 20 { return teens[num - 10] }
            if num < 100 {
                return tens[num / 10] + (num % 10 != 0 ? " " + ones[num % 10] : "")
            }
            return ones[num / 100] + " hundred"
                + (num % 100 != 0 ? " " + convertHundreds(num % 100) : "")
        }

        if n == 0 { return "zero" }
        var groups: [String] = []
        var num = n
        var groupIdx = 0
        while num > 0 {
            let group = num % 1000
            if group != 0 {
                var words = convertHundreds(group)
                if !thousands[groupIdx].isEmpty { words += " " + thousands[groupIdx] }
                groups.append(words)
            }
            num /= 1000
            groupIdx += 1
        }
        return groups.reversed().joined(separator: " ")
    }

    static func normalizeEnglish(_ text0: String) -> String {
        var text = expandContractions(text0)

        // currency: $123 -> "one hundred twenty three dollars"
        text = regexReplaceFn(text, #"\$\s*[0-9,.\s]+"#) { match in
            let digits = match.filter { $0.isNumber }
            guard let num = Int(digits) else { return match }
            return "\(numberToWords(num)) dollar\(num != 1 ? "s" : "") "
        }
        // spaced single digits: "1 2 3" -> "one two three"
        text = regexReplaceFn(text, #"\b\d(\s+\d)+\b"#) { match in
            let parts = match.split(separator: " ").map(String.init)
            if parts.allSatisfy({ $0.count == 1 && $0.allSatisfy(\.isNumber) }) {
                return parts.map { numberToWords(Int($0)!) }.joined(separator: " ")
            }
            return numberToWords(Int(match.filter { $0.isNumber }) ?? 0)
        }
        // integers with optional commas
        text = regexReplaceFn(text, #"\b\d+(?:,\d+)*\b"#) { match in
            let digits = match.filter { $0.isNumber }
            return digits.isEmpty ? match : numberToWords(Int(digits) ?? 0)
        }
        text = regexReplace(text, #"\s+"#, template: " ").trimmingCharacters(in: .whitespaces)
        return replaceChars(text, charMap)
    }

    static func normalizeChinese(_ text0: String) -> String {
        // wetext zh normalizer disabled in reference -> passthrough + char map.
        let text = expandContractions(text0.trimmingCharacters(in: .whitespaces))
        return replaceChars(text, zhCharMap)
    }

    static func normalize(_ text: String) -> String {
        useChinese(text) ? normalizeChinese(text) : normalizeEnglish(text)
    }

    /// CJK scalar ranges (matches the reference regex character class).
    static func isCJK(_ v: UInt32) -> Bool {
        (0x1100...0x11FF).contains(v) || (0x2E80...0xA4CF).contains(v)
            || (0xA840...0xD7AF).contains(v) || (0xF900...0xFAFF).contains(v)
            || (0xFE30...0xFE4F).contains(v) || (0xFF65...0xFFDC).contains(v)
            || (0x20000...0x2FFFF).contains(v)
    }

    /// Split CJK characters into individual tokens; upper-case everything.
    /// e.g. "你好 hello" -> "你 好 HELLO"
    static func tokenizeByCJKChar(_ line: String) -> String {
        var tokens: [String] = []
        var buffer = ""
        for scalar in line.trimmingCharacters(in: .whitespaces).unicodeScalars {
            if isCJK(scalar.value) {
                if !buffer.isEmpty { tokens.append(buffer); buffer = "" }
                tokens.append(String(scalar))
            } else {
                buffer.unicodeScalars.append(scalar)
            }
        }
        if !buffer.isEmpty { tokens.append(buffer) }
        return tokens.map { $0.trimmingCharacters(in: .whitespaces).uppercased() }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    // MARK: regex helpers

    static func regexReplace(_ text: String, _ pattern: String, template: String) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return re.stringByReplacingMatches(in: text, range: range, withTemplate: template)
    }

    static func regexReplaceFn(_ text: String, _ pattern: String, _ fn: (String) -> String)
        -> String
    {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return text }
        let ns = text as NSString
        var result = ""
        var last = 0
        for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            result += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            result += fn(ns.substring(with: m.range))
            last = m.range.location + m.range.length
        }
        result += ns.substring(from: last)
        return result
    }
}
