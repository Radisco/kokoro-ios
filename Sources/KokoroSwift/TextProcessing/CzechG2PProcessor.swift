// CzechG2PProcessor.swift
// VoxoLoxo fork — Swift port cs_g2p.py (MIT, bez GPL dependencies).
//
// Pravidlový Text → kompaktný IPA pre češtinu, matching eSpeak/phonemizer výstup
// použitý pri Kokoro CS trénovaní (compact IPA, preserve_punctuation = true).
//
// Pokrýva:
//   - Grafémy → IPA tokeny (digrafy: ch, dž, dz; palatalizácia: dě/tě/ně/di/ti/ni/mě)
//   - Diftongy: ou → oʊ, au → aʊ, eu → eʊ, ej → eɪ (podmienene)
//   - Velárna nasalizácia: n pred k/ɡ → ŋ
//   - ř devoicing pri kontakte s neznelými
//   - Regresívna asimilácia znelosti
//   - Finálne devoicing
//   - Degemination (dvojité spoluhlásky → jedna)
//   - Syllabické l̩/r̩
//   - Primary a secondary stress
//   - Lexikálny override pre klitiká a funkčné slová
//
// Licencia: MIT

import Foundation
import MLXUtilsLibrary  // MToken

// MARK: - Czech IPA Tables

private let vowelMap: [Character: String] = [
    "a": "a",  "á": "aː",
    "e": "e",  "é": "eː",
    "i": "i",  "í": "iː",
    "o": "o",  "ó": "oː",
    "u": "u",  "ú": "uː",  "ů": "uː",
    "y": "i",  "ý": "iː",
]

private let vowelIPA: Set<String> = Set(vowelMap.values)
private let vowelLatin: Set<Character> = Set(vowelMap.keys)

private let consonantMap: [Character: String] = [
    "b": "b",  "c": "ts", "č": "tʃ",
    "d": "d",  "ď": "ɟ",
    "f": "f",  "g": "ɡ",
    "h": "h",
    "j": "j",  "k": "k",  "l": "l",
    "m": "m",  "n": "n",  "ň": "ɲ",
    "p": "p",  "q": "k",
    "r": "r",  "ř": "r̝",
    "s": "s",  "š": "ʃ",
    "t": "t",  "ť": "c",
    "v": "v",  "w": "v",
    "x": "ks",
    "z": "z",  "ž": "ʒ",
]

private let voicedToVoiceless: [String: String] = [
    "b": "p",  "d": "t",   "ɟ": "c",
    "ɡ": "k",
    "z": "s",  "ʒ": "ʃ",
    "v": "f",
    "dz": "ts", "dʒ": "tʃ",
    "r̝": "r̝̊",
]
private let voicelessToVoiced: [String: String] = Dictionary(
    uniqueKeysWithValues: voicedToVoiceless.map { ($1, $0) }
)

private let voicelessIPA: Set<String> = [
    "p", "t", "c", "k", "s", "ʃ", "f", "x", "ts", "tʃ", "r̝̊", "h"
]
private let voicedIPA: Set<String> = [
    "b", "d", "ɟ", "ɡ", "z", "ʒ", "v", "dz", "dʒ", "r̝"
]
private let sonorantsIPA: Set<String> = [
    "r", "l", "m", "n", "ɲ", "ŋ", "j", "l̩", "r̩"
]
// Konsonanty ktoré NEspúšťajú regresívnu asimiláciu doľava
private let noTriggerBackward: Set<String> = sonorantsIPA.union(["v", "r̝", "r̝̊", "j"])

// Lexikálny override pre klitiká a funkčné slová
private let lexicon: [String: String] = [
    "od": "od",      "nad": "nad",    "pod": "pod",
    "před": "pr̝̊ed", "v": "v",        "s": "s",
    "z": "z",        "k": "k",
    "když": "kdˈiʒ", "kdo": "kdˈo",  "kde": "kdˈe",  "kdy": "kdˈi",
    "se": "se",      "si": "si",      "je": "je",      "jsem": "jsem",
    "že": "ʒe",      "by": "bi",      "bi": "bi",      "mi": "mi",
    "už": "ˈuʒ",     "a": "a",        "i": "i",        "o": "ˈo",
    "u": "ˈu",       "do": "do",      "na": "na",      "za": "za",
    "ten": "ten",    "ta": "ta",      "to": "to",
    "byl": "bil",    "byla": "bila",  "bylo": "bilo",  "byli": "bili",
    "být": "biːt",   "jsi": "jsi",    "jsou": "jsoʊ",
    "jej": "jej",    "jich": "jix",   "jim": "jim",
    "jen": "jen",    "jak": "jak",    "tak": "tak",
    "ale": "ale",    "ani": "ani",    "aby": "abi",
    "no": "no",      "nebo": "nebo",
]

// Regex pre slová a interpunkciu
private let wordCharacters = CharacterSet.letters
private let punctCharacters = CharacterSet(charactersIn: ".,!?;:–—-\"'()[]\u{2026}\u{201e}\u{201c}\u{201a}\u{2018}\u{2019}\u{00ab}\u{00bb}")

// MARK: - CzechG2PProcessor

/// Swift port `CzechG2P` z cs_g2p.py (MIT).
/// Implementuje `G2PProcessor` pre integráciu do `KokoroTTS`.
public final class CzechG2PProcessor: G2PProcessor {

    public init() {}

    // MARK: - G2PProcessor

    public func setLanguage(_ language: Language) throws {
        guard language == .csCZ else {
            throw G2PProcessorError.unsupportedLanguage
        }
    }

    public func process(input: String) throws -> (String, [MToken]?) {
        let ipa = phonemize(input)
        return (ipa, nil)
    }

    // MARK: - Phonemize

    /// Text → IPA (s interpunkciou, kompaktný formát).
    public func phonemize(_ text: String) -> String {
        // NFC normalizácia
        let normalized = text.precomposedStringWithCanonicalMapping
        var out: [String] = []
        var index = normalized.startIndex

        while index < normalized.endIndex {
            let ch = normalized[index]

            // Skús slovo
            if let wordEnd = wordEnd(in: normalized, from: index) {
                let word = String(normalized[index ..< wordEnd])
                out.append(wordToIPA(word))
                index = wordEnd
                continue
            }

            // Interpunkcia → zachovaj
            if punctCharacters.contains(ch.unicodeScalars.first!) {
                out.append(String(ch))
                index = normalized.index(after: index)
                continue
            }

            // Whitespace → jedna medzera
            if ch.isWhitespace {
                if out.last != " " { out.append(" ") }
                index = normalized.index(after: index)
                continue
            }

            // Iné (číslice, emoji) → preskočiť
            index = normalized.index(after: index)
        }

        return out.joined().trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Word End Detection

    private func wordEnd(in text: String, from start: String.Index) -> String.Index? {
        guard start < text.endIndex else { return nil }
        let ch = text[start]
        guard ch.isLetter else { return nil }
        var i = text.index(after: start)
        while i < text.endIndex, text[i].isLetter {
            i = text.index(after: i)
        }
        return i
    }

    // MARK: - Word → IPA

    private func wordToIPA(_ word: String) -> String {
        let lower = word.lowercased()

        // Lexikálny override
        if let override = lexicon[lower] { return override }

        var tokens = graphemesToTokens(lower)
        tokens = applyDiphthongs(tokens)
        tokens = applyVelarNasal(tokens)
        tokens = applyVoicingAssimilation(tokens)
        tokens = applyRzDevoicing(tokens)
        tokens = applyFinalDevoicing(tokens)
        tokens = degeminate(tokens)
        tokens = markSyllabic(tokens)
        tokens = applyPrimaryStress(tokens)

        // -koliv/-koli suffix fix
        let result = tokens.joined()
        if lower.hasSuffix("koliv") && result.hasSuffix("f") {
            return String(result.dropLast()) + "v"
        }
        return result
    }

    // MARK: - Grafémy → tokeny

    private func graphemesToTokens(_ word: String) -> [String] {
        var tokens: [String] = []
        let chars = Array(word)
        var i = 0

        while i < chars.count {
            let ch  = chars[i]
            let nxt = i + 1 < chars.count ? chars[i + 1] : "\0"

            // Digrafy
            if ch == "c", nxt == "h" { tokens.append("x");  i += 2; continue }
            if ch == "d", nxt == "ž" { tokens.append("dʒ"); i += 2; continue }
            if ch == "d", nxt == "z" { tokens.append("dz"); i += 2; continue }

            // Palatalizácia — dě/tě/ně
            if ch == "d", nxt == "ě" { tokens += ["ɟ", "e"]; i += 2; continue }
            if ch == "t", nxt == "ě" { tokens += ["c", "e"]; i += 2; continue }
            if ch == "n", nxt == "ě" { tokens += ["ɲ", "e"]; i += 2; continue }

            // bě/pě/vě/fě → bje/pje/vje/fje
            if ["b", "p", "v", "f"].contains(ch), nxt == "ě" {
                tokens += [consonantMap[ch] ?? String(ch), "j", "e"]; i += 2; continue
            }

            // mě → mɲe
            if ch == "m", nxt == "ě" { tokens += ["m", "ɲ", "e"]; i += 2; continue }

            // di/ti/ni → palatalizácia pred i/í (NIE pred y/ý)
            if ch == "d", nxt == "i" || nxt == "í" {
                tokens += ["ɟ", nxt == "í" ? "iː" : "i"]; i += 2; continue
            }
            if ch == "t", nxt == "i" || nxt == "í" {
                tokens += ["c", nxt == "í" ? "iː" : "i"]; i += 2; continue
            }
            if ch == "n", nxt == "i" || nxt == "í" {
                tokens += ["ɲ", nxt == "í" ? "iː" : "i"]; i += 2; continue
            }

            // Samohlásky
            if let v = vowelMap[ch] { tokens.append(v); i += 1; continue }

            // Spoluhlásky
            if let c = consonantMap[ch] { tokens.append(c); i += 1; continue }

            // Neznámy znak → preskočiť
            i += 1
        }
        return tokens
    }

    // MARK: - Diftongy

    private func applyDiphthongs(_ tokens: [String]) -> [String] {
        var out: [String] = []
        var i = 0
        while i < tokens.count {
            let cur  = tokens[i]
            let nxt  = i + 1 < tokens.count ? tokens[i + 1] : ""
            let nxt2 = i + 2 < tokens.count ? tokens[i + 2] : ""

            if cur == "o", nxt == "u" { out += ["o", "ʊ"]; i += 2; continue }
            if cur == "a", nxt == "u" { out += ["a", "ʊ"]; i += 2; continue }
            if cur == "e", nxt == "u" { out += ["e", "ʊ"]; i += 2; continue }

            // ej → eɪ iba pred obštruent alebo koncom slova
            if cur == "e", nxt == "j" {
                let isObstruent = nxt2.isEmpty
                    || voicelessIPA.contains(nxt2)
                    || voicedIPA.contains(nxt2)
                if isObstruent { out += ["e", "ɪ"]; i += 2; continue }
            }

            out.append(cur); i += 1
        }
        return out
    }

    // MARK: - Velárna nasalizácia

    private func applyVelarNasal(_ tokens: [String]) -> [String] {
        var out = tokens
        for i in 0 ..< max(0, out.count - 1) {
            if out[i] == "n", out[i + 1] == "k" || out[i + 1] == "ɡ" {
                out[i] = "ŋ"
            }
        }
        return out
    }

    // MARK: - ř devoicing

    private func applyRzDevoicing(_ tokens: [String]) -> [String] {
        var out = tokens
        for i in 0 ..< out.count {
            guard out[i] == "r̝" else { continue }
            let prev = i > 0 ? out[i - 1] : ""
            let next = i + 1 < out.count ? out[i + 1] : ""
            if voicelessIPA.contains(prev) || voicelessIPA.contains(next) {
                out[i] = "r̝̊"
            }
        }
        return out
    }

    // MARK: - Regresívna asimilácia znelosti

    private func applyVoicingAssimilation(_ tokens: [String]) -> [String] {
        var out = tokens
        guard out.count >= 2 else { return out }
        for i in stride(from: out.count - 2, through: 0, by: -1) {
            let cur = out[i]
            let nxt = out[i + 1]
            guard voicedIPA.contains(cur) || voicelessIPA.contains(cur) else { continue }
            if noTriggerBackward.contains(nxt) { continue }
            if vowelIPA.contains(nxt) { continue }
            if voicelessIPA.contains(nxt), let dv = voicedToVoiceless[cur] { out[i] = dv }
            else if voicedIPA.contains(nxt), let dv = voicelessToVoiced[cur] { out[i] = dv }
        }
        return out
    }

    // MARK: - Finálne devoicing

    private func applyFinalDevoicing(_ tokens: [String]) -> [String] {
        guard !tokens.isEmpty else { return tokens }
        var out = tokens
        if let dv = voicedToVoiceless[out[out.count - 1]] {
            out[out.count - 1] = dv
        }
        return out
    }

    // MARK: - Degemination

    private func degeminate(_ tokens: [String]) -> [String] {
        guard !tokens.isEmpty else { return tokens }
        var out = [tokens[0]]
        for t in tokens.dropFirst() {
            if t == out.last && !vowelIPA.contains(t) && t != "ˈ" && t != "ˌ" { continue }
            out.append(t)
        }
        return out
    }

    // MARK: - Syllabické l̩ / r̩

    private func markSyllabic(_ tokens: [String]) -> [String] {
        var out = tokens
        for i in 0 ..< out.count {
            guard out[i] == "l" || out[i] == "r" else { continue }
            let prev = i > 0 ? out[i - 1] : ""
            let next = i + 1 < out.count ? out[i + 1] : ""
            // Adjacent k samohláske → nie syllabic
            if vowelIPA.contains(prev) || vowelIPA.contains(next) { continue }
            out[i] = out[i] + "\u{0329}"   // kombinovaný syllabic accent
        }
        return out
    }

    // MARK: - Primary + Secondary Stress

    private func applyPrimaryStress(_ tokens: [String]) -> [String] {
        // Nuklei = samohlásky alebo syllabické sonority (končiace na \u0329)
        let nucleiIdx = tokens.indices.filter { i in
            vowelIPA.contains(tokens[i]) || tokens[i].hasSuffix("\u{0329}")
        }
        guard !nucleiIdx.isEmpty else { return tokens }

        var out = tokens
        // Primary stress na 1. nukleus
        out[nucleiIdx[0]] = "ˈ" + out[nucleiIdx[0]]

        // Secondary stress iba ak >= 4 nuklei (na 3., 5., 7. ...)
        if nucleiIdx.count >= 4 {
            for k in stride(from: 2, to: nucleiIdx.count, by: 2) {
                let idx = nucleiIdx[k]
                out[idx] = "ˌ" + out[idx]
            }
        }
        return out
    }
}
