// G2PFactory.swift
// VoxoLoxo fork — pridaný .czech case pre CzechG2PProcessor (MIT, bez GPL).

import Foundation

/// Available grapheme-to-phoneme engines.
public enum G2P {
    /// MisakiSwift-based G2P engine for English text.
    case misaki
    /// eSpeak NG-based G2P engine supporting multiple languages.
    case eSpeakNG
    /// Native Czech rule-based G2P engine (VoxoLoxo fork, MIT).
    /// Compact IPA matching eSpeak/phonemizer training format.
    case czech
}

/// Factory class for creating G2P processor instances.
final class G2PFactory {
    enum G2PError: Error {
        case noSuchEngine
    }

    private init() {}

    static func createG2PProcessor(engine: G2P) throws -> G2PProcessor {
        switch engine {

        case .misaki:
            #if canImport(MisakiSwift)
            return MisakiG2PProcessor()
            #else
            throw G2PError.noSuchEngine
            #endif

        case .eSpeakNG:
            #if canImport(eSpeakNGLib)
            return eSpeakNGG2PProcessor()
            #else
            throw G2PError.noSuchEngine
            #endif

        case .czech:
            return CzechG2PProcessor()
        }
    }
}
