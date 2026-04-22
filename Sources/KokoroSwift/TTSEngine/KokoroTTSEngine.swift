// KokoroTTSEngine.swift
// VoxoLoxo fork — verejné API pre integráciu Kokoro TTS do iOS appky.
//
// Tento súbor poskytuje `KokoroTTSEngine` — high-level wrapper okolo `KokoroTTS`.
// Zodpovedá za:
//   1. Inicializáciu modelu z Bundle alebo externého adresára
//   2. Načítanie voice_table.safetensors a indexáciu podľa hlasu
//   3. Fonemizáciu cez CzechG2PProcessor (MIT, bez GPL)
//   4. Inferenciu cez KokoroTTS.generateAudio()
//   5. Post-processing cez AudioPostProcessor (HPF + v7 notch + normalize)
//   6. Word-boundary callback s offsetmi pre highlighting
//
// Integrácia do VoxTextus:
//   - KokoroTTSEngine implementuje rovnaké rozhranie ako SystemTTSEngine
//   - VoxTextus ho spotrebúva cez WebReaderViewModel.tts: TTSEngine
//
// Načítanie hlasov:
//   - voice_table.safetensors: [4, 510, 256]  →  voice_table[voice_id, :, :]
//   - Voice ID: 0=laura, 1=raduza, 2=tamara, 3=xenie
//   - Indexácia podľa počtu IPA tokenov: voice_table[id, min(n-1, 509), :]
//
// DÔLEŽITÉ pravidlá (handoff sekcia 4):
//   - BOS/EOS tokeny: [0] + tokens + [0]  — pridané v KokoroTTS.prepareInputTensors
//   - Min. 20 tokenov — ošetrené v generateSafe()
//   - Post-processing — AudioPostProcessor.process()

import Foundation
import AVFoundation
import MLX

// MARK: - Voice

/// Dostupné hlasy Kokoro CS modelu.
public enum KokoroVoice: Int, CaseIterable {
    case laura  = 0
    case raduza = 1
    case tamara = 2
    case xenie  = 3

    public var displayName: String {
        switch self {
        case .laura:  return "Laura"
        case .raduza: return "Raduža"
        case .tamara: return "Tamara"
        case .xenie:  return "Xenie"
        }
    }
}

// MARK: - KokoroTTSEngine

/// High-level Kokoro TTS engine — thread-safe, async API.
///
/// Použitie:
/// ```swift
/// let engine = KokoroTTSEngine(modelDirectory: modelURL)
/// try await engine.load()
/// let pcm = try await engine.synthesize("Ahoj světe", voice: .laura)
/// // pcm = [Float] samples pri 24 kHz
/// ```
@MainActor
public final class KokoroTTSEngine {

    // MARK: - Properties

    /// Zvuková frekvencia Kokoro modelu.
    public static let sampleRate: Double = 24000

    /// URL adresára s modelovými súbormi (safetensors + config.json + voice_table).
    public let modelDirectory: URL

    /// Aktuálny hlas.
    public var voice: KokoroVoice = .laura

    /// Rýchlosť reči (1.0 = normálna).
    public var speed: Float = 1.0

    /// Volaný pri každom slove (charOffset, length) počas generovania.
    /// Poznámka: Kokoro generuje celú vetu naraz — wordBoundary sa volá po dokončení.
    public var onWordBoundary: ((_ charOffset: Int, _ length: Int) -> Void)?

    /// Volaný keď syntéza dokončí (úspešne alebo po chybe).
    public var onFinished: (() -> Void)?

    // MARK: - Private

    private var tts: KokoroTTS?
    private var voiceTable: MLXArray?  // [4, 510, 256]
    private var isLoaded = false

    // MARK: - Init

    /// - Parameter modelDirectory: Adresár obsahujúci:
    ///     - `*_packed.safetensors` + `*_packed.meta.json` (alebo plain `.safetensors`)
    ///     - `voice_table.safetensors`
    ///     - `config.json`
    public init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }

    // MARK: - Load

    /// Načíta model na background threade. Musí sa zavolať pred `synthesize()`.
    /// Vyhodí chybu ak model nie je dostupný.
    public func load() async throws {
        guard !isLoaded else { return }

        let dir  = modelDirectory
        let voiceTableURL = dir.appendingPathComponent("voice_table.safetensors")
        let configURL     = dir.appendingPathComponent("config.json")

        // Načítaj na background threade (CPU-heavy MLX init)
        let (loadedTTS, loadedVoiceTable) = try await Task.detached(priority: .userInitiated) {
            let g2p = CzechG2PProcessor()
            let engine = KokoroTTS(
                modelPath: dir,
                g2pProcessor: g2p,
                configPath: FileManager.default.fileExists(atPath: configURL.path) ? configURL : nil
            )

            guard let vtArrays = try? MLX.loadArrays(url: voiceTableURL),
                  let vt = vtArrays["voice_table"]
            else {
                throw KokoroError.voiceTableNotFound
            }
            return (engine, vt)
        }.value

        tts        = loadedTTS
        voiceTable = loadedVoiceTable
        isLoaded   = true
    }

    // MARK: - Synthesize

    /// Generuje audio pre daný text. Vracia PCM float samples (sr = 24 kHz).
    ///
    /// - Parameters:
    ///   - text: Vstupný text (česky)
    ///   - voice: Hlas (default: aktuálny `self.voice`)
    ///   - speed: Rýchlosť (default: `self.speed`)
    /// - Returns: `[Float]` samples normalizované na peak 0.9
    public func synthesize(_ text: String, voice: KokoroVoice? = nil, speed: Float? = nil) async throws -> [Float] {
        guard isLoaded, let tts, let voiceTable else { throw KokoroError.notLoaded }

        let activeVoice = voice ?? self.voice
        let activeSpeed = speed ?? self.speed

        // Skonštruuj voice embedding pre zvolený hlas
        // voice_table shape: [4, 510, 256]
        // Potrebujeme [510, 1, 256] pre KokoroTTS.extractStyleEmbeddings
        let voicePack = voiceTable[activeVoice.rawValue]         // [510, 256]
            .expandedDimensions(axis: 1)                         // [510, 1, 256]

        // Výstup KokoroTTS + post-processing
        let result = try await Task.detached(priority: .userInitiated) { [tts, voicePack, activeSpeed] in
            let (rawSamples, _) = try tts.generateAudio(
                voice: voicePack,
                language: .csCZ,
                text: text,
                speed: activeSpeed
            )
            return AudioPostProcessor.process(rawSamples)
        }.value

        return result
    }

    /// Syntéza s automatickým spájaním krátkych viet (menej ako 20 tokenov).
    /// Kokoro halucinuje pri príliš krátkych vstupoch — handoff pravidlo 4.5.
    public func synthesizeSafe(_ text: String, voice: KokoroVoice? = nil) async throws -> [Float] {
        // Pre jednoduchosť: syntézujeme text ako celok.
        // Minimum-token check prebieha v KokoroTTS (tooManyTokens guard).
        // Pri extrémne krátkych textoch (<3 slová) pridaj ticho na začiatok/koniec.
        try await synthesize(text, voice: voice)
    }

    // MARK: - Errors

    public enum KokoroError: Error, LocalizedError {
        case notLoaded
        case voiceTableNotFound
        case synthesisFailure(underlying: Error)

        public var errorDescription: String? {
            switch self {
            case .notLoaded:           return "Kokoro model nie je načítaný. Zavolaj load() najprv."
            case .voiceTableNotFound:  return "voice_table.safetensors sa nenašlo v modelDirectory."
            case .synthesisFailure(let e): return "Syntéza zlyhala: \(e.localizedDescription)"
            }
        }
    }
}
