// WeightLoader.swift
// VoxoLoxo fork — rozšírené o packed int4 safetensors loader.
//
// Podporuje dva formáty v modelPath adresári:
//   A. Packed q4  (*_packed.safetensors + *_packed.meta.json) — 49.9 MB Kokoro CS
//   B. Plain fp32 / bf16 safetensors — pôvodný anglický model
//
// Pri packed q4 sa najprv dequantizujú váhy (MLX.dequantized), potom sa aplikuje
// rovnaká sanitizácia ako pri plain formáte (transpozícia F0/N proj, weight_v, atď.)

import Foundation
import MLX
import MLXNN

// MARK: - Q4 Meta JSON štruktúra

/// Metadata popis jedného kvatizovaného tenzora.
private struct Q4TensorInfo: Decodable {
    let groupSize:    Int
    let reshapeMode:  String          // "default" alebo "transpose"
    let pad:          Int
    let originalShape: [Int]
    let originalDtype: String         // "mlx.core.float32" / "mlx.core.bfloat16"

    enum CodingKeys: String, CodingKey {
        case groupSize    = "group_size"
        case reshapeMode  = "reshape_mode"
        case pad          = "pad"
        case originalShape = "original_shape"
        case originalDtype = "original_dtype"
    }
}

/// Vrchná štruktúra meta.json súboru.
private struct Q4Meta: Decodable {
    let bits:        Int
    let quantized:   [String: Q4TensorInfo]
    let skipped1d:   [String]?
    let skippedNoGroup: [String: Q4TensorInfo]?

    // Backwards-compat: starší formát používal "skipped" dict
    let skippedLegacy: [String: Q4TensorInfo]?

    enum CodingKeys: String, CodingKey {
        case bits        = "bits"
        case quantized   = "quantized"
        case skipped1d   = "skipped_1d"
        case skippedNoGroup = "skipped_no_group"
        case skippedLegacy  = "skipped"
    }
}

// MARK: - WeightLoader

/// Utility class for loading and preprocessing neural network weights.
///
/// WeightLoader handles the loading of model weights from disk and applies necessary
/// transformations to ensure compatibility with the model architecture. This includes:
/// - Filtering out unnecessary weights (e.g., position_ids)
/// - Transposing weight tensors for specific layers
/// - Validating and processing weight shapes
/// - **NEW**: Dequantizing packed int4 safetensors (VoxoLoxo fork)
///
/// The class processes weights for different model components:
/// - BERT encoder weights
/// - Predictor (duration and prosody) weights
/// - Text encoder weights
/// - Decoder weights
final class WeightLoader {
    /// WeightLoader is a utility class with only static methods.
    private init() {}

    /// Načíta a sanitizuje váhy modelu z daného adresára alebo súboru.
    ///
    /// Auto-detekuje formát:
    ///   1. Ak existuje `*_packed.meta.json` → packed int4 → dequantize
    ///   2. Inak → načíta prvý `*.safetensors` ako fp32/bf16
    ///
    /// - Parameter modelPath: URL k adresáru s modelovými súbormi
    /// - Returns: Sanitizovaný slovník meno → MLXArray
    static func loadWeights(modelPath: URL) -> [String: MLXArray] {
        let raw: [String: MLXArray]

        // Pokús sa nájsť packed q4 (meta.json prítomné v adresári)
        if let (packedURL, metaURL) = findPackedQ4Files(in: modelPath) {
            raw = loadPackedQ4(packedURL: packedURL, metaURL: metaURL)
        } else {
            // Fallback: load plain safetensors (fp32/bf16)
            let safetensorsURL = findSafetensors(in: modelPath) ?? modelPath
            raw = (try? MLX.loadArrays(url: safetensorsURL)) ?? [:]
        }

        return sanitize(raw)
    }

    // MARK: - Packed Q4 Loader

    /// Hľadá pár (*_packed.safetensors, *_packed.meta.json) v adresári.
    private static func findPackedQ4Files(in dir: URL) -> (packed: URL, meta: URL)? {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: nil
        ) else { return nil }

        // Hľadáme *.meta.json
        guard let metaURL = contents.first(where: { $0.pathExtension == "json" && $0.lastPathComponent.contains("meta") }) else {
            return nil
        }

        // Zodpovedajúci .safetensors (rovnaké meno bez .meta.json resp. s .safetensors)
        let baseName = metaURL.deletingPathExtension().deletingPathExtension().lastPathComponent
        guard let packedURL = contents.first(where: {
            $0.lastPathComponent == baseName + ".safetensors"
        }) else { return nil }

        return (packedURL, metaURL)
    }

    /// Hľadá ľubovoľný *.safetensors súbor v adresári (pre plain formát).
    private static func findSafetensors(in dir: URL) -> URL? {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: nil
        ) else { return nil }

        // Uprednostni súbory BEZ "_packed" (nezobrazíme packed ako fallback)
        return contents
            .filter { $0.pathExtension == "safetensors" && !$0.lastPathComponent.contains("_packed") }
            .first
            ?? contents.first { $0.pathExtension == "safetensors" }
    }

    /// Načíta packed int4 safetensors + meta.json a dequantizuje naspäť na fp32.
    ///
    /// Algoritmus zodpovedá Python dequantize_packed() v load_q4_benchmark.py:
    ///   1. Pass-through skipped tensorov (1D biasy, malé bez skupiny)
    ///   2. Pre každý kvatizovaný tenzor: dequantize → cast → unpad → reshape
    ///
    /// - Note: Výsledok SA EŠTE SANITIZUJE rovnako ako plain safetensors,
    ///   pretože Swift model očakáva rovnaký layout bez ohľadu na formát vstupu.
    private static func loadPackedQ4(packedURL: URL, metaURL: URL) -> [String: MLXArray] {
        // 1. Načítaj packed safetensors
        guard let packed = try? MLX.loadArrays(url: packedURL) else {
            fatalError("WeightLoader: nedá sa načítať packed q4: \(packedURL.path)")
        }

        // 2. Načítaj meta.json
        guard let metaData = try? Data(contentsOf: metaURL),
              let meta = try? JSONDecoder().decode(Q4Meta.self, from: metaData)
        else {
            fatalError("WeightLoader: nedá sa parsovať meta.json: \(metaURL.path)")
        }

        var weights: [String: MLXArray] = [:]

        // 3. Pass-through skipped tensorov (1D biasy atď.)
        var skippedNames: [String] = meta.skipped1d ?? []
        if let noGroup = meta.skippedNoGroup { skippedNames += Array(noGroup.keys) }
        if let legacy  = meta.skippedLegacy  { skippedNames += Array(legacy.keys)  }
        for name in skippedNames {
            if let arr = packed[name] { weights[name] = arr }
        }

        // 4. Dequantize kvatizovaných tensorov
        let dtypeMap: [String: DType] = [
            "mlx.core.float32":  .float32,
            "mlx.core.float16":  .float16,
            "mlx.core.bfloat16": .bfloat16,
        ]

        for (name, info) in meta.quantized {
            guard let q      = packed[name + ".q"],
                  let scales = packed[name + ".scales"],
                  let biases = packed[name + ".biases"]
            else {
                // Tensor môže byť uložený aj priamo (bez .q/.scales/.biases) ak bol skipped
                if let direct = packed[name] { weights[name] = direct }
                continue
            }

            // Dequantize (bits = 4, group_size z meta)
            var w = MLX.dequantized(
                q,
                scales: scales,
                biases: biases,
                groupSize: info.groupSize,
                bits: meta.bits
            )

            // Cast na originálny dtype
            let targetDtype = dtypeMap[info.originalDtype] ?? .float32
            w = w.asType(targetDtype)

            // Odstrán padding (aplikovaný pri poslednej dimenzii 2D tenzora)
            if info.pad > 0 {
                let lastDim = w.shape.last ?? 0
                w = w[.ellipsis, 0 ..< (lastDim - info.pad)]
            }

            // Inverse reshape (transpose pred reshape ak reshape_mode == "transpose")
            if info.reshapeMode == "transpose" {
                w = w.T
            }
            w = w.reshaped(info.originalShape)

            weights[name] = w
        }

        return weights
    }

    // MARK: - Sanitizácia (rovnaká pre oba formáty)

    /// Sanitizuje váhy — transpozícia a filtrovanie špecifické pre architektúru.
    ///
    /// - BERT: preskočí position_ids
    /// - Predictor: transpozícia F0/N projection; podmienená transpozícia weight_v
    /// - TextEncoder: podmienená transpozícia weight_v
    /// - Decoder: transpozícia noise_convs; podmienená transpozícia weight_v
    private static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (key, value) in weights {

            if key.hasPrefix("bert") {
                if key.contains("position_ids") { continue }
                out[key] = value

            } else if key.hasPrefix("predictor") {
                if key.contains("F0_proj.weight") {
                    out[key] = value.transposed(0, 2, 1)
                } else if key.contains("N_proj.weight") {
                    out[key] = value.transposed(0, 2, 1)
                } else if key.contains("weight_v") {
                    out[key] = checkArrayShape(arr: value) ? value : value.transposed(0, 2, 1)
                } else {
                    out[key] = value
                }

            } else if key.hasPrefix("text_encoder") {
                if key.contains("weight_v") {
                    out[key] = checkArrayShape(arr: value) ? value : value.transposed(0, 2, 1)
                } else {
                    out[key] = value
                }

            } else if key.hasPrefix("decoder") {
                if key.contains("noise_convs"), key.hasSuffix(".weight") {
                    out[key] = value.transposed(0, 2, 1)
                } else if key.contains("weight_v") {
                    out[key] = checkArrayShape(arr: value) ? value : value.transposed(0, 2, 1)
                } else {
                    out[key] = value
                }
            }
        }

        return out
    }

    /// Checks if a 3D weight array has the correct shape and doesn't need transposition.
    private static func checkArrayShape(arr: MLXArray) -> Bool {
        guard arr.shape.count == 3 else { return false }
        let outChannels = arr.shape[0]
        let kH = arr.shape[1]
        let kW = arr.shape[2]
        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }
}
