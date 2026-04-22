// AudioPostProcessor.swift
// VoxoLoxo fork — post-processing pre Kokoro q4 audio výstup.
//
// Reťazec filtrov (zhodný s Python referenciou load_q4_benchmark.py):
//   1. HPF 80 Hz, 4th order Butterworth — odstraňuje DC rumble z mlx_audio
//   2. v7 multi-notch (9592 / 8774 / 10655 Hz) — potláča q4 kvantizačný šum
//   3. Peak normalize na 0.9
//
// Implementácia: priamy tvar II transponovaný (Direct Form II Transposed).
// Causal filtering — zodpovedá Python sosfilt (nie sosfiltfilt).
// Pre TTS audio (celé vety naraz) je causal dostatočný.
//
// Zdroj koeficientov: bilineárna transformácia (HPF) a scipy iirnotch formula (notch).

import Foundation

public struct AudioPostProcessor {

    // MARK: - Biquad Section

    /// Jeden biquad filter: [b0, b1, b2, a1, a2] (a0 = 1, normalizované).
    private struct Biquad {
        let b0, b1, b2: Double
        let a1, a2: Double
    }

    // MARK: - Filter Design

    /// 2nd-order Butterworth HPF sekcia (bilineárna transformácia).
    ///
    /// H_HP(s) = s² / (s² + (ωa/Q)·s + ωa²)
    /// Pre 4th-order Butterworth HP treba 2 sekcie s rôznymi Q:
    ///   Q₁ = 1/(2·sin(π/8))  ≈ 1.30656  (2. a 4. pól)
    ///   Q₂ = 1/(2·sin(3π/8)) ≈ 0.54120  (1. a 3. pól)
    ///
    /// - Parameters:
    ///   - fc: Cutoff frekvencia [Hz]
    ///   - fs: Vzorkovacia frekvencia [Hz]
    ///   - q:  Q faktor danej sekcie
    private static func hpfSection(fc: Double, fs: Double, q: Double) -> Biquad {
        let c = tan(.pi * fc / fs)   // = tan(wc/2) — bilineárny pre-warp
        let d = 1.0 + c / q + c * c
        return Biquad(
            b0:  1.0 / d,
            b1: -2.0 / d,
            b2:  1.0 / d,
            a1:  2.0 * (c * c - 1.0) / d,
            a2:  (1.0 - c / q + c * c) / d
        )
    }

    /// IIR notch biquad — scipy iirnotch formula.
    ///
    /// b0 = 1 / (1 + tan(bw/2))
    /// b  = [b0, -2·b0·cos(w0), b0]
    /// a  = [1,  -2·b0·cos(w0), 2·b0 - 1]
    ///
    /// - Parameters:
    ///   - freq: Notch frekvencia [Hz]
    ///   - Q:    Q faktor (šírka notchu — väčší Q = užší notch)
    ///   - fs:   Vzorkovacia frekvencia [Hz]
    private static func notchSection(freq: Double, Q: Double, fs: Double) -> Biquad {
        let w0 = 2.0 * .pi * freq / fs
        let bw = w0 / Q
        let b0 = 1.0 / (1.0 + tan(bw / 2.0))
        let c  = cos(w0)
        return Biquad(
            b0:  b0,
            b1: -2.0 * b0 * c,
            b2:  b0,
            a1: -2.0 * b0 * c,
            a2:  2.0 * b0 - 1.0
        )
    }

    // MARK: - Filter Chain (lazy static, computed raz pri prvom volaní)

    private static let filterChain: [Biquad] = buildChain()

    private static func buildChain() -> [Biquad] {
        // Q faktory pre 4th-order Butterworth prototype (Butterworth pole angles)
        let q1 = 1.0 / (2.0 * sin(.pi / 8.0))        // ≈ 1.30656
        let q2 = 1.0 / (2.0 * sin(3.0 * .pi / 8.0))  // ≈ 0.54120
        return [
            // HPF 80 Hz, 4th order = 2× 2nd-order sekcie
            hpfSection(fc: 80, fs: 24000, q: q1),
            hpfSection(fc: 80, fs: 24000, q: q2),
            // v7 multi-notch (z spectrum_filter_notch_9592.py)
            notchSection(freq: 9592,  Q: 25, fs: 24000),
            notchSection(freq: 8774,  Q: 35, fs: 24000),
            notchSection(freq: 10655, Q: 35, fs: 24000)
        ]
    }

    // MARK: - Public API

    /// Aplikuje celý filter reťazec na audio samples z Kokoro q4.
    ///
    /// - Parameter samples: Raw float samples zo `KokoroTTS.generateAudio()` (sr = 24 kHz)
    /// - Returns: Filtrované a normalizované samples (peak ≤ 0.9)
    public static func process(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return samples }

        var signal = samples.map { Double($0) }

        for filter in filterChain {
            signal = applyBiquad(filter, to: signal)
        }

        // Peak normalize → 0.9
        let peak = signal.reduce(0.0) { max($0, abs($1)) }
        if peak > 0 {
            let scale = 0.9 / peak
            signal = signal.map { $0 * scale }
        }

        return signal.map { Float($0) }
    }

    // MARK: - Direct Form II Transposed

    /// Aplikuje jeden biquad filter (causal, priamy tvar II transponovaný).
    private static func applyBiquad(_ f: Biquad, to x: [Double]) -> [Double] {
        var w1 = 0.0, w2 = 0.0
        return x.map { xn in
            let yn = f.b0 * xn + w1
            w1 = f.b1 * xn - f.a1 * yn + w2
            w2 = f.b2 * xn - f.a2 * yn
            return yn
        }
    }
}
