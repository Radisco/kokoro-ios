// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "KokoroSwift",
  platforms: [
    .iOS(.v18), .macOS(.v15)
  ],
  products: [
    .library(
      name: "KokoroSwift",
      // VoxoLoxo fork: odstránené .dynamic — default static linking.
      // Dôvod: AUv3 extension target v Xcode SPM-dynamic libs
      // nevloží do .appex bundlu bez explicit Embed Frameworks phase,
      // čo spôsobí dyld load error pri spúšťaní extensie.
      targets: ["KokoroSwift"]
    ),
  ],
  dependencies: [
    // VoxoLoxo fork: používa Radisco/mlx-swift voxo-static branch (static libs pre AUv3)
    .package(url: "https://github.com/Radisco/mlx-swift", branch: "voxo-static"),
    // .package(url: "https://github.com/mlalma/eSpeakNGSwift", from: "1.0.1"),
    .package(url: "https://github.com/mlalma/MisakiSwift", exact: "1.0.6"),
    .package(url: "https://github.com/mlalma/MLXUtilsLibrary.git", exact: "0.0.6")
  ],
  targets: [
    .target(
      name: "KokoroSwift",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "MLXFFT", package: "mlx-swift"),
        // .product(name: "eSpeakNGLib", package: "eSpeakNGSwift"),
        .product(name: "MisakiSwift", package: "MisakiSwift"),
        .product(name: "MLXUtilsLibrary", package: "MLXUtilsLibrary")
      ],
      resources: [
       .copy("../../Resources/")
      ]
    ),
    .testTarget(
      name: "KokoroSwiftTests",
      dependencies: ["KokoroSwift"]
    ),
  ]
)
