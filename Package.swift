// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "IndexTTS2",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "IndexTTS2Kit", targets: ["IndexTTS2Kit"]),
        .executable(name: "indextts2", targets: ["indextts2-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.0"),
    ],
    targets: [
        .target(
            name: "IndexTTS2Kit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "indextts2-cli",
            dependencies: ["IndexTTS2Kit"]
        ),
    ]
)
