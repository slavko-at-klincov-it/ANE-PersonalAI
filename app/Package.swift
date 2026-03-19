// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PersonalAI",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "PersonalAI",
            path: "Sources"
        )
    ]
)
