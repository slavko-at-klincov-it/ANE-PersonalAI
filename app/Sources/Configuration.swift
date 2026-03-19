import Foundation

// MARK: - Configuration Model

struct PAIConfig: Codable {
    var sources: [KnowledgeSource]
    var fileTypes: FileTypeConfig
    var training: TrainingConfig
    var firstLaunchComplete: Bool
    var paiPath: String

    static let defaultConfig = PAIConfig(
        sources: KnowledgeSource.defaults,
        fileTypes: .default,
        training: .default,
        firstLaunchComplete: false,
        paiPath: "~/bin/pai"
    )

    static var configDir: String {
        NSString(string: "~/.local/personal-ai").expandingTildeInPath
    }

    static var configPath: String {
        "\(configDir)/config.json"
    }

    static func load() -> PAIConfig {
        let path = configPath
        guard FileManager.default.fileExists(atPath: path),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let config = try? JSONDecoder().decode(PAIConfig.self, from: data) else {
            return .defaultConfig
        }
        return config
    }

    func save() {
        let dir = PAIConfig.configDir
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(self) {
            try? data.write(to: URL(fileURLWithPath: PAIConfig.configPath))
        }
    }
}

// MARK: - Knowledge Source

struct KnowledgeSource: Codable, Identifiable, Hashable {
    var id: String { path }
    var name: String
    var path: String
    var enabled: Bool
    var icon: String
    var isCustom: Bool

    static let defaults: [KnowledgeSource] = [
        KnowledgeSource(name: "Dokumente", path: "~/Documents", enabled: true,
                        icon: "doc.text.fill", isCustom: false),
        KnowledgeSource(name: "Code", path: "~/Code", enabled: true,
                        icon: "chevron.left.forwardslash.chevron.right", isCustom: false),
        KnowledgeSource(name: "Desktop", path: "~/Desktop", enabled: true,
                        icon: "desktopcomputer", isCustom: false),
        KnowledgeSource(name: "Notizen", path: "~/Notes", enabled: true,
                        icon: "note.text", isCustom: false),
        KnowledgeSource(name: "Mail", path: "~/Library/Mail", enabled: false,
                        icon: "envelope.fill", isCustom: false),
        KnowledgeSource(name: "Downloads", path: "~/Downloads", enabled: false,
                        icon: "arrow.down.circle.fill", isCustom: false),
    ]

    var expandedPath: String {
        NSString(string: path).expandingTildeInPath
    }

    var exists: Bool {
        FileManager.default.fileExists(atPath: expandedPath)
    }
}

// MARK: - File Type Configuration

struct FileTypeConfig: Codable {
    var plainText: Bool
    var code: Bool
    var richText: Bool
    var pdf: Bool
    var office: Bool
    var email: Bool

    static let `default` = FileTypeConfig(
        plainText: true, code: true, richText: true,
        pdf: true, office: true, email: false
    )
}

// MARK: - Training Configuration

struct TrainingConfig: Codable {
    var debounceSeconds: Int
    var minSteps: Int
    var maxSteps: Int
    var nightlyEnabled: Bool

    static let `default` = TrainingConfig(
        debounceSeconds: 30, minSteps: 10, maxSteps: 50, nightlyEnabled: false
    )
}
