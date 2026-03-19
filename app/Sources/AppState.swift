import SwiftUI
import Combine

class AppState: ObservableObject {
    static let shared = AppState()

    // Configuration (persisted to config.json)
    @Published var config: PAIConfig {
        didSet { config.save() }
    }

    // Live status
    @Published var isLearning = false
    @Published var totalSteps = 0
    @Published var totalBatches = 0
    @Published var lastTrainTime: String?
    @Published var corpusDocuments = 0
    @Published var corpusSizeMB: Double = 0
    @Published var trainingTokens = 0

    private var refreshTimer: Timer?
    private let dataDir = NSString(string: "~/.local/personal-ai").expandingTildeInPath

    init() {
        self.config = PAIConfig.load()
        refresh()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
            self?.refresh()
        }
    }

    deinit {
        refreshTimer?.invalidate()
    }

    // MARK: - Setup

    var isFirstLaunch: Bool {
        !config.firstLaunchComplete
    }

    func completeSetup() {
        config.firstLaunchComplete = true
    }

    // MARK: - Source Management

    func toggleSource(at index: Int) {
        guard index < config.sources.count else { return }
        config.sources[index].enabled.toggle()
    }

    func addCustomSource(name: String, path: String) {
        let source = KnowledgeSource(
            name: name, path: path, enabled: true,
            icon: "folder.fill", isCustom: true
        )
        config.sources.append(source)
    }

    func removeSource(at index: Int) {
        guard index < config.sources.count, config.sources[index].isCustom else { return }
        config.sources.remove(at: index)
    }

    func enableAllSources() {
        for i in config.sources.indices {
            config.sources[i].enabled = true
        }
    }

    // MARK: - Learning Control

    func startLearning() {
        guard let paiPath = findPai() else { return }

        DispatchQueue.global(qos: .userInitiated).async {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/bin/bash")
            task.arguments = [paiPath, "learn", "--daemon"]
            try? task.run()
            task.waitUntilExit()

            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                self.refresh()
            }
        }
    }

    func stopLearning() {
        guard let paiPath = findPai() else { return }

        DispatchQueue.global(qos: .userInitiated).async {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/bin/bash")
            task.arguments = [paiPath, "learn", "--stop"]
            try? task.run()
            task.waitUntilExit()

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.refresh()
            }
        }
    }

    // MARK: - Refresh

    func refresh() {
        checkDaemonStatus()
        loadLearnState()
        loadCorpusStats()
    }

    private func findPai() -> String? {
        let configured = NSString(string: config.paiPath).expandingTildeInPath
        if FileManager.default.isExecutableFile(atPath: configured) {
            return configured
        }
        let candidates = [
            NSString(string: "~/bin/pai").expandingTildeInPath,
            "/usr/local/bin/pai",
        ]
        for path in candidates {
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }
        return nil
    }

    private func checkDaemonStatus() {
        let pidPath = "\(dataDir)/learn.pid"
        guard FileManager.default.fileExists(atPath: pidPath),
              let pidStr = try? String(contentsOfFile: pidPath, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
              let pid = Int32(pidStr) else {
            DispatchQueue.main.async { self.isLearning = false }
            return
        }
        let running = kill(pid, 0) == 0
        DispatchQueue.main.async { self.isLearning = running }
    }

    private func loadLearnState() {
        let statePath = "\(dataDir)/learn_state.json"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: statePath)),
              let state = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        DispatchQueue.main.async {
            self.totalSteps = state["total_steps"] as? Int ?? 0
            self.totalBatches = state["total_batches"] as? Int ?? 0
            self.lastTrainTime = state["last_train_time"] as? String
        }
    }

    private func loadCorpusStats() {
        let corpusPath = "\(dataDir)/corpus.jsonl"
        let tokenDataPath = "\(dataDir)/training_data.bin"

        DispatchQueue.global(qos: .utility).async {
            var docs = 0
            var sizeMB: Double = 0
            var tokens = 0

            if let attrs = try? FileManager.default.attributesOfItem(atPath: corpusPath) {
                let size = attrs[.size] as? Int64 ?? 0
                sizeMB = Double(size) / 1_000_000
                if let content = try? String(contentsOfFile: corpusPath, encoding: .utf8) {
                    docs = content.components(separatedBy: "\n").filter { !$0.isEmpty }.count
                }
            }

            if let attrs = try? FileManager.default.attributesOfItem(atPath: tokenDataPath) {
                let size = attrs[.size] as? Int64 ?? 0
                tokens = Int(size / 2)
            }

            DispatchQueue.main.async {
                self.corpusDocuments = docs
                self.corpusSizeMB = sizeMB
                self.trainingTokens = tokens
            }
        }
    }

    // MARK: - Helpers

    var lastActivityFormatted: String {
        guard let timeStr = lastTrainTime else { return "—" }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let date = formatter.date(from: timeStr) else {
            // Try without fractional seconds
            formatter.formatOptions = [.withInternetDateTime]
            guard let date = formatter.date(from: timeStr) else {
                // Try simple format
                let df = DateFormatter()
                df.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
                guard let d = df.date(from: timeStr) else { return timeStr.prefix(19).description }
                return RelativeDateTimeFormatter().localizedString(for: d, relativeTo: Date())
            }
            return RelativeDateTimeFormatter().localizedString(for: date, relativeTo: Date())
        }
        return RelativeDateTimeFormatter().localizedString(for: date, relativeTo: Date())
    }
}
