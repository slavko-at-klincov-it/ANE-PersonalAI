import SwiftUI

struct MenuBarView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.openWindow) var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("Personal AI")
                    .font(.headline)
                Spacer()
                StatusBadge(isActive: appState.isLearning)
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 10)

            Divider()

            // Stats
            VStack(alignment: .leading, spacing: 6) {
                if appState.totalBatches > 0 {
                    StatRow(label: "Training heute", value: "\(appState.totalSteps) Steps")
                    StatRow(label: "Mini-Batches", value: "\(appState.totalBatches)")
                    StatRow(label: "Letzte Aktivit\u{00e4}t", value: appState.lastActivityFormatted)
                }

                if appState.corpusDocuments > 0 {
                    if appState.totalBatches > 0 {
                        Spacer().frame(height: 4)
                    }
                    StatRow(label: "Corpus", value: "\(appState.corpusDocuments) Dateien")
                    StatRow(label: "Gr\u{00f6}\u{00df}e", value: String(format: "%.1f MB", appState.corpusSizeMB))
                    if appState.trainingTokens > 0 {
                        StatRow(label: "Tokens", value: formatNumber(appState.trainingTokens))
                    }
                }

                if appState.totalBatches == 0 && appState.corpusDocuments == 0 {
                    Text("Noch keine Daten gesammelt.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Starte Learning um zu beginnen.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            // Control Button
            VStack {
                if appState.isLearning {
                    Button(action: { appState.stopLearning() }) {
                        Label("Learning stoppen", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .controlSize(.large)
                    .buttonStyle(.bordered)
                } else {
                    Button(action: { appState.startLearning() }) {
                        Label("Learning starten", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            // Navigation Links
            VStack(alignment: .leading, spacing: 2) {
                MenuLink(icon: "folder.badge.gearshape", title: "Quellen verwalten...") {
                    openWindow(id: "sources")
                }
                SettingsLink {
                    MenuLinkContent(icon: "gearshape", title: "Einstellungen...")
                }
                MenuLink(icon: "questionmark.circle", title: "So funktioniert's") {
                    openWindow(id: "how-it-works")
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)

            Divider()

            // Quit
            Button(action: { NSApplication.shared.terminate(nil) }) {
                MenuLinkContent(icon: "xmark.circle", title: "Beenden")
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .padding(.bottom, 4)
        }
        .frame(width: 300)
    }

    private func formatNumber(_ n: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = Locale(identifier: "de_DE")
        return formatter.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}

// MARK: - Components

struct StatusBadge: View {
    let isActive: Bool

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(isActive ? .green : .gray)
                .frame(width: 8, height: 8)
            Text(isActive ? "Aktiv" : "Gestoppt")
                .font(.caption)
                .foregroundStyle(isActive ? .green : .secondary)
        }
    }
}

struct StatRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }
}

struct MenuLink: View {
    let icon: String
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            MenuLinkContent(icon: icon, title: title)
        }
        .buttonStyle(.plain)
    }
}

struct MenuLinkContent: View {
    let icon: String
    let title: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .frame(width: 16)
                .foregroundStyle(.secondary)
            Text(title)
                .font(.body)
            Spacer()
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .contentShape(Rectangle())
    }
}
