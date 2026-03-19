import SwiftUI
import AppKit

struct SourcesView: View {
    @EnvironmentObject var appState: AppState
    @State private var showingFolderPicker = false

    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(alignment: .leading, spacing: 4) {
                Text("Wissensquellen")
                    .font(.title2)
                    .bold()
                Text("W\u{00e4}hle welche Ordner Personal AI lesen und daraus lernen soll.")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 20)
            .padding(.top, 20)
            .padding(.bottom, 12)

            Divider()

            // Source List
            List {
                Section("Standard-Quellen") {
                    ForEach(Array(appState.config.sources.enumerated().filter { !$0.element.isCustom }),
                            id: \.element.id) { index, source in
                        SourceListRow(
                            source: source,
                            isEnabled: Binding(
                                get: { appState.config.sources[index].enabled },
                                set: { appState.config.sources[index].enabled = $0 }
                            ),
                            onDelete: nil
                        )
                    }
                }

                let customSources = Array(appState.config.sources.enumerated().filter { $0.element.isCustom })
                if !customSources.isEmpty {
                    Section("Eigene Ordner") {
                        ForEach(customSources, id: \.element.id) { index, source in
                            SourceListRow(
                                source: source,
                                isEnabled: Binding(
                                    get: { appState.config.sources[index].enabled },
                                    set: { appState.config.sources[index].enabled = $0 }
                                ),
                                onDelete: { appState.removeSource(at: index) }
                            )
                        }
                    }
                }
            }
            .listStyle(.inset(alternatesRowBackgrounds: true))

            Divider()

            // Bottom toolbar
            HStack {
                Button(action: addFolder) {
                    Label("Ordner hinzuf\u{00fc}gen...", systemImage: "plus")
                }

                Spacer()

                Button("Alle aktivieren") {
                    appState.enableAllSources()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
        }
    }

    private func addFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "W\u{00e4}hle einen Ordner den Personal AI lesen soll"
        panel.prompt = "Hinzuf\u{00fc}gen"

        if panel.runModal() == .OK, let url = panel.url {
            let path = url.path
            let name = url.lastPathComponent
            // Use ~ prefix if in home directory
            let home = NSString(string: "~").expandingTildeInPath
            let displayPath = path.hasPrefix(home)
                ? "~" + path.dropFirst(home.count)
                : path
            appState.addCustomSource(name: name, path: displayPath)
        }
    }
}

// MARK: - Source List Row

struct SourceListRow: View {
    let source: KnowledgeSource
    @Binding var isEnabled: Bool
    let onDelete: (() -> Void)?

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: source.icon)
                .foregroundStyle(isEnabled ? Color.accentColor : .secondary)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text(source.name)
                    .font(.body)
                Text(source.path)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if !source.exists {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .help("Ordner existiert nicht")
            }

            if let onDelete = onDelete {
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Entfernen")
            }

            Toggle("", isOn: $isEnabled)
                .toggleStyle(.switch)
                .controlSize(.small)
                .labelsHidden()
        }
        .padding(.vertical, 2)
    }
}
