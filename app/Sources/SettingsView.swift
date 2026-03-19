import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        TabView {
            FileTypesTab()
                .environmentObject(appState)
                .tabItem { Label("Dateitypen", systemImage: "doc.badge.gearshape") }

            TrainingTab()
                .environmentObject(appState)
                .tabItem { Label("Training", systemImage: "brain") }

            GeneralTab()
                .environmentObject(appState)
                .tabItem { Label("Allgemein", systemImage: "gearshape") }
        }
        .frame(width: 460, height: 380)
    }
}

// MARK: - File Types Tab

struct FileTypesTab: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        Form {
            Section {
                Toggle(isOn: $appState.config.fileTypes.plainText) {
                    VStack(alignment: .leading) {
                        Text("Plain Text")
                        Text(".txt, .md, .csv, .json, .yaml, .xml, .toml")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Toggle(isOn: $appState.config.fileTypes.code) {
                    VStack(alignment: .leading) {
                        Text("Code")
                        Text(".py, .js, .ts, .swift, .rs, .go, .c, .cpp, .java, .rb, .sh")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } header: {
                Text("Textdateien")
            }

            Section {
                Toggle(isOn: $appState.config.fileTypes.richText) {
                    VStack(alignment: .leading) {
                        Text("Rich Text")
                        Text(".rtf, .doc \u{2014} extrahiert via macOS textutil")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Toggle(isOn: $appState.config.fileTypes.pdf) {
                    VStack(alignment: .leading) {
                        Text("PDF")
                        Text(".pdf \u{2014} extrahiert via PyPDF2 oder pdftotext")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Toggle(isOn: $appState.config.fileTypes.office) {
                    VStack(alignment: .leading) {
                        Text("Office")
                        Text(".docx, .odt \u{2014} extrahiert via macOS textutil")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } header: {
                Text("Dokumente")
            }

            Section {
                Toggle(isOn: $appState.config.fileTypes.email) {
                    VStack(alignment: .leading) {
                        Text("E-Mail")
                        Text(".emlx \u{2014} Apple Mail Nachrichten")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } header: {
                Text("Kommunikation")
            } footer: {
                Text("E-Mails k\u{00f6}nnen sensible Inhalte enthalten. Aktiviere diese Option nur wenn du sicher bist.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

// MARK: - Training Tab

struct TrainingTab: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        Form {
            Section {
                Stepper(
                    "Debounce: \(appState.config.training.debounceSeconds) Sekunden",
                    value: $appState.config.training.debounceSeconds,
                    in: 10...120, step: 10
                )
                Text("Wartezeit nach der letzten Datei\u{00e4}nderung bevor Training startet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Stepper(
                    "Min Steps: \(appState.config.training.minSteps)",
                    value: $appState.config.training.minSteps,
                    in: 5...50, step: 5
                )
                Stepper(
                    "Max Steps: \(appState.config.training.maxSteps)",
                    value: $appState.config.training.maxSteps,
                    in: 10...200, step: 10
                )
                Text("Mehr ge\u{00e4}nderte Dateien = mehr Steps pro Mini-Batch.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } header: {
                Text("Continuous Learning")
            }

            Section {
                Toggle("Nightly Training (2 Uhr)", isOn: $appState.config.training.nightlyEnabled)
                Text("500 Steps pro Nacht \u{2014} gr\u{00f6}\u{00df}ere Batches wenn der Mac eingesteckt ist und du schl\u{00e4}fst.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } header: {
                Text("Nightly Training")
            }

            Section {
                HStack {
                    VStack(alignment: .leading) {
                        Text("Training-Speed")
                        Text("80.9 ms/step auf M3 Pro")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("2.15 TFLOPS")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(.tint)
                }
                HStack {
                    VStack(alignment: .leading) {
                        Text("ANE Peak")
                        Text("Maximale FP16-Performance")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("12.79 TFLOPS")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(.tint)
                }
            } header: {
                Text("Hardware")
            }
        }
        .formStyle(.grouped)
    }
}

// MARK: - General Tab

struct GeneralTab: View {
    @EnvironmentObject var appState: AppState
    @State private var showDeleteConfirm = false

    var body: some View {
        Form {
            Section {
                LabeledContent("pai CLI") {
                    Text(appState.config.paiPath)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                LabeledContent("Datenverzeichnis") {
                    Text("~/.local/personal-ai/")
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                LabeledContent("ANE-Training") {
                    Text("~/Code/ANE-Training/")
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            } header: {
                Text("Pfade")
            }

            Section {
                Button("Daten im Finder zeigen") {
                    let path = NSString(string: "~/.local/personal-ai").expandingTildeInPath
                    NSWorkspace.shared.open(URL(fileURLWithPath: path))
                }

                Button("Einrichtung erneut starten...") {
                    appState.config.firstLaunchComplete = false
                    let delegate = NSApp.delegate as? AppDelegate
                    delegate?.showOnboarding()
                }
            } header: {
                Text("Aktionen")
            }

            Section {
                Button("Alle Daten l\u{00f6}schen...", role: .destructive) {
                    showDeleteConfirm = true
                }
                .confirmationDialog(
                    "Alle Personal AI Daten l\u{00f6}schen?",
                    isPresented: $showDeleteConfirm,
                    titleVisibility: .visible
                ) {
                    Button("L\u{00f6}schen", role: .destructive) {
                        deleteAllData()
                    }
                } message: {
                    Text("Corpus, Trainings-Daten und Checkpoints werden gel\u{00f6}scht. Diese Aktion kann nicht r\u{00fc}ckg\u{00e4}ngig gemacht werden.")
                }
            } header: {
                Text("Gefahrenzone")
            }
        }
        .formStyle(.grouped)
    }

    private func deleteAllData() {
        let dataDir = NSString(string: "~/.local/personal-ai").expandingTildeInPath
        let filesToDelete = [
            "corpus.jsonl", "training_data.bin", "checkpoint.bin",
            "watcher_state.json", "learn_state.json", "train.log", "learn.log"
        ]
        for file in filesToDelete {
            try? FileManager.default.removeItem(atPath: "\(dataDir)/\(file)")
        }
        appState.refresh()
    }
}
