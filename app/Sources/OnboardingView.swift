import SwiftUI

struct OnboardingView: View {
    @EnvironmentObject var appState: AppState
    var dismiss: (() -> Void)?

    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 12) {
                Image(systemName: "brain")
                    .font(.system(size: 48))
                    .foregroundStyle(.tint)

                Text("Willkommen bei Personal AI")
                    .font(.title)
                    .bold()

                Text("Dein Mac hat eine Neural Engine \u{2014} einen KI-Chip der meistens\nungenutzt bleibt. Personal AI nutzt ihn, um aus deinen Dateien\nzu lernen. Komplett lokal, ohne Cloud, ohne Kosten.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .lineSpacing(2)
            }
            .padding(.top, 32)
            .padding(.bottom, 24)

            Divider()

            // Source Selection
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Was soll deine AI lesen k\u{00f6}nnen?")
                        .font(.headline)
                    Spacer()
                    Button("Alles") {
                        appState.enableAllSources()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                VStack(spacing: 4) {
                    ForEach(Array(appState.config.sources.enumerated()), id: \.element.id) { index, source in
                        SourceToggleRow(
                            source: source,
                            isEnabled: Binding(
                                get: { appState.config.sources[index].enabled },
                                set: { appState.config.sources[index].enabled = $0 }
                            )
                        )
                    }
                }
            }
            .padding(.horizontal, 28)
            .padding(.vertical, 16)

            Divider()

            // Privacy Info
            VStack(alignment: .leading, spacing: 8) {
                Label("Was passiert mit deinen Daten?", systemImage: "lock.shield.fill")
                    .font(.subheadline)
                    .bold()

                VStack(alignment: .leading, spacing: 4) {
                    InfoBullet(text: "Alles bleibt lokal auf deinem Mac")
                    InfoBullet(text: "Keine Cloud, keine Server, kein Internet n\u{00f6}tig")
                    InfoBullet(text: "Sensible Dateien (.env, SSH-Keys, etc.) werden automatisch \u{00fc}bersprungen")
                    InfoBullet(text: "Du kannst jederzeit Quellen deaktivieren oder Daten l\u{00f6}schen")
                }
            }
            .padding(.horizontal, 28)
            .padding(.vertical, 14)

            Spacer()

            // Go Button
            VStack(spacing: 8) {
                Button(action: startLearning) {
                    Text("Los geht's")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                Text("Du kannst die Einstellungen jederzeit \u{00fc}ber das Men\u{00fc}bar-Icon \u{00e4}ndern.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 28)
            .padding(.bottom, 24)
        }
        .frame(width: 540, height: 660)
    }

    private func startLearning() {
        appState.completeSetup()
        appState.startLearning()
        dismiss?()
    }
}

// MARK: - Components

struct SourceToggleRow: View {
    let source: KnowledgeSource
    @Binding var isEnabled: Bool

    var body: some View {
        Toggle(isOn: $isEnabled) {
            HStack(spacing: 10) {
                Image(systemName: source.icon)
                    .foregroundStyle(.tint)
                    .frame(width: 20)

                VStack(alignment: .leading, spacing: 1) {
                    Text(source.name)
                        .font(.body)
                    Text(source.path)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if !source.exists {
                    Text("nicht vorhanden")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }
        }
        .toggleStyle(.switch)
        .controlSize(.small)
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.5)))
    }
}

struct InfoBullet: View {
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
                .padding(.top, 1)
            Text(text)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
