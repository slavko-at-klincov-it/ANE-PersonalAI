import SwiftUI

struct HowItWorksView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                // Header
                VStack(alignment: .leading, spacing: 8) {
                    Text("So funktioniert Personal AI")
                        .font(.largeTitle)
                        .bold()
                    Text("Dein Mac lernt aus deinen Dateien \u{2014} komplett lokal, ohne Cloud")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }

                // Pipeline
                VStack(alignment: .leading, spacing: 20) {
                    PipelineStep(
                        icon: "doc.text.magnifyingglass",
                        number: "1",
                        title: "Sammeln",
                        description: "Personal AI \u{00fc}berwacht deine konfigurierten Ordner via macOS FSEvents. Wenn du eine Datei erstellst oder \u{00e4}nderst, wird der Text automatisch extrahiert und in einem lokalen Corpus gespeichert.",
                        detail: "Unterst\u{00fc}tzte Formate: Plain Text, Code, PDF, RTF, DOCX und mehr. Sensible Dateien (.env, SSH-Keys, Passw\u{00f6}rter) werden automatisch \u{00fc}bersprungen."
                    )

                    PipelineStep(
                        icon: "textformat.abc",
                        number: "2",
                        title: "Tokenisieren",
                        description: "Der gesammelte Text wird in Tokens umgewandelt \u{2014} die Sprache die das KI-Modell versteht. Wir nutzen tiktoken (BPE), den gleichen Tokenizer wie GPT-4, f\u{00fc}r effiziente Textkodierung.",
                        detail: "100K Vocabulary, 3.2x effizienter als Character-Level Encoding. Automatische Vocab-Compaction f\u{00fc}r schnelleres Training."
                    )

                    PipelineStep(
                        icon: "brain",
                        number: "3",
                        title: "Training auf der Neural Engine",
                        description: "Die Tokens werden als Trainingsdaten an die Apple Neural Engine (ANE) gesendet. Ein 109M-Parameter Transformer-Modell lernt Muster in deinen Daten \u{2014} mit 80.9ms pro Trainingsschritt.",
                        detail: "Das Training l\u{00e4}uft kontinuierlich: bei jeder Datei\u{00e4}nderung werden 10-50 Steps trainiert. Nachts optional gr\u{00f6}\u{00df}ere Batches (500 Steps)."
                    )

                    PipelineStep(
                        icon: "magnifyingglass",
                        number: "4",
                        title: "Suche & Query",
                        description: "Aktuell: Keyword-Suche in deinem Corpus mit Snippet-Anzeige. Du kannst in deinen gesammelten Daten suchen und relevante Dateien finden.",
                        detail: "Geplant: Semantische Suche und Textgenerierung nach gen\u{00fc}gend Training-Steps."
                    )
                }

                Divider()

                // Neural Engine
                InfoSection(
                    icon: "cpu",
                    title: "Was ist die Neural Engine?",
                    paragraphs: [
                        "Jeder Apple Silicon Mac (M1\u{2013}M4) hat eine Neural Engine \u{2014} einen spezialisierten KI-Chip mit 16 Cores. Er ist f\u{00fc}r Machine Learning optimiert und erreicht 12.79 TFLOPS Peak-Performance (FP16).",
                        "Normalerweise wird die ANE nur von Apples eigenen Apps genutzt (Siri, Fotos, etc.). Personal AI nutzt sie via reverse-engineered private APIs f\u{00fc}r eigenes Training \u{2014} das erste Open-Source Projekt das dies tut."
                    ]
                )

                // Zero Impact
                InfoSection(
                    icon: "leaf.fill",
                    title: "Zero Impact",
                    paragraphs: [
                        "Training l\u{00e4}uft mit QoS=9 (Background) \u{2014} der niedrigsten Systempriorit\u{00e4}t. Das bedeutet:"
                    ]
                )

                VStack(alignment: .leading, spacing: 6) {
                    ZeroImpactRow(icon: "speaker.slash.fill", text: "Kein L\u{00fc}fterger\u{00e4}usch")
                    ZeroImpactRow(icon: "battery.100percent", text: "Kein zus\u{00e4}tzlicher Akku-Verbrauch")
                    ZeroImpactRow(icon: "cpu", text: "CPU und GPU bleiben frei f\u{00fc}r deine Arbeit")
                    ZeroImpactRow(icon: "bolt.fill", text: "Die ANE hat eine eigene Stromversorgung auf dem Chip")
                }
                .padding(.leading, 48)
                .padding(.top, -16)

                // Privacy
                InfoSection(
                    icon: "lock.shield.fill",
                    title: "Datenschutz",
                    paragraphs: [
                        "Deine Daten verlassen niemals deinen Mac. Es gibt keine Cloud-Verbindung, keinen Server, kein Telemetrie. Alles passiert lokal.",
                        "Sensible Dateien werden automatisch \u{00fc}bersprungen: .env, SSH-Keys, Credentials, API-Keys, Passw\u{00f6}rter. Die gesammelten Daten liegen in ~/.local/personal-ai/ und du kannst sie jederzeit l\u{00f6}schen.",
                        "Personal AI braucht kein Internet \u{2014} weder f\u{00fc}r Training noch f\u{00fc}r Inference."
                    ]
                )

                // File Types
                InfoSection(
                    icon: "doc.on.doc.fill",
                    title: "Unterst\u{00fc}tzte Dateitypen",
                    paragraphs: [
                        "Personal AI kann Text aus vielen Formaten extrahieren:"
                    ]
                )

                VStack(alignment: .leading, spacing: 8) {
                    FileTypeRow(category: "Plain Text", extensions: ".txt, .md, .csv, .json, .yaml, .xml")
                    FileTypeRow(category: "Code", extensions: ".py, .js, .ts, .swift, .rs, .go, .c, .cpp")
                    FileTypeRow(category: "Rich Text", extensions: ".rtf, .doc (via macOS textutil)")
                    FileTypeRow(category: "PDF", extensions: ".pdf (via PyPDF2 oder pdftotext)")
                    FileTypeRow(category: "Office", extensions: ".docx, .odt (via macOS textutil)")
                    FileTypeRow(category: "E-Mail", extensions: ".emlx (Apple Mail)")
                }
                .padding(.leading, 48)
                .padding(.top, -16)

                Text("Automatisch \u{00fc}bersprungen: Bilder, Videos, Archive, Bin\u{00e4}rdateien, Dateien > 1MB")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.leading, 48)
                    .padding(.top, -12)

                Divider()

                // Technical Details
                InfoSection(
                    icon: "wrench.and.screwdriver.fill",
                    title: "Technische Details",
                    paragraphs: [
                        "Modell: Stories110M (109.5M Parameter, 12 Layers, 768 Dimensionen, 12 Attention Heads)",
                        "Training: 80.9ms/step, 2.15 TFLOPS Durchsatz, Gradient Accumulation \u{00fc}ber 5 Batches",
                        "Hardware: ANE Peak bei 12.79 TFLOPS (FP16), 10 Kernels einmal kompiliert (520ms), Weights im IOSurface-Spatial-Dimension gepackt (kein Recompile bei Update)"
                    ]
                )

                // Links
                VStack(alignment: .leading, spacing: 4) {
                    Text("Mehr erfahren")
                        .font(.headline)
                    Text("Die vollst\u{00e4}ndige Forschungsdokumentation und der Quellcode liegen im ANE-Training Repository auf GitHub.")
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(28)
        }
    }
}

// MARK: - Components

struct PipelineStep: View {
    let icon: String
    let number: String
    let title: String
    let description: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(0.15))
                    .frame(width: 36, height: 36)
                Image(systemName: icon)
                    .font(.system(size: 16))
                    .foregroundStyle(.tint)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("\(number). \(title)")
                    .font(.headline)
                Text(description)
                    .font(.body)
                    .foregroundStyle(.primary)
                    .lineSpacing(2)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineSpacing(2)
            }
        }
    }
}

struct InfoSection: View {
    let icon: String
    let title: String
    let paragraphs: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .foregroundStyle(.tint)
                Text(title)
                    .font(.headline)
            }

            ForEach(paragraphs, id: \.self) { text in
                Text(text)
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .lineSpacing(2)
                    .padding(.leading, 28)
            }
        }
    }
}

struct ZeroImpactRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(.green)
                .frame(width: 16)
            Text(text)
                .font(.body)
                .foregroundStyle(.secondary)
        }
    }
}

struct FileTypeRow: View {
    let category: String
    let extensions: String

    var body: some View {
        HStack(alignment: .top) {
            Text(category)
                .font(.body)
                .fontWeight(.medium)
                .frame(width: 80, alignment: .leading)
            Text(extensions)
                .font(.body)
                .foregroundStyle(.secondary)
        }
    }
}
