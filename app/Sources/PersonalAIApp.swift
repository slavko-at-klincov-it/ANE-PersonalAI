import SwiftUI

@main
struct PersonalAIApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var appState = AppState.shared

    var body: some Scene {
        MenuBarExtra("Personal AI", systemImage: "brain") {
            MenuBarView()
                .environmentObject(appState)
        }
        .menuBarExtraStyle(.window)

        Window("Quellen verwalten", id: "sources") {
            SourcesView()
                .environmentObject(appState)
        }
        .defaultSize(width: 560, height: 500)

        Window("So funktioniert's", id: "how-it-works") {
            HowItWorksView()
        }
        .defaultSize(width: 620, height: 740)

        Settings {
            SettingsView()
                .environmentObject(appState)
        }
    }
}
