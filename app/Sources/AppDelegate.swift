import SwiftUI
import AppKit

class AppDelegate: NSObject, NSApplicationDelegate {
    private var onboardingWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        if !AppState.shared.config.firstLaunchComplete {
            showOnboarding()
        }
    }

    func showOnboarding() {
        let view = OnboardingView(dismiss: { [weak self] in
            self?.onboardingWindow?.close()
            self?.onboardingWindow = nil
        })
        .environmentObject(AppState.shared)

        let controller = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: controller)
        window.title = "Personal AI — Einrichtung"
        window.setContentSize(NSSize(width: 560, height: 680))
        window.styleMask = [.titled, .closable, .fullSizeContentView]
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        onboardingWindow = window
    }
}
