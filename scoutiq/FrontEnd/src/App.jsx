import { useCallback, useEffect, useState } from "react";
import Dashboard from "./pages/Dashboard";
import BootScreen from "./components/BootScreen";
import { requestShutdown } from "./API/scoutiqAPI";

function requestAppFullscreen() {
  const root = document.documentElement;
  if (!root || document.fullscreenElement) return;
  const req =
    root.requestFullscreen ||
    root.webkitRequestFullscreen ||
    root.msRequestFullscreen;
  if (typeof req === "function") {
    try {
      const result = req.call(root);
      if (result && typeof result.catch === "function") {
        result.catch(() => {
          // Browser may require a gesture; launcher --start-fullscreen covers app mode.
        });
      }
    } catch {
      // Ignore — Chromium app flags still handle fullscreen.
    }
  }
}

function App() {
  const [ready, setReady] = useState(false);

  const handleReady = useCallback(() => {
    setReady(true);
  }, []);

  useEffect(() => {
    // Enter fullscreen as soon as the window loads (app-mode / allowed hosts).
    requestAppFullscreen();
    const onFirstInteract = () => {
      requestAppFullscreen();
      window.removeEventListener("pointerdown", onFirstInteract);
      window.removeEventListener("keydown", onFirstInteract);
    };
    window.addEventListener("pointerdown", onFirstInteract, { once: true });
    window.addEventListener("keydown", onFirstInteract, { once: true });
    return () => {
      window.removeEventListener("pointerdown", onFirstInteract);
      window.removeEventListener("keydown", onFirstInteract);
    };
  }, []);

  async function handleQuit() {
    try {
      if (document.fullscreenElement && document.exitFullscreen) {
        await document.exitFullscreen().catch(() => {});
      }
    } catch {
      // Continue quitting even if fullscreen exit fails.
    }
    try {
      await requestShutdown();
    } catch {
      // Launcher still watches the window; closing is enough.
    }
    window.setTimeout(() => {
      window.close();
    }, 200);
  }

  if (!ready) {
    return <BootScreen onReady={handleReady} />;
  }

  return (
    <>
      <button
        type="button"
        className="app-quit-button"
        onClick={handleQuit}
        title="Quit ScoutIQ and stop the local server"
      >
        Quit
      </button>
      <Dashboard />
    </>
  );
}

export default App;
