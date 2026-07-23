import { useCallback, useState } from "react";
import Dashboard from "./pages/Dashboard";
import BootScreen from "./components/BootScreen";
import { requestShutdown } from "./API/scoutiqAPI";

function App() {
  const [ready, setReady] = useState(false);

  const handleReady = useCallback(() => {
    setReady(true);
  }, []);

  async function handleQuit() {
    try {
      await requestShutdown();
    } catch {
      // Launcher still watches the window; closing is enough.
    }
    // App-mode Edge/Chrome windows close via window.close().
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
