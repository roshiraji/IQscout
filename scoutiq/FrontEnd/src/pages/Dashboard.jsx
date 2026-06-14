import VideoUpload from "../components/VideoUpload";
import HeatmapViewer from "../components/HeatmapViewer";
function Dashboard() {
  return (
    <div>
      <h1>ScoutIQ WNBA Dashboard</h1>

      <VideoUpload />
      <HeatmapViewer />

      <p>Dashboard is loading.</p>
    </div>
  );
}

export default Dashboard;