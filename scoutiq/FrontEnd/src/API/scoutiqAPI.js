/**
 * ScoutIQ API client — Anne's integration layer, wired to FastAPI (/api via Vite proxy).
 * Replaces the Flask :5000 stubs from f-Anne_ReactComponent.
 */

const API_BASE = "/api";

async function parseJsonResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg || String(item)).join(", ")
      : detail || data.error || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

/** @deprecated Anne's upload stub — use analyzeVideo instead. */
export async function uploadVideo(videoFile) {
  const result = await analyzeVideo(videoFile);
  return {
    players: result.job?.result?.snapshot?.players || [],
    job_id: result.job_id,
    validation: result.validation,
  };
}

/**
 * Upload + validate + enqueue full-clip tracking (XHR for upload progress).
 */
export function analyzeVideo(
  file,
  gameContext = {},
  { onUploadProgress, runSnapshot = false, runLabeling = false } = {}
) {
  const formData = new FormData();
  formData.append("file", file);
  Object.entries(gameContext).forEach(([key, value]) => {
    if (value) formData.append(key, value);
  });

  const params = new URLSearchParams({
    run_snapshot: String(runSnapshot),
    run_labeling: String(runLabeling),
  });

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/analyze?${params}`);

    if (onUploadProgress) {
      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) return;
        onUploadProgress({
          loaded: event.loaded,
          total: event.total,
          phase: event.loaded < event.total ? "uploading" : "validating",
        });
      };
      xhr.upload.onload = () => {
        onUploadProgress({ loaded: 1, total: 1, phase: "validating" });
      };
    }

    xhr.onload = () => {
      let body = {};
      try {
        body = JSON.parse(xhr.responseText || "{}");
      } catch {
        body = {};
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body);
        return;
      }
      const detail = body.detail;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg || String(item)).join(", ")
        : detail || "Analysis request failed.";
      reject(new Error(message));
    };
    xhr.onerror = () => reject(new Error("Network error during upload."));
    xhr.send(formData);
  });
}

export async function getJob(jobId) {
  const response = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`);
  return parseJsonResponse(response);
}

export async function getConfirmStatus(jobId) {
  const response = await fetch(
    `${API_BASE}/jobs/${encodeURIComponent(jobId)}/confirm-status`
  );
  return parseJsonResponse(response);
}

export async function getIsolateStatus(jobId, generation) {
  const params = generation
    ? `?generation=${encodeURIComponent(generation)}`
    : "";
  const response = await fetch(
    `${API_BASE}/jobs/${encodeURIComponent(jobId)}/isolate-status${params}`
  );
  return parseJsonResponse(response);
}

/** @deprecated Anne's track stub — use saveTrackingSelection instead. */
export async function startTracking(selectedPlayers) {
  return saveTrackingSelection({
    selected_players: selectedPlayers,
  });
}

export async function saveTrackingSelection({
  job_id,
  selected_person_group_ids,
  selected_players,
}) {
  const response = await fetch(`${API_BASE}/tracking/selection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_id,
      selected_person_group_ids,
      selected_players,
    }),
  });
  return parseJsonResponse(response);
}

export async function getIdentity(jobId) {
  const response = await fetch(
    `${API_BASE}/identity/${encodeURIComponent(jobId)}`
  );
  return parseJsonResponse(response);
}

export async function resolveIdentity(jobId, payload) {
  const response = await fetch(
    `${API_BASE}/identity/${encodeURIComponent(jobId)}/resolve`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  return parseJsonResponse(response);
}

export async function setManualJersey(jobId, payload) {
  const response = await fetch(
    `${API_BASE}/identity/${encodeURIComponent(jobId)}/manual-jersey`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  return parseJsonResponse(response);
}

export async function setOffenseTeam(jobId, { mode = "manual", offense_team_id = null } = {}) {
  const response = await fetch(
    `${API_BASE}/jobs/${encodeURIComponent(jobId)}/offense-team`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        offense_team_id: mode === "auto" ? null : offense_team_id,
      }),
    }
  );
  return parseJsonResponse(response);
}

export function jobFileUrl(jobId, kind, query = "") {
  const q = query ? (query.startsWith("?") ? query : `?${query}`) : "";
  return `${API_BASE}/jobs/${encodeURIComponent(jobId)}/files/${kind}${q}`;
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return parseJsonResponse(response);
}

/** Ask the desktop launcher / API to shut down local services. */
export async function requestShutdown() {
  const response = await fetch(`${API_BASE}/shutdown`, { method: "POST" });
  return parseJsonResponse(response);
}
