import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { Link, useNavigate } from "react-router-dom";
import { getAiStatus } from "../api/activity";
import "../dashboard.css";
import "../automation.css";
import ThemeToggle from "./ThemeToggle";

const AI_LABELS = {
  ready: "AI engine ready",
  loading: "AI engine loading",
  offline: "AI engine offline",
};

// Checks the GPU model server (Mistral 7B + Stable Diffusion 1.5) every minute.
function useAiStatus() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    let active = true;

    const check = async () => {
      try {
        const data = await getAiStatus();
        if (!active) return;

        if (data.online) setStatus({ state: "ready", data });
        else if (data.status === "loading") setStatus({ state: "loading", data });
        else setStatus({ state: "offline", data });
      } catch {
        if (active) setStatus({ state: "offline", data: null });
      }
    };

    check();
    const timer = setInterval(check, 60000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return status;
}

function aiTooltip(status) {
  const data = status?.data;

  if (!data) return "Start the Colab GPU notebook and set MODEL_SERVER_URL in backend/.env";

  if (status.state !== "ready") return data.message || "The GPU model server is not ready";

  const tuned = (yes) => (yes ? " (fine-tuned)" : "");

  return `${data.text_model}${tuned(data.text_fine_tuned)} + ${data.image_model}${tuned(data.image_fine_tuned)} on ${data.gpu || "GPU"}`;
}

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const aiStatus = useAiStatus();

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <header className="dx-nav">
      <div className="max-w-7xl mx-auto px-5 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <svg
            className="dx-gear"
            viewBox="0 0 24 24"
            fill="none"
            strokeWidth="1.6"
            style={{ stroke: "var(--accent)" }}
          >
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1" />
          </svg>
          <div>
            <h2 className="dx-brand">Marketing AI</h2>
            <p className="dx-tagline">Smart Plans. Premium Posts. Social Growth.</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden md:block">
            {aiStatus && (
              <span
                className={`dx-live${aiStatus.state === "ready" ? "" : " dx-live-off"}`}
                title={aiTooltip(aiStatus)}
              >
                <span className={aiStatus.state === "ready" ? "dx-online" : "dx-offline"} />{" "}
                {AI_LABELS[aiStatus.state]}
              </span>
            )}
          </div>

          {user?.photo && (
            <img src={user.photo} alt={user.name} className="dx-avatar" />
          )}

          <div className="hidden sm:block text-right">
            <p className="dx-user-name">{user?.name}</p>
            <p className="dx-user-email">{user?.email}</p>
          </div>

          <Link to="/scheduled" className="dx-btn-ghost dx-btn-sm">
            🗓️ Scheduled posts
          </Link>

          <ThemeToggle />

          <button onClick={handleLogout} className="dx-btn-ghost dx-btn-sm">
            Logout
          </button>
        </div>
      </div>
    </header>
  );
}
