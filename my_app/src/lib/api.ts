const DEFAULT_API_BASE = "http://localhost:5001";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_API_BASE;

type ApiFetchOptions = RequestInit & {
  auth?: boolean;
  _isRetry?: boolean; // internal: prevents infinite refresh loops
};

const isBrowser = typeof window !== "undefined";

// ─── Token storage ──────────────────────────────────────────────────────────

export const getStoredToken = (): string | null => {
  if (!isBrowser) return null;
  return localStorage.getItem("token");
};

export const setStoredToken = (token: string) => {
  if (!isBrowser) return;
  localStorage.setItem("token", token);
};

export const clearStoredAuth = () => {
  if (!isBrowser) return;
  localStorage.removeItem("token");
  localStorage.removeItem("username");
  localStorage.removeItem("session");
  localStorage.removeItem("user_data");
};

// ─── JWT helpers ─────────────────────────────────────────────────────────────

function getTokenExpiry(token: string): number | null {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return typeof payload.exp === "number" ? payload.exp : null;
  } catch {
    return null;
  }
}

/** True if the token expires within the next `thresholdSecs` seconds. */
function tokenExpiresSoon(token: string, thresholdSecs = 300): boolean {
  const exp = getTokenExpiry(token);
  if (exp === null) return false;
  return Date.now() / 1000 + thresholdSecs >= exp;
}

// ─── Token refresh ───────────────────────────────────────────────────────────

let refreshPromise: Promise<string | null> | null = null;

async function refreshAccessToken(): Promise<string | null> {
  // Deduplicate concurrent refresh attempts
  if (refreshPromise) return refreshPromise;

  refreshPromise = (async () => {
    const token = getStoredToken();
    if (!token) return null;
    try {
      const res = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return null;
      const data = await res.json();
      if (data.access_token) {
        setStoredToken(data.access_token);
        return data.access_token;
      }
      return null;
    } catch {
      return null;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

// ─── Core fetch ─────────────────────────────────────────────────────────────

export async function apiFetch(
  path: string,
  options: ApiFetchOptions = {}
): Promise<Response> {
  const { auth = true, _isRetry = false, ...rest } = options;
  const headers = new Headers(rest.headers || {});

  if (auth) {
    let token = getStoredToken();

    // Proactively refresh if the token is about to expire
    if (token && tokenExpiresSoon(token) && !_isRetry) {
      const newToken = await refreshAccessToken();
      if (newToken) token = newToken;
    }

    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }

  const url = path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
  const response = await fetch(url, { ...rest, headers });

  // Auto-refresh on 401 (once) — covers race conditions where the token
  // expired between the proactive check and the server receiving the request
  if (response.status === 401 && auth && !_isRetry) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      // Retry original request with fresh token
      return apiFetch(path, { ...options, _isRetry: true });
    }
    // Refresh failed → token is truly invalid, clear auth
    clearStoredAuth();
    if (isBrowser) window.location.href = "/login";
  }

  return response;
}

// ─── Logout helper ───────────────────────────────────────────────────────────

export async function logout(): Promise<void> {
  try {
    await apiFetch("/api/auth/logout", { method: "POST" });
  } catch {
    // Ignore — we always clear locally
  } finally {
    clearStoredAuth();
    if (isBrowser) window.location.href = "/login";
  }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

/** True when the response is a definitive auth failure (not a network error). */
export function isAuthError(response: Response): boolean {
  return response.status === 401 || response.status === 403;
}
