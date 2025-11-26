const DEFAULT_API_BASE = "http://localhost:5001";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_API_BASE;

type ApiFetchOptions = RequestInit & {
  auth?: boolean;
};

const isBrowser = typeof window !== "undefined";

export const getStoredToken = (): string | null => {
  if (!isBrowser) {
    return null;
  }
  return localStorage.getItem("token");
};

export const setStoredToken = (token: string) => {
  if (!isBrowser) {
    return;
  }
  localStorage.setItem("token", token);
};

export const clearStoredAuth = () => {
  if (!isBrowser) {
    return;
  }
  localStorage.removeItem("token");
  localStorage.removeItem("username");
  localStorage.removeItem("session");
};

export async function apiFetch(
  path: string,
  options: ApiFetchOptions = {}
): Promise<Response> {
  const { auth = true, ...rest } = options;
  const headers = new Headers(rest.headers || {});

  if (auth) {
    const token = getStoredToken();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }

  const fetchOptions: RequestInit = {
    ...rest,
    headers,
  };

  const url = path.startsWith("http")
    ? path
    : `${API_BASE_URL}${path}`;

  return fetch(url, fetchOptions);
}

