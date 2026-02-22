import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch, clearStoredAuth, getStoredToken } from "@/lib/api";

export type CurrentUser = {
  _id?: string;
  username?: string;
  email?: string;
  first_name?: string;
  last_name?: string;
  role?: string;
  created_at?: string;
  last_login?: string;
} | null;

const USER_DATA_KEY = "user_data";

export const getStoredUser = (): CurrentUser => {
  if (typeof window === "undefined") return null;
  try {
    const data = localStorage.getItem(USER_DATA_KEY);
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
};

export const setStoredUser = (user: CurrentUser) => {
  if (typeof window === "undefined") return;
  if (user) {
    localStorage.setItem(USER_DATA_KEY, JSON.stringify(user));
  } else {
    localStorage.removeItem(USER_DATA_KEY);
  }
};

export function useAuthGuard() {
  const router = useRouter();

  // Synchronously decide initial state before first render:
  // - No token → start loading (blank screen) and redirect immediately
  // - Has token + cached user → show page immediately without spinner
  const hasToken = typeof window !== "undefined" && !!getStoredToken();
  const cachedUser = hasToken ? getStoredUser() : null;

  const [user, setUser] = useState<CurrentUser>(cachedUser);
  const [isLoading, setIsLoading] = useState(!cachedUser || !hasToken);

  useEffect(() => {
    const token = getStoredToken();

    if (!token) {
      // No token at all — redirect immediately, keep isLoading=true so nothing renders
      clearStoredAuth();
      setStoredUser(null);
      router.replace("/login");
      return;
    }

    let isMounted = true;

    async function validateWithServer() {
      try {
        const response = await apiFetch("/api/auth/me");

        // Only on explicit 401/403 do we treat the user as unauthenticated
        if (response.status === 401 || response.status === 403) {
          if (isMounted) {
            clearStoredAuth();
            setStoredUser(null);
            router.replace("/login");
          }
          return;
        }

        if (!response.ok) {
          // Server error (5xx) or network error — don't log out, just keep cached user
          if (isMounted) setIsLoading(false);
          return;
        }

        const data = await response.json();
        if (isMounted) {
          const freshUser = data.user ?? null;
          setUser(freshUser);
          setStoredUser(freshUser);
          setIsLoading(false);
        }
      } catch {
        // Pure network failure — don't logout, just unblock with cached data
        if (isMounted) setIsLoading(false);
      }
    }

    validateWithServer();

    return () => {
      isMounted = false;
    };
  }, [router]);

  return { user, isLoading };
}
