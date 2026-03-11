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

  // Initialize state purely, without reading from localStorage during SSR
  const [user, setUser] = useState<CurrentUser>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    console.log(`[useAuthGuard] useEffect running`);
    const token = getStoredToken();
    const cachedUser = token ? getStoredUser() : null;

    if (cachedUser) {
        setUser(cachedUser);
        setIsLoading(false); // Unblock rendering immediately if cache exists
    }

    if (!token) {
      console.log(`[useAuthGuard] No token found, redirecting to login...`);
      clearStoredAuth();
      setStoredUser(null);
      router.replace("/login");
      return;
    }

    let isMounted = true;

    async function validateWithServer() {
      console.log(`[useAuthGuard] Calling /api/auth/me to validate token`);
      try {
        const response = await apiFetch("/api/auth/me");
        console.log(`[useAuthGuard] /api/auth/me responded with status: ${response.status}`);

        if (response.status === 401 || response.status === 403) {
          console.error(`[useAuthGuard] 401/403 received, clearing auth and redirecting...`);
          if (isMounted) {
            clearStoredAuth();
            setStoredUser(null);
            router.replace("/login");
          }
          return;
        }

        if (!response.ok) {
          console.warn(`[useAuthGuard] Server error or network error, keeping cached user...`);
          if (isMounted && !cachedUser) setIsLoading(false);
          return;
        }

        const data = await response.json();
        console.log(`[useAuthGuard] Successfully validated token, updating user state`);
        if (isMounted) {
          const freshUser = data.user ?? null;
          setUser(freshUser);
          setStoredUser(freshUser);
          setIsLoading(false);
        }
      } catch (e) {
        console.error(`[useAuthGuard] Network failure validating token`, e);
        if (isMounted && !cachedUser) setIsLoading(false);
      }
    }

    validateWithServer();

    return () => {
      isMounted = false;
    };
  }, [router]);

  return { user, isLoading };
}
