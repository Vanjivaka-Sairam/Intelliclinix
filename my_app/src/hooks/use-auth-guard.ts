import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch, clearStoredAuth, getStoredToken } from "@/lib/api";

type CurrentUser = {
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
  const data = localStorage.getItem(USER_DATA_KEY);
  return data ? JSON.parse(data) : null;
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
  const [user, setUser] = useState<CurrentUser>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      router.replace("/login");
      return;
    }

    let isMounted = true;

    async function fetchCurrentUser() {
      try {
        const response = await apiFetch("/api/auth/me");
        if (!response.ok) {
          throw new Error("Unauthorized");
        }
        const data = await response.json();
        if (isMounted) {
          setUser(data.user ?? null);
          setStoredUser(data.user ?? null);
          setIsLoading(false);
        }
      } catch {
        if (isMounted) {
          clearStoredAuth();
          setStoredUser(null);
          router.replace("/login");
        }
      }
    }

    fetchCurrentUser();

    return () => {
      isMounted = false;
    };
  }, [router]);

  return { user, isLoading };
}

