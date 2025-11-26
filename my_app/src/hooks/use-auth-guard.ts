import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch, clearStoredAuth, getStoredToken } from "@/lib/api";

type CurrentUser = {
  _id?: string;
  username?: string;
  email?: string;
} | null;

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
          if (response.status === 404) {
            if (isMounted) {
              setUser(null);
              setIsLoading(false);
            }
            return;
          }
          throw new Error("Unauthorized");
        }
        const data = await response.json();
        if (isMounted) {
          setUser(data.user ?? null);
          setIsLoading(false);
        }
      } catch {
        if (isMounted) {
          clearStoredAuth();
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

