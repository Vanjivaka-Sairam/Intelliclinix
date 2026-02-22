"use client";

import { usePathname } from "next/navigation";
import DashboardNav from "./DashboardNav";
import { getStoredToken } from "@/lib/api";

export default function AppLayout({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    const isAuthPage = pathname === "/login" || pathname === "/signup";
    const isLoggedIn = !!getStoredToken();

    // Show nav only on non-auth pages AND when a token is present.
    // This prevents the navbar from flashing before the auth-guard redirect fires.
    const showNav = !isAuthPage && isLoggedIn;

    return (
        <div className="min-h-screen bg-cvat-bg-primary">
            {showNav && <DashboardNav />}
            <main
                className={!showNav ? "w-full h-full" : undefined}
                style={showNav ? { paddingTop: 'var(--navbar-h)' } : undefined}
            >
                {children}
            </main>
        </div>
    );
}
