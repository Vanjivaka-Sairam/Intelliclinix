"use client";

import { usePathname } from "next/navigation";
import DashboardNav from "./DashboardNav";

export default function AppLayout({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    const isAuthPage = pathname === "/login" || pathname === "/signup";

    return (
        <div className="min-h-screen bg-cvat-bg-primary">
            {!isAuthPage && <DashboardNav />}
            <main className={!isAuthPage ? "pt-16" : "w-full h-full"}>
                {children}
            </main>
        </div>
    );
}
