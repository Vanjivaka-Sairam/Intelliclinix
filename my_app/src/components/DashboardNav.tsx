"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect, useRef } from "react";
import {
  Upload,
  PieChart,
  CheckCircle,
  Brain,
  CircleUser,
  LogOut,
  ExternalLink,
  Settings,
  User,
  Cpu,
} from "lucide-react";

export default function DashboardNav() {
  const pathname = usePathname();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);

  // Helper to determine if the current path is active
  const isActive = (path: string) => pathname === path;

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Helper to format breadcrumb text
  const getPageTitle = () => {
    const current = pathname.replace("/", "");
    if (!current) return "Dashboard";
    return current.charAt(0).toUpperCase() + current.slice(1);
  };

  const handleLogout = async () => {
    try {
      await fetch("http://localhost:5328/auth/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch (err) {
      console.error("Logout error:", err);
    } finally {
      localStorage.removeItem("token");
      localStorage.removeItem("session");
      localStorage.removeItem("username");
      window.location.href = "/login";
    }
  };

  return (
    <div className="cvat-header">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between">
          {/* Logo and Brand */}
          <div className="flex items-center py-2">
            <div className="flex items-center space-x-2">
              <Brain className="h-6 w-6 text-cvat-primary" />
              <span className="text-cvat-text-white font-bold text-xl">IntelliClinix</span>
            </div>
            <span className="ml-2 px-2 py-1 bg-cvat-bg-header-light text-xs text-cvat-text-white rounded-md">
              AI Annotation
            </span>
          </div>

          {/* Main Navigation */}
          <div className="flex space-x-1">
            <Link
              href="/newupload"
              className={`cvat-nav-item flex items-center px-3 py-3 text-sm font-medium ${
                isActive("/newupload") ? "active" : ""
              }`}
            >
              <Upload className="h-4 w-4 mr-2" />
              New Upload
            </Link>

            <Link
              href="/models"
              className={`cvat-nav-item flex items-center px-3 py-3 text-sm font-medium ${
                isActive("/models") ? "active" : ""
              }`}
            >
              <Cpu className="h-4 w-4 mr-2" />
              Models
            </Link>

            <Link
              href="/predictions"
              className={`cvat-nav-item flex items-center px-3 py-3 text-sm font-medium ${
                isActive("/predictions") ? "active" : ""
              }`}
            >
              <PieChart className="h-4 w-4 mr-2" />
              Predictions
            </Link>

            <Link
              href="/corrected"
              className={`cvat-nav-item flex items-center px-3 py-3 text-sm font-medium ${
                isActive("/corrected") ? "active" : ""
              }`}
            >
              <CheckCircle className="h-4 w-4 mr-2" />
              Corrected
            </Link>
          </div>

          {/* User Menu */}
          <div className="flex items-center space-x-3">
            <a
              href="http://localhost:8080"
              target="_blank"
              rel="noopener noreferrer"
              className="cvat-nav-item flex items-center px-3 py-3 text-sm font-medium"
              title="Open CVAT in new tab"
            >
              <ExternalLink className="h-4 w-4 mr-2" />
              CVAT
            </a>
            <div className="h-5 border-r border-cvat-border-dark"></div>
            <div className="relative" ref={userMenuRef}>
              <button 
                className="p-1.5 rounded-full text-cvat-text-white/70 hover:text-cvat-text-white hover:bg-cvat-bg-header-light transition-colors duration-100"
                title="User Profile"
                onClick={() => setShowUserMenu(!showUserMenu)}
              >
                <CircleUser className="h-4 w-4" />
              </button>
              
              {/* User Dropdown Menu */}
              {showUserMenu && (
                <div className="absolute right-0 top-full mt-2 w-48 bg-cvat-bg-secondary border border-cvat-border rounded-lg shadow-cvat z-50">
                  <div className="py-2">
                    <div className="px-4 py-2 border-b border-cvat-border-light">
                      <p className="text-sm font-medium text-cvat-text-primary">
                        {typeof window !== 'undefined' ? localStorage.getItem('username') || 'User' : 'User'}
                      </p>
                      <p className="text-xs text-cvat-text-secondary">Medical Image Annotation</p>
                    </div>
                    <button className="w-full px-4 py-2 text-left text-sm text-cvat-text-primary hover:bg-cvat-bg-tertiary flex items-center">
                      <User className="h-4 w-4 mr-2" />
                      Profile
                    </button>
                    <button className="w-full px-4 py-2 text-left text-sm text-cvat-text-primary hover:bg-cvat-bg-tertiary flex items-center">
                      <Settings className="h-4 w-4 mr-2" />
                      Settings
                    </button>
                    <div className="border-t border-cvat-border-light mt-1 pt-1">
                      <button 
                        onClick={handleLogout}
                        className="w-full px-4 py-2 text-left text-sm text-cvat-error hover:bg-cvat-bg-tertiary flex items-center"
                      >
                        <LogOut className="h-4 w-4 mr-2" />
                        Logout
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Breadcrumb / Subtitle Bar */}
      <div className="bg-cvat-bg-header-light py-1 px-4">
        <div className="max-w-7xl mx-auto flex items-center text-xs text-cvat-text-white/70">
          <span>Medical Image Annotation Platform</span>
          <span className="mx-2">•</span>
          <span className="font-medium text-cvat-text-white">{getPageTitle()}</span>
        </div>
      </div>
    </div>
  );
}
