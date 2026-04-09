import { useState, useCallback } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Navbar } from "./Navbar";

/**
 * Main application shell with sidebar navigation, top navbar, and routed content area.
 * Manages mobile sidebar open/close state shared between Navbar (hamburger) and Sidebar (overlay).
 */
export function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const toggleMobile = useCallback(() => setMobileOpen((v) => !v), []);
  const closeMobile = useCallback(() => setMobileOpen(false), []);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar mobileOpen={mobileOpen} onMobileClose={closeMobile} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Navbar onMenuToggle={toggleMobile} mobileMenuOpen={mobileOpen} />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
