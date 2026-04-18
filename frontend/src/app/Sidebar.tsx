import { NavLink, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  TrendingUp,
  ArrowLeftRight,
  BarChart3,
  Activity,
  Settings,
  FlaskConical,
  ShieldCheck,
  Wallet,
  ChevronLeft,
  ChevronRight,
  X,
  Target,
} from "lucide-react";
import { useEffect, useState } from "react";

const NAV_ITEMS: { to: string; icon: typeof LayoutDashboard; label: string; end?: boolean }[] = [
  { to: "/", icon: LayoutDashboard, label: "Overview", end: true },
  { to: "/portfolio", icon: Wallet, label: "Portfolio" },
  { to: "/trades", icon: TrendingUp, label: "Trades" },
  { to: "/decisions", icon: ArrowLeftRight, label: "Decisions" },
  // "Model Monitor" link removed -- the online ML pipeline (and the
  // /api/model-ops, /api/label-metrics, /api/strategy-metrics endpoints
  // it consumed) was decommissioned.
  { to: "/performance", icon: BarChart3, label: "Performance" },
  { to: "/gex", icon: Activity, label: "GEX / Market" },
  { to: "/strategy", icon: FlaskConical, label: "Strategy" },
  { to: "/optimizer", icon: Target, label: "Optimizer" },
  { to: "/admin", icon: Settings, label: "Admin / Ops", end: true },
  { to: "/admin/auth-audit", icon: ShieldCheck, label: "Auth Audit" },
];

interface SidebarProps {
  mobileOpen?: boolean;
  onMobileClose?: () => void;
}

/**
 * Collapsible left sidebar with page navigation links.
 * On screens below `md`, renders as an overlay drawer controlled by AppShell.
 * On `md+`, renders as the standard collapsible sidebar.
 */
export function Sidebar({ mobileOpen = false, onMobileClose }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  // Close mobile drawer on navigation
  useEffect(() => {
    onMobileClose?.();
  }, [location.pathname]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      {/* Backdrop for mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={onMobileClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar panel */}
      <aside
        className={cn(
          // Mobile: fixed overlay drawer
          "fixed inset-y-0 left-0 z-50 flex flex-col border-r border-border bg-sidebar transition-transform duration-200 md:relative md:z-auto md:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
          // Desktop: collapsible width
          collapsed ? "md:w-16" : "w-56"
        )}
      >
        <div className="flex items-center justify-between px-3 py-4 border-b border-border">
          {!collapsed && (
            <span className="text-sm font-bold tracking-tight text-foreground">
              ISL
            </span>
          )}
          {/* Close button on mobile, collapse toggle on desktop */}
          {/* Mobile: close drawer button */}
          <button
            onClick={() => onMobileClose?.()}
            className="p-1 rounded hover:bg-sidebar-hover text-muted transition-colors md:hidden"
            aria-label="Close navigation menu"
          >
            <X className="h-4 w-4" />
          </button>
          {/* Desktop: collapse/expand toggle */}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded hover:bg-sidebar-hover text-muted transition-colors hidden md:inline-flex"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-expanded={!collapsed}
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>

        <nav className="flex-1 py-2 space-y-0.5 px-2 overflow-y-auto">
          {NAV_ITEMS.map(({ to, icon: Icon, label, end }) => {
            const isActive = end
              ? location.pathname === to
              : location.pathname.startsWith(to);

            return (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-active text-foreground"
                    : "text-muted hover:bg-sidebar-hover hover:text-foreground-secondary"
                )}
                title={collapsed ? label : undefined}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {!collapsed && <span>{label}</span>}
              </NavLink>
            );
          })}
        </nav>
      </aside>
    </>
  );
}
