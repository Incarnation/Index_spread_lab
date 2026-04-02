import { NavLink, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  TrendingUp,
  ArrowLeftRight,
  Brain,
  BarChart3,
  Activity,
  Settings,
  FlaskConical,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useState } from "react";

const NAV_ITEMS: { to: string; icon: typeof LayoutDashboard; label: string; end?: boolean }[] = [
  { to: "/", icon: LayoutDashboard, label: "Overview", end: true },
  { to: "/trades", icon: TrendingUp, label: "Trades" },
  { to: "/decisions", icon: ArrowLeftRight, label: "Decisions" },
  { to: "/model", icon: Brain, label: "Model Monitor" },
  { to: "/performance", icon: BarChart3, label: "Performance" },
  { to: "/gex", icon: Activity, label: "GEX / Market" },
  { to: "/backtest", icon: FlaskConical, label: "Backtest" },
  { to: "/admin", icon: Settings, label: "Admin / Ops", end: true },
  { to: "/admin/auth-audit", icon: ShieldCheck, label: "Auth Audit" },
];

/**
 * Collapsible left sidebar with page navigation links.
 * Uses lucide-react icons and react-router NavLink for active state.
 */
export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  return (
    <aside
      className={cn(
        "flex flex-col border-r border-border bg-sidebar transition-all duration-200",
        collapsed ? "w-16" : "w-56"
      )}
    >
      <div className="flex items-center justify-between px-3 py-4 border-b border-border">
        {!collapsed && (
          <span className="text-sm font-bold tracking-tight text-foreground">
            ISL
          </span>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1 rounded hover:bg-sidebar-hover text-muted transition-colors"
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      <nav className="flex-1 py-2 space-y-0.5 px-2">
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
  );
}
