import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { cn } from "@/lib/utils";
import { LogOut, RefreshCw, Circle, Menu } from "lucide-react";

interface NavbarProps {
  onMenuToggle?: () => void;
  mobileMenuOpen?: boolean;
}

/**
 * Top navigation bar with branding, market status, and user controls.
 * Shows a hamburger menu button on mobile to toggle the sidebar drawer.
 */
export function Navbar({ onMenuToggle, mobileMenuOpen = false }: NavbarProps) {
  const { user, logout } = useAuth();
  const { isMarketHours, paused, setPaused } = useAutoRefresh();

  return (
    <header className="flex h-12 items-center justify-between border-b border-border bg-card px-4">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={onMenuToggle}
          aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={mobileMenuOpen}
        >
          <Menu className="h-4 w-4" />
        </Button>
        <h1 className="text-sm font-bold text-foreground tracking-tight">
          Index Spread Lab
        </h1>
        <Badge variant={isMarketHours ? "profit" : "muted"}>
          <Circle className="mr-1 h-2 w-2 fill-current" />
          {isMarketHours ? "Market Open" : "Market Closed"}
        </Badge>
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setPaused(!paused)}
          title={paused ? "Resume auto-refresh" : "Pause auto-refresh"}
          aria-label={paused ? "Resume auto-refresh" : "Pause auto-refresh"}
        >
          <RefreshCw className={cn("h-3.5 w-3.5", !paused && "animate-spin-slow")} />
        </Button>

        {user && (
          <span className="text-xs text-muted-foreground">
            {user.username}
          </span>
        )}

        <Button variant="ghost" size="icon" onClick={logout} title="Log out" aria-label="Log out">
          <LogOut className="h-3.5 w-3.5" />
        </Button>
      </div>
    </header>
  );
}
