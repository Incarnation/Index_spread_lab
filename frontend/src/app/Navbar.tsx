import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { cn } from "@/lib/utils";
import { LogOut, RefreshCw, Circle } from "lucide-react";

/**
 * Top navigation bar with branding, market status, and user controls.
 */
export function Navbar() {
  const { user, logout } = useAuth();
  const { isMarketHours, paused, setPaused } = useAutoRefresh();

  return (
    <header className="flex h-12 items-center justify-between border-b border-border bg-card px-4">
      <div className="flex items-center gap-3">
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
        >
          <RefreshCw className={cn("h-3.5 w-3.5", !paused && "animate-spin-slow")} />
        </Button>

        {user && (
          <span className="text-xs text-muted-foreground">
            {user.username}
          </span>
        )}

        <Button variant="ghost" size="icon" onClick={logout} title="Log out">
          <LogOut className="h-3.5 w-3.5" />
        </Button>
      </div>
    </header>
  );
}
