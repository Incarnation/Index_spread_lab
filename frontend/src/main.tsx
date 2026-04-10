import React, { Suspense } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import { AuthProvider } from "./contexts/AuthContext";
import { AppShell } from "./app/AppShell";
import { LoginPage } from "./pages/LoginPage";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { RefreshCw } from "lucide-react";

const OverviewPage = React.lazy(() => import("./pages/OverviewPage").then((m) => ({ default: m.OverviewPage })));
const TradesPage = React.lazy(() => import("./pages/TradesPage").then((m) => ({ default: m.TradesPage })));
const DecisionsPage = React.lazy(() => import("./pages/DecisionsPage").then((m) => ({ default: m.DecisionsPage })));
const ModelMonitorPage = React.lazy(() => import("./pages/ModelMonitorPage").then((m) => ({ default: m.ModelMonitorPage })));
const PerformancePage = React.lazy(() => import("./pages/PerformancePage").then((m) => ({ default: m.PerformancePage })));
const GexPage = React.lazy(() => import("./pages/GexPage").then((m) => ({ default: m.GexPage })));
const AdminPage = React.lazy(() => import("./pages/AdminPage").then((m) => ({ default: m.AdminPage })));
const StrategyConfigPage = React.lazy(() => import("./pages/StrategyConfigPage").then((m) => ({ default: m.StrategyConfigPage })));
const PortfolioPage = React.lazy(() => import("./pages/PortfolioPage").then((m) => ({ default: m.PortfolioPage })));
const AuthAuditPage = React.lazy(() => import("./pages/AuthAuditPage").then((m) => ({ default: m.AuthAuditPage })));
const OptimizerPage = React.lazy(() => import("./pages/OptimizerPage"));

function PageLoader() {
  return (
    <div className="flex items-center justify-center py-24 text-muted-foreground">
      <RefreshCw className="h-5 w-5 animate-spin mr-2" />
      <span className="text-sm">Loading...</span>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route
              element={
                <ProtectedRoute>
                  <AppShell />
                </ProtectedRoute>
              }
            >
              <Route index element={<Suspense fallback={<PageLoader />}><OverviewPage /></Suspense>} />
              <Route path="portfolio" element={<Suspense fallback={<PageLoader />}><PortfolioPage /></Suspense>} />
              <Route path="trades" element={<Suspense fallback={<PageLoader />}><TradesPage /></Suspense>} />
              <Route path="decisions" element={<Suspense fallback={<PageLoader />}><DecisionsPage /></Suspense>} />
              <Route path="model" element={<Suspense fallback={<PageLoader />}><ModelMonitorPage /></Suspense>} />
              <Route path="performance" element={<Suspense fallback={<PageLoader />}><PerformancePage /></Suspense>} />
              <Route path="gex" element={<Suspense fallback={<PageLoader />}><GexPage /></Suspense>} />
              <Route path="admin" element={<Suspense fallback={<PageLoader />}><AdminPage /></Suspense>} />
              <Route path="admin/auth-audit" element={<Suspense fallback={<PageLoader />}><AuthAuditPage /></Suspense>} />
              <Route path="strategy" element={<Suspense fallback={<PageLoader />}><StrategyConfigPage /></Suspense>} />
              <Route path="optimizer" element={<Suspense fallback={<PageLoader />}><OptimizerPage /></Suspense>} />
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  </React.StrictMode>
);
