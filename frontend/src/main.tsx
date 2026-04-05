import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import { AuthProvider } from "./contexts/AuthContext";
import { AppShell } from "./app/AppShell";
import { LoginPage } from "./pages/LoginPage";
import { OverviewPage } from "./pages/OverviewPage";
import { TradesPage } from "./pages/TradesPage";
import { DecisionsPage } from "./pages/DecisionsPage";
import { ModelMonitorPage } from "./pages/ModelMonitorPage";
import { PerformancePage } from "./pages/PerformancePage";
import { GexPage } from "./pages/GexPage";
import { AdminPage } from "./pages/AdminPage";
import { StrategyConfigPage } from "./pages/StrategyConfigPage";
import { PortfolioPage } from "./pages/PortfolioPage";
import { AuthAuditPage } from "./pages/AuthAuditPage";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { ErrorBoundary } from "./components/ErrorBoundary";

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
              <Route index element={<OverviewPage />} />
              <Route path="portfolio" element={<PortfolioPage />} />
              <Route path="trades" element={<TradesPage />} />
              <Route path="decisions" element={<DecisionsPage />} />
              <Route path="model" element={<ModelMonitorPage />} />
              <Route path="performance" element={<PerformancePage />} />
              <Route path="gex" element={<GexPage />} />
              <Route path="admin" element={<AdminPage />} />
              <Route path="admin/auth-audit" element={<AuthAuditPage />} />
              <Route path="strategy" element={<StrategyConfigPage />} />
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  </React.StrictMode>
);
