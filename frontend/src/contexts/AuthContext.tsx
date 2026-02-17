/**
 * Auth context: token, user, login, register, logout.
 * Validates token on mount via GET /api/auth/me; clears token on 401.
 */

import React, { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as authStorage from "../auth";
import { API_BASE } from "../api";

export type AuthUser = { id: number; username: string; is_admin: boolean };

type AuthState = {
  token: string | null;
  user: AuthUser | null;
  loading: boolean;
  error: string | null;
};

type AuthContextValue = AuthState & {
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => void;
  clearError: () => void;
};

const apiUrl = (path: string) => `${(API_BASE || "").replace(/\/+$/, "")}${path}`;

const AuthContext = React.createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const [token, setTokenState] = useState<string | null>(() => authStorage.getToken());
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const clearError = useCallback(() => setError(null), []);

  /** Log out: call backend to record logout event, then clear token and redirect. */
  const logout = useCallback(async () => {
    const t = authStorage.getToken();
    if (t) {
      try {
        await fetch(apiUrl("/api/auth/logout"), {
          method: "POST",
          headers: { Authorization: `Bearer ${t}` },
        });
      } catch {
        // Ignore network errors; still clear local state.
      }
    }
    authStorage.clearToken();
    setTokenState(null);
    setUser(null);
    setError(null);
    navigate("/login", { replace: true });
  }, [navigate]);

  /** Validate token and load user; on 401 clear token. */
  const validateToken = useCallback(async () => {
    const t = authStorage.getToken();
    if (!t) {
      setLoading(false);
      return;
    }
    try {
      const r = await fetch(apiUrl("/api/auth/me"), {
        headers: { Authorization: `Bearer ${t}` },
      });
      if (r.status === 401) {
        authStorage.clearToken();
        setTokenState(null);
        setUser(null);
      } else if (r.ok) {
        const data = (await r.json()) as AuthUser;
        setUser(data);
        setTokenState(t);
      }
    } catch {
      setTokenState(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    validateToken();
  }, [validateToken]);

  const login = useCallback(
    async (username: string, password: string) => {
      setError(null);
      const r = await fetch(apiUrl("/api/auth/login"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setError((data.detail as string) || "Login failed");
        throw new Error((data.detail as string) || "Login failed");
      }
      const accessToken = data.access_token as string;
      const userPayload = data.user as AuthUser;
      authStorage.setToken(accessToken);
      setTokenState(accessToken);
      setUser(userPayload);
      setError(null);
      navigate("/", { replace: true });
    },
    [navigate]
  );

  const register = useCallback(
    async (username: string, password: string) => {
      setError(null);
      const r = await fetch(apiUrl("/api/auth/register"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setError((data.detail as string) || "Registration failed");
        throw new Error((data.detail as string) || "Registration failed");
      }
      const accessToken = data.access_token as string;
      const userPayload = data.user as AuthUser;
      authStorage.setToken(accessToken);
      setTokenState(accessToken);
      setUser(userPayload);
      setError(null);
      navigate("/", { replace: true });
    },
    [navigate]
  );

  const value: AuthContextValue = {
    token,
    user,
    loading,
    error,
    login,
    register,
    logout,
    clearError,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = React.useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
