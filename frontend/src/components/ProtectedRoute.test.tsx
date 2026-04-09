import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: vi.fn(),
}));

import { useAuth } from "@/contexts/AuthContext";
import { ProtectedRoute } from "./ProtectedRoute";

const mockedUseAuth = vi.mocked(useAuth);

function renderWithRouter(authState: { token: string | null; loading: boolean }) {
  mockedUseAuth.mockReturnValue({
    ...authState,
    user: authState.token ? { id: 1, username: "testuser", is_admin: false } : null,
    error: null,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
    clearError: vi.fn(),
  });

  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <div data-testid="protected-content">Secret Page</div>
            </ProtectedRoute>
          }
        />
        <Route path="/login" element={<div data-testid="login-page">Login</div>} />
      </Routes>
    </MemoryRouter>
  );
}

describe("ProtectedRoute", () => {
  it("shows loading spinner while auth is loading", () => {
    renderWithRouter({ token: null, loading: true });
    expect(screen.getByText("Checking auth…")).toBeInTheDocument();
    expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
  });

  it("renders children when authenticated", () => {
    renderWithRouter({ token: "valid-jwt", loading: false });
    expect(screen.getByTestId("protected-content")).toBeInTheDocument();
    expect(screen.getByText("Secret Page")).toBeInTheDocument();
  });

  it("redirects to login when not authenticated", () => {
    renderWithRouter({ token: null, loading: false });
    expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
    expect(screen.getByTestId("login-page")).toBeInTheDocument();
  });
});
