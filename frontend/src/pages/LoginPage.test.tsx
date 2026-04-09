/**
 * Smoke tests for LoginPage: rendering, form interaction, and error display.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { LoginPage } from "./LoginPage";

const mockLogin = vi.fn();
const mockClearError = vi.fn();

let mockError: string | null = null;

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({
    login: mockLogin,
    error: mockError,
    clearError: mockClearError,
    token: null,
    user: null,
    loading: false,
    register: vi.fn(),
    logout: vi.fn(),
  }),
}));

function renderLogin() {
  return render(
    <MemoryRouter>
      <LoginPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockError = null;
});

describe("LoginPage", () => {
  it("renders sign-in heading and form fields", () => {
    renderLogin();
    expect(screen.getByText("Index Spread Lab")).toBeInTheDocument();
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("calls login on form submit with trimmed values", async () => {
    mockLogin.mockResolvedValue(undefined);
    renderLogin();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText(/username/i), " admin ");
    await user.type(screen.getByLabelText(/password/i), "secret");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(mockLogin).toHaveBeenCalledWith("admin", "secret");
  });

  it("displays error message from auth context", () => {
    mockError = "Invalid credentials";
    renderLogin();
    expect(screen.getByText("Invalid credentials")).toBeInTheDocument();
  });

  it("disables button while submitting", async () => {
    mockLogin.mockImplementation(() => new Promise(() => {}));
    renderLogin();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText(/username/i), "admin");
    await user.type(screen.getByLabelText(/password/i), "pass");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(screen.getByRole("button", { name: /signing in/i })).toBeDisabled();
  });
});
