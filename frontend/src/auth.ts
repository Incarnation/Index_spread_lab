/**
 * Client-side auth token storage (sessionStorage).
 * Used by AuthContext and api.ts to attach Bearer token to requests.
 */

const AUTH_TOKEN_KEY = "auth_token";

export function getToken(): string | null {
  try {
    return sessionStorage.getItem(AUTH_TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  sessionStorage.setItem(AUTH_TOKEN_KEY, token);
}

export function clearToken(): void {
  sessionStorage.removeItem(AUTH_TOKEN_KEY);
}
