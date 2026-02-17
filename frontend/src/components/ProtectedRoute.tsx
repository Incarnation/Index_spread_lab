/**
 * Renders children only when the user is authenticated; otherwise redirects to /login.
 * Shows a brief loading state while token is validated.
 */

import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { Box, Loader, Text } from "@mantine/core";
import { useAuth } from "../contexts/AuthContext";

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { token, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <Box ta="center" py="xl">
        <Loader size="lg" />
        <Text size="sm" c="dimmed" mt="sm">
          Checking auth…
        </Text>
      </Box>
    );
  }

  if (!token) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}
