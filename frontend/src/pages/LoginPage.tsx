/**
 * Login page for allowed users only. No self-registration; users must be pre-created.
 */

import React, { useState } from "react";
import { Alert, Button, Container, Paper, Stack, TextInput, Title } from "@mantine/core";
import { useAuth } from "../contexts/AuthContext";

export function LoginPage() {
  const { login, error, clearError } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    setSubmitting(true);
    try {
      await login(username.trim(), password);
    } catch {
      // Error already set in context
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Container size="xs" py="xl">
      <Paper p="xl" shadow="sm" withBorder>
        <Title order={3} mb="md">
          Log in
        </Title>
        <form onSubmit={handleSubmit}>
          <Stack gap="md">
            {error && (
              <Alert color="red" onClose={clearError} withCloseButton>
                {error}
              </Alert>
            )}
            <TextInput
              label="Username"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.currentTarget.value)}
              required
              minLength={1}
              autoComplete="username"
            />
            <TextInput
              label="Password"
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.currentTarget.value)}
              required
              minLength={1}
              autoComplete="current-password"
            />
            <Button type="submit" loading={submitting} fullWidth>
              Log in
            </Button>
          </Stack>
        </form>
      </Paper>
    </Container>
  );
}
