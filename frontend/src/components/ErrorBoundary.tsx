import React from "react";
import { Alert, Button, Stack, Text, Title } from "@mantine/core";

type ErrorBoundaryProps = { children: React.ReactNode };
type ErrorBoundaryState = { hasError: boolean; error: Error | null };

/**
 * Catch render errors in child components and display a recovery UI
 * instead of crashing the entire dashboard with a white screen.
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <Stack align="center" justify="center" mih="50vh" p="xl">
          <Alert color="red" title="Something went wrong" w="100%" maw={600}>
            <Title order={4} mb="sm">
              An unexpected error occurred
            </Title>
            <Text size="sm" mb="md" style={{ fontFamily: "monospace", wordBreak: "break-word" }}>
              {this.state.error?.message || "Unknown error"}
            </Text>
            <Button variant="outline" color="red" onClick={this.handleReset}>
              Try again
            </Button>
          </Alert>
        </Stack>
      );
    }
    return this.props.children;
  }
}
