import React from "react";

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
        <div className="flex min-h-[50vh] items-center justify-center p-8">
          <div className="w-full max-w-lg rounded-lg border border-loss/30 bg-loss-bg p-6">
            <h3 className="text-base font-semibold text-loss">Something went wrong</h3>
            <p className="mt-2 text-sm text-foreground-secondary break-words font-mono">
              {this.state.error?.message || "Unknown error"}
            </p>
            <button
              onClick={this.handleReset}
              className="mt-4 rounded-md border border-loss/30 bg-transparent px-4 py-2 text-sm text-loss hover:bg-loss/10 transition-colors"
            >
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
