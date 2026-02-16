import React from "react";
import ReactDOM from "react-dom/client";
import "@mantine/core/styles.css";
import { MantineProvider } from "@mantine/core";
import { DashboardApp } from "./DashboardApp";

// Mount the React dashboard into the Vite root element.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <MantineProvider defaultColorScheme="light">
      <DashboardApp />
    </MantineProvider>
  </React.StrictMode>
);

