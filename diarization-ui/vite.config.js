import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:5001",
    },
    host: "0.0.0.0",
    allowedHosts: ["voice.opensource.mieweb.org"],
  },
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});