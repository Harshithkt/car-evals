import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In dev, /api is proxied to the FastAPI server, so the browser talks to one origin.
// Set VITE_API_URL to call a backend elsewhere instead (CORS is enabled for :5173).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { chunkSizeWarningLimit: 1000 },
  server: {
    port: 5173,
    proxy: { '/api': { target: process.env.API_PROXY ?? 'http://127.0.0.1:8000', changeOrigin: true } },
  },
})
