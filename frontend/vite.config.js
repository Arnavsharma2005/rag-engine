import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// VITE_BASE is the path the UI is served under, e.g. /rag-engine/ on GitHub Pages
export default defineConfig({
  base: process.env.VITE_BASE || '/',
  plugins: [react()],
})
