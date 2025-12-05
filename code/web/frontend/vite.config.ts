import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/', // Temporarily reset for local dev verification
  build: {
    outDir: 'dist',
    sourcemap: false
  }
})
