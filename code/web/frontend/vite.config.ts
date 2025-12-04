import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/baby-cry-analyser/', // Updated to match GitHub repository name
  build: {
    outDir: 'dist',
    sourcemap: false
  }
})
