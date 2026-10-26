import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    // O TensorFlow.js e carregado sob demanda e e inerentemente grande.
    chunkSizeWarningLimit: 1500,
  },
});
