import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.jsx';

// Sem <StrictMode>: o motor do Pac-Man (script legado) nao tolera ser iniciado duas vezes.
createRoot(document.getElementById('root')).render(<App />);
