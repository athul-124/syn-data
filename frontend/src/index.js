import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Add global error handlers
window.addEventListener('unhandledrejection', event => {
  console.warn('Unhandled promise rejection:', event.reason);
  event.preventDefault();
});

window.addEventListener('error', event => {
  console.error('Global error:', event.error);
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
