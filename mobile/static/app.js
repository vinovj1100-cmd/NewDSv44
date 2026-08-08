/* Neural Fulfillment Mobile PWA — app.js */
const API_BASE = window.location.origin;
let currentView = 'dashboard';

async function api(path, opts={}) {
  const res = await fetch(`${API_BASE}${path}`, opts);
  return res.json();
}

function showView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  currentView = id;
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(console.error);
}

console.log('Neural Fulfillment Mobile v4.4 loaded');
