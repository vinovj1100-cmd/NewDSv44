const CACHE = 'ozon-wms-v21';
const SHELL = ['/', '/manifest.json', '/icons/icon-192.png', '/icons/icon-512.png',
               'https://unpkg.com/@zxing/library@0.21.3'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/api/') || url.pathname === '/events') return;   // keep live
  if (url.hostname !== self.location.hostname) {                                // fonts + ZXing: cache-first
    e.respondWith(caches.match(e.request).then(hit => hit ||
      fetch(e.request).then(r => { const c = r.clone(); caches.open(CACHE).then(cc => cc.put(e.request, c)); return r; })
        .catch(() => hit)));
    return;
  }
  e.respondWith(fetch(e.request).then(r => {                                    // shell: network-first
    const c = r.clone(); caches.open(CACHE).then(cc => cc.put(e.request, c)); return r;
  }).catch(() => caches.match(e.request)));
});
