/* LLM UI service worker — app-shell caching for PWA installability + offline shell */
const CACHE = 'llm-ui-v2';
const APP_SHELL = [
  '/',
  '/static/css/theme.css',
  '/static/js/main.js',
  '/static/js/services/offline.js',
  '/static/icons/icon.svg',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(APP_SHELL)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  // Only handle same-origin requests; never cache API or dynamic data.
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/api/')) return;
  if (url.pathname.startsWith('/uploads/')) return;
  if (url.pathname.startsWith('/outputs/')) return;

  // Navigations: network-first, fall back to cache (then cached shell).
  if (req.mode === 'navigation') {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
          return res;
        })
        .catch(() => caches.match(req).then((m) => m || caches.match('/')))
    );
    return;
  }

  // Static assets: cache-first, then network (populate cache).
  event.respondWith(
    caches.match(req).then(
      (cached) =>
        cached ||
        fetch(req)
          .then((res) => {
            if (res && res.ok) {
              const copy = res.clone();
              caches.open(CACHE).then((c) => c.put(req, copy));
            }
            return res;
          })
          .catch(() => cached)
    )
  );
});

// Broadcast online/offline status to every open client so the chat store can
// flip the queue drain on. Pages also listen on `online`/`offline` directly,
// this is a belt-and-suspenders second channel for the SW's vantage point.
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'ping-status') {
    event.source && event.source.postMessage({ type: 'offline-status', online: self.navigator.onLine })
  }
});
