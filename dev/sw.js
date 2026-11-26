// テスト版は終了(正式版に統合)。古いキャッシュを消して何もしない
self.addEventListener('install',()=>self.skipWaiting());
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k.startsWith('kaigi-dev')).map(k=>caches.delete(k)))).then(()=>self.registration.unregister())));
