/**
 * Offline cache — IndexedDB-backed store for conversations, messages, and an
 * outbound message queue. Lets the PWA keep working when the network drops:
 *
 *   • saveConversations(list)  — overwrite cached conversation list
 *   • saveConversation(id, conv) — cache one conversation + its messages
 *   • getCachedConversations() — read on cold start
 *   • getCachedConversation(id) — read one conversation
 *   • enqueue({conversationId, body}) — buffer an outgoing message
 *   • drainQueue(sender) — when back online, replay queued messages
 *   • onStatusChange(cb) — fires with {online, queued}
 *
 * The IDB schema is intentionally tiny: one object store per kind, keyed by
 * the server's id (string for conversations, `${conversationId}:${messageId}`
 * for messages, autoincrement for the queue). Migrations: bump DB_VERSION
 * and the upgrade callback.
 */
const DB_NAME = 'llm-ui'
const DB_VERSION = 1
const STORE_CONVERSATIONS = 'conversations'
const STORE_MESSAGES = 'messages'
const STORE_QUEUE = 'queue'

let _dbPromise = null
function openDb() {
  if (_dbPromise) return _dbPromise
  _dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') return reject(new Error('IndexedDB unavailable'))
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_CONVERSATIONS)) db.createObjectStore(STORE_CONVERSATIONS, { keyPath: 'id' })
      if (!db.objectStoreNames.contains(STORE_MESSAGES)) db.createObjectStore(STORE_MESSAGES, { keyPath: 'key' })
      if (!db.objectStoreNames.contains(STORE_QUEUE)) db.createObjectStore(STORE_QUEUE, { keyPath: 'id', autoIncrement: true })
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
  return _dbPromise
}

function tx(store, mode = 'readonly') {
  return openDb().then((db) => db.transaction(store, mode).objectStore(store))
}

function reqAsPromise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

function messageKey(conversationId, messageId) {
  return `${conversationId}:${messageId}`
}

export const offline = {
  isSupported() { return typeof indexedDB !== 'undefined' },

  async saveConversations(list) {
    if (!Array.isArray(list) || !list.length) return
    const store = await tx(STORE_CONVERSATIONS, 'readwrite')
    await Promise.all(list.map((c) => reqAsPromise(store.put(c))))
  },

  async saveConversation(conv, messages) {
    if (!conv?.id) return
    const store = await tx(STORE_CONVERSATIONS, 'readwrite')
    await reqAsPromise(store.put(conv))
    if (Array.isArray(messages) && messages.length) {
      const mStore = await tx(STORE_MESSAGES, 'readwrite')
      await Promise.all(messages.map((m) => reqAsPromise(mStore.put({
        key: messageKey(conv.id, m.id),
        conversationId: conv.id,
        message: m,
      }))))
    }
  },

  async getCachedConversations() {
    try {
      const store = await tx(STORE_CONVERSATIONS)
      return await reqAsPromise(store.getAll())
    } catch { return [] }
  },

  async getCachedConversation(id) {
    try {
      const store = await tx(STORE_CONVERSATIONS)
      const conv = await reqAsPromise(store.get(String(id)))
      if (!conv) return null
      const mStore = await tx(STORE_MESSAGES)
      const all = await reqAsPromise(mStore.getAll())
      const msgs = all
        .filter((r) => r.conversationId === String(id))
        .map((r) => r.message)
        .sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''))
      return { conversation: conv, messages: msgs }
    } catch { return null }
  },

  async enqueue(item) {
    const store = await tx(STORE_QUEUE, 'readwrite')
    const id = await reqAsPromise(store.add({ ...item, queuedAt: Date.now() }))
    notifyStatus()
    return id
  },

  async listQueue() {
    try {
      const store = await tx(STORE_QUEUE)
      return await reqAsPromise(store.getAll())
    } catch { return [] }
  },

  async deleteQueueItem(id) {
    const store = await tx(STORE_QUEUE, 'readwrite')
    await reqAsPromise(store.delete(id))
    notifyStatus()
  },

  /**
   * Drain the queue. `sender(item)` is awaited per item; on success the
   * queue entry is removed. Failed sends stay queued for the next attempt.
   * Caller decides what "success" means (HTTP 2xx, response stream open, etc.).
   */
  async drainQueue(sender) {
    const items = await this.listQueue()
    let sent = 0
    for (const item of items) {
      try {
        const ok = await sender(item)
        if (ok) { await this.deleteQueueItem(item.id); sent++ }
      } catch { /* leave queued; try again next online event */ }
    }
    return sent
  },

  // ─── Status ─────────────────────────────────────────
  isOnline() { return typeof navigator !== 'undefined' ? navigator.onLine : true },

  async queueLength() { return (await this.listQueue()).length },

  /**
   * Subscribe to status changes. Fires immediately with the current state,
   * then again on online/offline events or queue changes. Returns an
   * unsubscribe function.
   */
  onStatusChange(cb) {
    const handler = () => notifyStatus()
    if (typeof window !== 'undefined') {
      window.addEventListener('online', handler)
      window.addEventListener('offline', handler)
    }
    notifyStatus()
    return () => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('online', handler)
        window.removeEventListener('offline', handler)
      }
    }
  },
}

let _statusListeners = []
async function notifyStatus() {
  if (!_statusListeners.length) return
  const status = { online: offline.isOnline(), queued: await offline.queueLength() }
  _statusListeners.forEach((cb) => { try { cb(status) } catch { /* listener errored; ignore */ } })
}

// Allow the SW to push status updates too (postMessage from sw → 'offline-status')
if (typeof window !== 'undefined') {
  window.addEventListener('message', (e) => {
    if (e.data && e.data.type === 'offline-status') notifyStatus()
  })
}

export function onStatus(cb) {
  _statusListeners.push(cb)
  notifyStatus()
  return () => { _statusListeners = _statusListeners.filter((x) => x !== cb) }
}

// Self-check — exercises every read/write path. Run via a quick test page
// (`?selftest=offline`) or call `offlineSelfTest()` from the devtools console.
export async function offlineSelfTest() {
  if (!offline.isSupported()) return { ok: false, reason: 'no IDB' }
  const stamp = Date.now()
  const conv = { id: `t-${stamp}`, title: 'self test', updated_at: new Date().toISOString() }
  const msgs = [
    { id: `m1-${stamp}`, role: 'user', content: 'hi', created_at: new Date().toISOString() },
    { id: `m2-${stamp}`, role: 'assistant', content: 'hello', created_at: new Date().toISOString() },
  ]
  await offline.saveConversation(conv, msgs)
  const got = await offline.getCachedConversation(conv.id)
  if (!got || got.messages.length !== 2) throw new Error('roundtrip failed')
  const qid = await offline.enqueue({ conversationId: conv.id, body: 'queued' })
  if ((await offline.queueLength()) !== 1) throw new Error('queue length wrong')
  await offline.deleteQueueItem(qid)
  if ((await offline.queueLength()) !== 0) throw new Error('queue delete failed')
  return { ok: true }
}
