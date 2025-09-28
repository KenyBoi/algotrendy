import React from 'react'
import { fetchJson } from './api'
import { formatCurrency, formatQuantity, sortBy } from './utils'

export default function Portfolio() {
  const [overview, setOverview] = React.useState(null)
  const [positions, setPositions] = React.useState(null)
  const [error, setError] = React.useState(null)
  const [query, setQuery] = React.useState('')
  const [sortKey, setSortKey] = React.useState('market_value')
  const [sortDir, setSortDir] = React.useState('desc')

  React.useEffect(() => {
    let mounted = true

    async function load() {
      const ov = await fetchJson('/api/proxy/portfolio')
      const pos = await fetchJson('/api/proxy/positions')

      if (!mounted) return

      if (ov.error) setError(ov.error)
      else setOverview(ov)

      if (pos.error) setError(pos.error)
      else setPositions(pos.positions || [])
    }

    // show skeletons for 300ms minimum for better perceived performance
    const t = setTimeout(load, 120)

    return () => { mounted = false; clearTimeout(t) }
  }, [])

  // Listen for optimistic close events to update local state immediately
  React.useEffect(() => {
    function handler(e) {
      const sym = e.detail?.symbol
      if (!sym) return
      setPositions(prev => (prev || []).filter(p => p.symbol !== sym))
    }

    window.addEventListener('close-position', handler)
    return () => window.removeEventListener('close-position', handler)
  }, [])

  const [page, setPage] = React.useState(1)
  const [pageSize] = React.useState(8)
  const [modalItem, setModalItem] = React.useState(null)

  const filtered = React.useMemo(() => {
    if (!positions) return []
    const q = query.trim().toLowerCase()
    let list = positions.filter(p => {
      if (!q) return true
      return p.symbol.toLowerCase().includes(q) || (p.name || '').toLowerCase().includes(q)
    })
    list = sortBy(list, sortKey, sortDir)
    return list
  }, [positions, query, sortKey, sortDir])

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
  const pageItems = filtered.slice((page - 1) * pageSize, page * pageSize)

  function toggleSort(key) {
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }

  // Toast + undo cache (kept inside the component so hooks work correctly)
  const [toasts, setToasts] = React.useState([])
  const undoCacheRef = React.useRef(new Map()) // symbol -> { item, timeoutId }

  function showToast(message, { label = 'Undo', onClick = null, timeout = 5000 } = {}) {
    const id = Math.random().toString(36).slice(2, 9)
    const item = { id, message, label, onClick }
    setToasts(t => [item, ...t])
    if (timeout) setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), timeout)
  }

  async function backendClose(symbol) {
    try {
      const resp = await fetch(`/api/proxy/positions/${symbol}/close`, { method: 'POST' })
      return resp.ok
    } catch (e) {
      return false
    }
  }

  function closePositionOptimistic(symbol) {
    // remove from local store and cache the removed item for undo
    setPositions(prev => {
      const removed = (prev || []).find(p => p.symbol === symbol)
      if (!removed) return prev
      const next = (prev || []).filter(p => p.symbol !== symbol)

      // add to undo cache for a short window
      const timeoutId = setTimeout(() => {
        undoCacheRef.current.delete(symbol)
      }, 6000)
      undoCacheRef.current.set(symbol, { item: removed, timeoutId })

      // show toast with undo button
      showToast(`Closed ${symbol}`, {
        label: 'Undo',
        onClick: async () => {
          const ent = undoCacheRef.current.get(symbol)
          if (!ent) return
          clearTimeout(ent.timeoutId)
          undoCacheRef.current.delete(symbol)
          // restore locally
          setPositions(pos => [ent.item, ...(pos || [])])
          // attempt a backend reopen (best-effort)
          try { await fetch(`/api/proxy/positions/${symbol}/reopen`, { method: 'POST' }) } catch (e) { }
          // remove toast(s) for this symbol
          setToasts(t => t.filter(x => x.message !== `Closed ${symbol}`))
        },
        timeout: 6000,
      })

      return next
    })

    // call backend but don't block UI; on failure, restore from cache and show error
    backendClose(symbol).then(ok => {
      if (!ok) {
        const ent = undoCacheRef.current.get(symbol)
        if (ent) {
          clearTimeout(ent.timeoutId)
          setPositions(pos => [ent.item, ...(pos || [])])
          undoCacheRef.current.delete(symbol)
        }
        showToast(`Failed to close ${symbol}`, { label: null, timeout: 4000 })
      } else {
        // success - we can clear cache entry if still present
        const ent = undoCacheRef.current.get(symbol)
        if (ent) {
          clearTimeout(ent.timeoutId)
          undoCacheRef.current.delete(symbol)
        }
      }
    })
  }

  if (error) return <div style={{ color: 'crimson' }}>Error: {error}</div>

  return (
    <div className="app-shell">
      <div className="card">
        <div className="h-row">
          <div>
            <div className="small">Portfolio Total</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{overview ? formatCurrency(overview.total_value) : <span className="small">—</span>}</div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input className="input" placeholder="Search symbol or name" value={query} onChange={(e) => setQuery(e.target.value)} />
            <button className="btn" onClick={() => { setQuery('') }}>Clear</button>
          </div>
        </div>
      </div>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <h3 style={{ margin: 0 }}>Positions {positions ? `(${positions.length})` : ''}</h3>
          <div className="small">{overview ? `Cash: ${formatCurrency(overview.cash_balance)}` : ''}</div>
        </div>

        {!positions && (
          <div>
            {/* skeleton rows */}
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="skel-row" style={{ marginBottom: 8 }}>
                <div className="skeleton" style={{ width: 100 }} />
                <div className="skeleton" style={{ width: '60%' }} />
                <div className="skeleton" style={{ width: 80 }} />
                <div className="skeleton" style={{ width: 100 }} />
              </div>
            ))}
          </div>
        )}

        {positions && (
          <>
            <table className="table">
              <thead>
                <tr>
                  <th onClick={() => toggleSort('symbol')} style={{ cursor: 'pointer' }}>Symbol</th>
                  <th onClick={() => toggleSort('name')} style={{ cursor: 'pointer' }}>Name</th>
                  <th className="align-right" onClick={() => toggleSort('quantity')} style={{ cursor: 'pointer' }}>Quantity</th>
                  <th className="align-right" onClick={() => toggleSort('market_value')} style={{ cursor: 'pointer' }}>Market Value</th>
                  <th style={{ width: 160 }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {pageItems.map(p => (
                  <tr key={p.symbol}>
                    <td style={{ padding: 8 }} data-label="Symbol">{p.symbol}</td>
                    <td style={{ padding: 8 }} className="small" data-label="Name">{p.name}</td>
                    <td style={{ padding: 8, textAlign: 'right' }} data-label="Quantity">{formatQuantity(p.quantity)}</td>
                    <td style={{ padding: 8, textAlign: 'right' }} data-label="Market Value">{formatCurrency(p.market_value)}</td>
                    <td style={{ padding: 8 }}>
                      <div className="row-actions">
                        <button className="action-ghost" onClick={() => setModalItem(p)} aria-label={`View ${p.symbol}`}>View</button>
                        <button className="action-ghost" onClick={() => closePositionOptimistic(p.symbol)} aria-label={`Close ${p.symbol}`}>Close</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
              <div className="small">Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, filtered.length)} of {filtered.length}</div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="action-ghost" onClick={() => setPage(1)} disabled={page === 1}>« First</button>
                <button className="action-ghost" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>‹ Prev</button>
                <div className="small" style={{ padding: '6px 10px' }}>Page {page} / {totalPages}</div>
                <button className="action-ghost" onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}>Next ›</button>
                <button className="action-ghost" onClick={() => setPage(totalPages)} disabled={page === totalPages}>Last »</button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Position detail modal */}
      {modalItem && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal">
            <button className="close" onClick={() => setModalItem(null)}>✕</button>
            <h3>{modalItem.symbol} — {modalItem.name}</h3>
            <div className="meta">Quantity: {formatQuantity(modalItem.quantity)} • Market Value: {formatCurrency(modalItem.market_value)}</div>
            <div style={{ marginTop: 8 }}>
              <p className="small">Details: {modalItem.details || 'No additional details available.'}</p>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <button className="btn" onClick={() => { setModalItem(null) }}>Close</button>
              <button className="action-ghost" onClick={() => { closePositionOptimistic(modalItem.symbol); setModalItem(null) }}>Close Position</button>
            </div>
          </div>
        </div>
      )}
      {/* Toasts */}
      <div className="toast-container" aria-live="polite">
        {toasts.map(t => (
          <div key={t.id} className="toast">
            <div style={{ fontSize: 13 }}>{t.message}</div>
            {t.onClick ? <button className="undo" onClick={() => { t.onClick(); setToasts(s => s.filter(x => x.id !== t.id)) }}>{t.label}</button> : null}
          </div>
        ))}
      </div>
    </div>
  )
}


