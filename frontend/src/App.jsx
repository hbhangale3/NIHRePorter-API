import React, { useMemo, useState } from 'react'
import yaml from 'js-yaml'
import './styles.css'

const DEFAULT_YAML = `query:
  fiscal_years: [2024, 2025]
  broad_keywords:
    - health disparities
    - telemedicine
    - artificial intelligence
  text_search_field: all
  text_search_operator: or

  mesh_expansion:
    enabled: true
    max_terms_per_keyword: 15
    include_entry_terms: true
    include_tree_children: true
    max_tree_depth: 1
    fallback_to_original: true
    cache_enabled: true
  
  ai_expansion:
    enabled: false
    openai_api_key: null
    model: gpt-4o-mini
    max_expansions_per_keyword: 5
    context: "biomedical research and health disparities"

topics:
  - name: AI + Health Disparities
    include_any:
      - ai
      - artificial intelligence
      - machine learning
    include_all: []
    exclude_any:
      - mouse
      - mice
      - rat
    co_require_groups:
      - [health disparities, equity, inequity]
      - [clinical, community, public health]
`

function downloadBlob(bytes, filename, contentType) {
  const blob = new Blob([bytes], { type: contentType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename
  document.body.appendChild(a); a.click(); a.remove()
  URL.revokeObjectURL(url)
}

// ── Topic tag colours (cycles through a palette) ───────────────────────────
const TOPIC_COLORS = [
  { bg: 'rgba(107,69,245,0.15)', color: '#a78bfa', border: 'rgba(107,69,245,0.3)' },
  { bg: 'rgba(20,184,166,0.12)', color: '#5eead4', border: 'rgba(20,184,166,0.3)' },
  { bg: 'rgba(245,158,11,0.12)', color: '#fcd34d', border: 'rgba(245,158,11,0.3)' },
  { bg: 'rgba(239,68,68,0.12)',  color: '#fca5a5', border: 'rgba(239,68,68,0.3)'  },
  { bg: 'rgba(59,130,246,0.12)', color: '#93c5fd', border: 'rgba(59,130,246,0.3)' },
]

function TopicTag({ name, index }) {
  const c = TOPIC_COLORS[index % TOPIC_COLORS.length]
  return (
    <span className="topic-tag" style={{ background: c.bg, color: c.color, borderColor: c.border }}>
      {name}
    </span>
  )
}

function StatusPill({ status, busy }) {
  if (busy || status === 'running' || status === 'queued') {
    return (
      <span className="status-pill running">
        <span className="spinner" />
        {status === 'queued' ? 'Queued…' : 'Running…'}
      </span>
    )
  }
  if (status === 'completed') return <span className="status-pill completed">✓ Completed</span>
  if (status === 'failed')    return <span className="status-pill failed">✗ Failed</span>
  return <span className="status-pill idle">Idle</span>
}

export default function App() {
  const [configYaml, setConfigYaml] = useState(DEFAULT_YAML)
  const [maxPages, setMaxPages]     = useState(10)

  const [runId,   setRunId]   = useState(null)
  const [status,  setStatus]  = useState(null)
  const [message, setMessage] = useState(null)
  const [summary, setSummary] = useState(null)
  const [keywordExpansions, setKeywordExpansions] = useState(null)
  const [expansionTrace, setExpansionTrace] = useState(null)

  const [rows,   setRows]   = useState([])
  const [total,  setTotal]  = useState(0)
  const [offset, setOffset] = useState(0)
  const limit = 50

  const [busy, setBusy] = useState(false)
  const [showSuggestModal, setShowSuggestModal] = useState(false)
  const [topicDescription, setTopicDescription] = useState('')
  const [suggestingKeywords, setSuggestingKeywords] = useState(false)
  const [suggestedConfig, setSuggestedConfig] = useState(null)

  const yamlValid = useMemo(() => {
    try { yaml.load(configYaml); return true } catch { return false }
  }, [configYaml])

  // collect topic names from parsed config for consistent colour assignment
  const topicNames = useMemo(() => {
    try {
      const parsed = yaml.load(configYaml)
      return (parsed?.topics || []).map(t => t.name).filter(Boolean)
    } catch { return [] }
  }, [configYaml])

  function topicColor(name) {
    const idx = topicNames.indexOf(name)
    return TOPIC_COLORS[(idx >= 0 ? idx : 0) % TOPIC_COLORS.length]
  }

  async function startRun() {
    setBusy(true); setMessage(null); setSummary(null); setKeywordExpansions(null); setExpansionTrace(null)
    setRows([]); setTotal(0); setOffset(0)
    try {
      const resp = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_yaml: configYaml, max_pages: maxPages || null })
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      setRunId(data.run_id); setStatus(data.status)
      await pollUntilDone(data.run_id)
    } catch (e) {
      setMessage(String(e))
    } finally {
      setBusy(false)
    }
  }

  async function pollUntilDone(id) {
    for (let i = 0; i < 600; i++) {
      const resp = await fetch(`/api/runs/${id}`)
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      setStatus(data.status)
      if (data.message) setMessage(data.message)
      if (data.keyword_expansions) setKeywordExpansions(data.keyword_expansions)
      if (data.expansion_trace) setExpansionTrace(data.expansion_trace)
      if (data.status === 'completed') { await loadResults(id, 0); return }
      if (data.status === 'failed')    return
      await new Promise(r => setTimeout(r, 1000))
    }
    throw new Error('Timed out waiting for run completion')
  }

  async function loadResults(id, newOffset) {
    const resp = await fetch(`/api/runs/${id}/results?offset=${newOffset}&limit=${limit}`)
    if (!resp.ok) throw new Error(await resp.text())
    const data = await resp.json()
    setOffset(data.offset); setTotal(data.total)
    setRows(data.items || []); setSummary(data.summary || null)
  }

  async function downloadCsv() {
    if (!runId) return
    const resp = await fetch(`/api/runs/${runId}/export.csv`)
    if (!resp.ok) throw new Error(await resp.text())
    const bytes = await resp.arrayBuffer()
    downloadBlob(bytes, `nih_outreach_${runId.slice(0,8)}.csv`, 'text/csv')
  }

  function onUploadYaml(e) {
    const f = e.target.files?.[0]; if (!f) return
    const reader = new FileReader()
    reader.onload = () => setConfigYaml(String(reader.result || ''))
    reader.readAsText(f)
  }

  function fmtDate(s) {
    if (!s) return ''
    return s.slice(0, 10) // "YYYY-MM-DD"
  }

  async function suggestKeywords() {
    if (!topicDescription.trim()) return
    setSuggestingKeywords(true)
    try {
      const parsed = yaml.load(configYaml)
      const apiKey = parsed?.query?.ai_expansion?.openai_api_key || null
      
      const resp = await fetch('/api/suggest-keywords', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic_description: topicDescription,
          openai_api_key: apiKey,
          max_broad_keywords: 3,
          max_topic_terms: 10
        })
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      setSuggestedConfig(data)
    } catch (e) {
      alert('Failed to suggest keywords: ' + e.message)
    } finally {
      setSuggestingKeywords(false)
    }
  }

  function applyKeywords() {
    try {
      const parsed = yaml.load(configYaml)
      if (!parsed.query) parsed.query = {}
      if (!parsed.topics) parsed.topics = []
      
      // Apply broad keywords
      parsed.query.broad_keywords = suggestedConfig.broad_keywords
      
      // Create or update first topic with suggested terms
      const topicName = topicDescription.slice(0, 50)
      if (parsed.topics.length === 0) {
        parsed.topics.push({
          name: topicName,
          include_any: suggestedConfig.topic_terms
        })
      } else {
        parsed.topics[0].name = topicName
        parsed.topics[0].include_any = suggestedConfig.topic_terms
      }
      
      setConfigYaml(yaml.dump(parsed, { indent: 2 }))
      setShowSuggestModal(false)
      setSuggestedConfig(null)
      setTopicDescription('')
    } catch (e) {
      alert('Failed to update config: ' + e.message)
    }
  }

  const canPrev = offset > 0
  const canNext = offset + limit < total
  const page    = Math.floor(offset / limit) + 1
  const pages   = Math.ceil(total / limit) || 1

  return (
    <div className="app">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="header">
        <div className="header-left">
          <div className="header-icon">🔬</div>
          <div>
            <div className="header-title">NIH RePORTER PI Finder</div>
            <div className="header-sub">Search NIH-funded PIs by topic · powered by NIH RePORTER API</div>
          </div>
        </div>
        <span className="header-badge">/api</span>
      </div>

      {/* ── Config + Run panel ──────────────────────────────────────────────── */}
      <div className="top-grid">

        {/* YAML editor */}
        <div className="card yaml-card">
          <p className="card-title">Search Configuration (YAML)</p>

          <div className="yaml-toolbar">
            <label className="file-label">
              📂 Upload YAML
              <input type="file" accept=".yaml,.yml" onChange={onUploadYaml} />
            </label>
            <button className="btn-ghost" onClick={() => setShowSuggestModal(true)} type="button">
              🤖 Suggest Keywords
            </button>
            <button className="btn-ghost" onClick={() => setConfigYaml(DEFAULT_YAML)} type="button">
              Reset example
            </button>
            <span className={`yaml-status ${yamlValid ? 'valid' : 'invalid'}`}>
              <span className="yaml-dot" />
              {yamlValid ? 'Valid YAML' : 'Invalid YAML'}
            </span>
          </div>

          <textarea
            value={configYaml}
            onChange={e => setConfigYaml(e.target.value)}
            spellCheck={false}
          />
        </div>

        {/* Run controls */}
        <div className="run-panel">
          <div className="card">
            <p className="card-title">Run Settings</p>

            <label className="field-label">Max pages (500 projects / page)</label>
            <input
              type="number" min={1} value={maxPages}
              onChange={e => setMaxPages(Number(e.target.value))}
            />
            <p className="field-hint">NIH rate-limit: ~1 req/sec. Start small, increase for larger searches.</p>
          </div>

          <div className="card">
            <div className="status-row" style={{ marginBottom: 12 }}>
              <p className="card-title" style={{ margin: 0 }}>Actions</p>
              <StatusPill status={status} busy={busy} />
            </div>

            <button className="btn-primary" onClick={startRun} disabled={!yamlValid || busy}>
              {busy ? <><span className="spinner" style={{ borderColor: 'rgba(255,255,255,0.3)', borderTopColor: '#fff' }} />Searching…</> : '▶ Start Search'}
            </button>

            {message && !busy && (
              <div className="error-msg" style={{ marginTop: 10 }}>{message}</div>
            )}

            <div style={{ marginTop: 10 }}>
              <button className="btn-secondary" onClick={downloadCsv} disabled={!runId || status !== 'completed'}>
                ⬇ Export CSV
              </button>
            </div>
          </div>

          {/* AI Keyword Expansions */}
          {keywordExpansions && Object.keys(keywordExpansions).length > 0 && (
            <div className="card">
              <p className="card-title">🤖 AI Keyword Expansions</p>
              <div style={{ fontSize: 12, color: '#5a7299', marginBottom: 10 }}>
                AI expanded your keywords to improve search recall:
              </div>
              {Object.entries(keywordExpansions).map(([original, expanded]) => (
                <div key={original} style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#93c5fd', marginBottom: 4 }}>
                    {original} →
                  </div>
                  <div style={{ fontSize: 11, color: '#8b9dc3', lineHeight: 1.5 }}>
                    {expanded.join(', ')}
                  </div>
                </div>
              ))}
            </div>
          )}

          {expansionTrace && (
            <div className="card">
              <p className="card-title">Search Expansion Trace</p>
              <div style={{ fontSize: 12, color: '#5a7299', marginBottom: 8 }}>Original keywords:</div>
              <div style={{ fontSize: 12, color: '#c8d8f0', lineHeight: 1.6, marginBottom: 14 }}>
                {(expansionTrace.original_keywords || []).join(', ') || 'None'}
              </div>

              <div style={{ fontSize: 12, color: '#5a7299', marginBottom: 8 }}>
                MeSH expansion: {expansionTrace.mesh?.enabled ? 'enabled' : 'disabled'}
              </div>
              {expansionTrace.mesh?.enabled && (
                <div style={{ marginBottom: 14 }}>
                  {Object.entries(expansionTrace.mesh?.terms_by_keyword || {}).map(([original, expanded]) => (
                    <div key={original} style={{ marginBottom: 10 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: '#93c5fd', marginBottom: 4 }}>
                        {original} →
                      </div>
                      <div style={{ fontSize: 11, color: '#8b9dc3', lineHeight: 1.5 }}>
                        {Array.isArray(expanded) && expanded.length ? expanded.join(', ') : 'No additional MeSH terms'}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div style={{ fontSize: 12, color: '#5a7299', marginBottom: 8 }}>Final NIH search terms:</div>
              <div style={{ fontSize: 12, color: '#c8d8f0', lineHeight: 1.6 }}>
                {(expansionTrace.final_keywords || []).join(', ') || 'None'}
              </div>
            </div>
          )}

          {/* Summary stats */}
          {summary && (
            <div className="card">
              <p className="card-title">Results Summary</p>
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-value">{total}</div>
                  <div className="stat-label">Unique PIs</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{summary.matched_project_count ?? 0}</div>
                  <div className="stat-label">Project records</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{Object.keys(summary.counts_by_topic || {}).length}</div>
                  <div className="stat-label">Topics matched</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{Object.keys(summary.counts_by_admin_ic || {}).length}</div>
                  <div className="stat-label">Institutes (ICs)</div>
                </div>
              </div>

              {/* Per-topic breakdown */}
              {summary.counts_by_topic && Object.keys(summary.counts_by_topic).length > 0 && (
                <div style={{ marginTop: 14 }}>
                  {Object.entries(summary.counts_by_topic).map(([name, count], i) => {
                    const c = topicColor(name)
                    return (
                      <div key={name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                        <span className="topic-tag" style={{ background: c.bg, color: c.color, borderColor: c.border }}>
                          {name}
                        </span>
                        <span style={{ fontSize: 12, color: '#5a7299' }}>{count} projects</span>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Results table ───────────────────────────────────────────────────── */}
      <div className="results-section">
        <div className="results-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="results-title">Results</span>
            {total > 0 && <span className="total-badge">{total} PIs found</span>}
          </div>
          {total > 0 && (
            <div className="pagination">
              <span className="page-info">Page {page} of {pages}</span>
              <button className="btn-page" disabled={!canPrev} onClick={() => loadResults(runId, offset - limit)}>← Prev</button>
              <button className="btn-page" disabled={!canNext} onClick={() => loadResults(runId, offset + limit)}>Next →</button>
            </div>
          )}
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th className="col-pi">Contact PI</th>
                <th className="col-email">Email</th>
                <th className="col-org">Organization</th>
                <th className="col-ic">IC</th>
                <th className="col-fys">Fiscal Years</th>
                <th className="col-num">Project Numbers</th>
                <th className="col-date">Date Range</th>
                <th className="col-topic">Topics</th>
                <th className="col-title">Project Title</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={9}>
                    <div className="empty-state">
                      <div className="empty-state-icon">🔍</div>
                      <div className="empty-state-text">
                        {status === 'completed'
                          ? 'No results matched your topic rules. Try broadening your keywords.'
                          : 'Configure your search above and click Start Search.'}
                      </div>
                    </div>
                  </td>
                </tr>
              ) : rows.map((r, idx) => {
                const location = [r.organization_city, r.organization_state, r.organization_country].filter(Boolean).join(', ')
                const startD = fmtDate(r.project_start_date)
                const endD   = fmtDate(r.project_end_date)
                const rowKey = r.pi_profile_id || `${r.pi_name}-${r.organization_name}-${idx}`

                return (
                  <tr key={rowKey}>
                    {/* PI */}
                    <td className="col-pi">
                      <div className="pi-name">
                        {r.pi_last_name && r.pi_first_name
                          ? `${r.pi_last_name}, ${r.pi_first_name}`
                          : r.pi_name || <span className="empty-val">—</span>}
                      </div>
                      {r.pi_profile_id && (
                        <div className="pi-id">
                          <a href={`https://reporter.nih.gov/pi-details/${r.pi_profile_id}`}
                            target="_blank" rel="noreferrer"
                            style={{ color: '#4f7cdc', textDecoration: 'none', fontSize: 11 }}>
                            Profile ↗
                          </a>
                        </div>
                      )}
                    </td>

                    {/* Email */}
                    <td className="col-email">
                      {r.pi_email
                        ? <a href={`mailto:${r.pi_email}`} className="email-link">{r.pi_email}</a>
                        : <span className="empty-val">—</span>}
                    </td>

                    {/* Organization */}
                    <td className="col-org">
                      <div className="org-name">{r.organization_name || <span className="empty-val">—</span>}</div>
                      {location && <div className="org-loc">{location}</div>}
                    </td>

                    {/* IC */}
                    <td className="col-ic">
                      {r.admin_ic
                        ? <span className="ic-tag">{r.admin_ic}</span>
                        : <span className="empty-val">—</span>}
                    </td>

                    {/* Fiscal years */}
                    <td className="col-fys">
                      {Array.isArray(r.fiscal_years) && r.fiscal_years.length
                        ? r.fiscal_years.map(fy => <span key={fy} className="fy-pill">{fy}</span>)
                        : <span className="empty-val">—</span>}
                    </td>

                    {/* Project numbers */}
                    <td className="col-num">
                      {Array.isArray(r.project_numbers) && r.project_numbers.length
                        ? r.project_numbers.map((n, i) => {
                            const appl = r.project_ids?.[i]
                            return appl
                              ? <a key={i} href={`https://reporter.nih.gov/project-details/${appl}`} target="_blank" rel="noreferrer" className="proj-num">{n} ↗</a>
                              : <span key={i} className="proj-num" style={{ color: '#5a7299' }}>{n}</span>
                          })
                        : <span className="empty-val">—</span>}
                    </td>

                    {/* Date range */}
                    <td className="col-date">
                      {startD || endD
                        ? <div className="date-range">{startD}<span className="date-sep">↓</span>{endD}</div>
                        : <span className="empty-val">—</span>}
                    </td>

                    {/* Topics */}
                    <td className="col-topic">
                      {Array.isArray(r.matched_topics) && r.matched_topics.length
                        ? r.matched_topics.map((t, i) => {
                            const c = topicColor(t)
                            return <span key={i} className="topic-tag" style={{ background: c.bg, color: c.color, borderColor: c.border }}>{t}</span>
                          })
                        : <span className="empty-val">—</span>}
                    </td>

                    {/* Sample title */}
                    <td className="col-title">
                      {Array.isArray(r.sample_project_titles) && r.sample_project_titles.length
                        ? r.sample_project_titles.map((t, i) => <div key={i} className="proj-title">{t}</div>)
                        : <span className="empty-val">—</span>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {total > 0 && (
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
            <div className="pagination">
              <span className="page-info">Page {page} of {pages} · {total} total</span>
              <button className="btn-page" disabled={!canPrev} onClick={() => loadResults(runId, offset - limit)}>← Prev</button>
              <button className="btn-page" disabled={!canNext} onClick={() => loadResults(runId, offset + limit)}>Next →</button>
            </div>
          </div>
        )}
      </div>

      <p className="footer-tip">
        <strong>Tip:</strong> Keep broad_keywords broad for Stage 1 · use topic rules for precise filtering · abstracts & terms available in CSV export
      </p>

      {/* Keyword Suggestion Modal */}
      {showSuggestModal && (
        <div className="modal-overlay" onClick={() => setShowSuggestModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: 18, color: '#e2e8f0' }}>🤖 AI Keyword Suggester</h3>
            
            <label style={{ display: 'block', marginBottom: 8, fontSize: 13, color: '#94a3b8' }}>
              Describe your research topic:
            </label>
            <textarea
              value={topicDescription}
              onChange={e => setTopicDescription(e.target.value)}
              placeholder="e.g., AI and machine learning applications in reducing health disparities"
              style={{ width: '100%', minHeight: 80, marginBottom: 16, padding: 10, fontSize: 13, background: '#1e293b', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0' }}
            />

            <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
              <button 
                className="btn-primary" 
                onClick={suggestKeywords}
                disabled={!topicDescription.trim() || suggestingKeywords}
                style={{ flex: 1 }}
              >
                {suggestingKeywords ? 'Suggesting...' : 'Generate Keywords'}
              </button>
              <button className="btn-ghost" onClick={() => setShowSuggestModal(false)}>
                Cancel
              </button>
            </div>

            {suggestedConfig && (suggestedConfig.broad_keywords?.length > 0 || suggestedConfig.topic_terms?.length > 0) && (
              <div>
                {suggestedConfig.broad_keywords?.length > 0 && (
                  <>
                    <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8, fontWeight: 600 }}>
                      📍 Broad Keywords (for NIH API query):
                    </p>
                    <div style={{ background: '#0f172a', padding: 12, borderRadius: 6, marginBottom: 16 }}>
                      {suggestedConfig.broad_keywords.map((kw, i) => (
                        <div key={i} style={{ fontSize: 12, color: '#93c5fd', marginBottom: 4 }}>
                          • {kw}
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {suggestedConfig.topic_terms?.length > 0 && (
                  <>
                    <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8, fontWeight: 600 }}>
                      🎯 Topic Matching Terms (for local filtering):
                    </p>
                    <div style={{ background: '#0f172a', padding: 12, borderRadius: 6, marginBottom: 16, maxHeight: 200, overflow: 'auto' }}>
                      {suggestedConfig.topic_terms.map((kw, i) => (
                        <div key={i} style={{ fontSize: 12, color: '#a78bfa', marginBottom: 4 }}>
                          • {kw}
                        </div>
                      ))}
                    </div>
                  </>
                )}

                <button className="btn-primary" onClick={applyKeywords} style={{ width: '100%' }}>
                  ✓ Apply to Config
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
