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
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

const TOPIC_COLORS = [
  { bg: 'rgba(92, 158, 149, 0.16)', color: '#2f6f69', border: 'rgba(92, 158, 149, 0.24)' },
  { bg: 'rgba(135, 170, 132, 0.18)', color: '#557451', border: 'rgba(135, 170, 132, 0.26)' },
  { bg: 'rgba(241, 170, 137, 0.18)', color: '#a65d3d', border: 'rgba(241, 170, 137, 0.28)' },
  { bg: 'rgba(142, 186, 201, 0.18)', color: '#4b7488', border: 'rgba(142, 186, 201, 0.28)' },
  { bg: 'rgba(217, 192, 140, 0.2)', color: '#8b6b24', border: 'rgba(217, 192, 140, 0.28)' },
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
        {status === 'queued' ? 'Queued' : 'Searching'}
      </span>
    )
  }
  if (status === 'completed') return <span className="status-pill completed">Completed</span>
  if (status === 'failed') return <span className="status-pill failed">Needs attention</span>
  return <span className="status-pill idle">Ready</span>
}

function KeywordChips({ terms, variant = 'default', emptyLabel = 'None' }) {
  if (!Array.isArray(terms) || terms.length === 0) {
    return <div className="chips-empty">{emptyLabel}</div>
  }

  return (
    <div className="chip-group">
      {terms.map((term, index) => (
        <span key={`${term}-${index}`} className={`chip chip-${variant}`}>
          {term}
        </span>
      ))}
    </div>
  )
}

function TraceGroup({ title, subtitle, terms, variant = 'default' }) {
  return (
    <div className="trace-group">
      <div className="trace-group-head">
        <h4>{title}</h4>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      <KeywordChips terms={terms} variant={variant} emptyLabel="No terms available" />
    </div>
  )
}

export default function App() {
  const [configYaml, setConfigYaml] = useState(DEFAULT_YAML)
  const [maxPages, setMaxPages] = useState(10)

  const [runId, setRunId] = useState(null)
  const [status, setStatus] = useState(null)
  const [message, setMessage] = useState(null)
  const [summary, setSummary] = useState(null)
  const [keywordExpansions, setKeywordExpansions] = useState(null)
  const [expansionTrace, setExpansionTrace] = useState(null)

  const [rows, setRows] = useState([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const limit = 50

  const [busy, setBusy] = useState(false)
  const [showSuggestModal, setShowSuggestModal] = useState(false)
  const [topicDescription, setTopicDescription] = useState('')
  const [suggestingKeywords, setSuggestingKeywords] = useState(false)
  const [suggestedConfig, setSuggestedConfig] = useState(null)

  const yamlValid = useMemo(() => {
    try {
      yaml.load(configYaml)
      return true
    } catch {
      return false
    }
  }, [configYaml])

  const topicNames = useMemo(() => {
    try {
      const parsed = yaml.load(configYaml)
      return (parsed?.topics || []).map((t) => t.name).filter(Boolean)
    } catch {
      return []
    }
  }, [configYaml])

  const statusDescription = useMemo(() => {
    if (busy || status === 'running') return 'Running NIH search and local filtering.'
    if (status === 'queued') return 'Your search is queued and will begin shortly.'
    if (status === 'completed') return 'Results are ready to review and export.'
    if (status === 'failed') return message || 'The run failed. Review the message below.'
    return 'Ready for a new search run.'
  }, [busy, message, status])

  function topicColor(name) {
    const idx = topicNames.indexOf(name)
    return TOPIC_COLORS[(idx >= 0 ? idx : 0) % TOPIC_COLORS.length]
  }

  async function startRun() {
    setBusy(true)
    setMessage(null)
    setSummary(null)
    setKeywordExpansions(null)
    setExpansionTrace(null)
    setRows([])
    setTotal(0)
    setOffset(0)
    try {
      const resp = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_yaml: configYaml, max_pages: maxPages || null }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      setRunId(data.run_id)
      setStatus(data.status)
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
      if (data.status === 'completed') {
        await loadResults(id, 0)
        return
      }
      if (data.status === 'failed') return
      await new Promise((r) => setTimeout(r, 1000))
    }
    throw new Error('Timed out waiting for run completion')
  }

  async function loadResults(id, newOffset) {
    const resp = await fetch(`/api/runs/${id}/results?offset=${newOffset}&limit=${limit}`)
    if (!resp.ok) throw new Error(await resp.text())
    const data = await resp.json()
    setOffset(data.offset)
    setTotal(data.total)
    setRows(data.items || [])
    setSummary(data.summary || null)
  }

  async function downloadCsv() {
    if (!runId) return
    const resp = await fetch(`/api/runs/${runId}/export.csv`)
    if (!resp.ok) throw new Error(await resp.text())
    const bytes = await resp.arrayBuffer()
    downloadBlob(bytes, `nih_outreach_${runId.slice(0, 8)}.csv`, 'text/csv')
  }

  function onUploadYaml(e) {
    const f = e.target.files?.[0]
    if (!f) return
    const reader = new FileReader()
    reader.onload = () => setConfigYaml(String(reader.result || ''))
    reader.readAsText(f)
  }

  function fmtDate(s) {
    if (!s) return ''
    return s.slice(0, 10)
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
          max_topic_terms: 10,
        }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      setSuggestedConfig(data)
    } catch (e) {
      alert(`Failed to suggest keywords: ${e.message}`)
    } finally {
      setSuggestingKeywords(false)
    }
  }

  function applyKeywords() {
    try {
      const parsed = yaml.load(configYaml)
      if (!parsed.query) parsed.query = {}
      if (!parsed.topics) parsed.topics = []

      parsed.query.broad_keywords = suggestedConfig.broad_keywords

      const topicName = topicDescription.slice(0, 50)
      if (parsed.topics.length === 0) {
        parsed.topics.push({
          name: topicName,
          include_any: suggestedConfig.topic_terms,
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
      alert(`Failed to update config: ${e.message}`)
    }
  }

  const canPrev = offset > 0
  const canNext = offset + limit < total
  const page = Math.floor(offset / limit) + 1
  const pages = Math.ceil(total / limit) || 1

  const summaryCards = [
    { label: 'Unique PIs', value: total },
    { label: 'Project records', value: summary?.matched_project_count ?? 0 },
    { label: 'Topics matched', value: Object.keys(summary?.counts_by_topic || {}).length },
    { label: 'Institutes (ICs)', value: Object.keys(summary?.counts_by_admin_ic || {}).length },
  ]

  return (
    <div className="app-shell">
      <div className="page-glow page-glow-one" />
      <div className="page-glow page-glow-two" />

      <div className="app">
        <header className="hero-card">
          <div className="hero-copy">
            <span className="eyebrow">Clean research dashboard</span>
            <h1>NIH RePORTER PI Finder</h1>
            <p className="hero-subtitle">
              Semantic MeSH-powered researcher discovery for Health TechQuity outreach.
            </p>
            <p className="hero-description">
              Build a topic-focused NIH search, expand terms thoughtfully, and review outreach-ready
              principal investigator results in one calm workspace.
            </p>
            <div className="hero-metrics">
              <div className="hero-metric">
                <span className="hero-metric-value">YAML</span>
                <span className="hero-metric-label">Config-driven workflow</span>
              </div>
              <div className="hero-metric">
                <span className="hero-metric-value">MeSH</span>
                <span className="hero-metric-label">Lexical + semantic expansion</span>
              </div>
              <div className="hero-metric">
                <span className="hero-metric-value">CSV</span>
                <span className="hero-metric-label">Exportable PI outreach lists</span>
              </div>
            </div>
          </div>

          <div className="hero-aside">
            <div className="hero-badge">NIH RePORTER API</div>
            <div className="hero-status-card">
              <div className="hero-status-top">
                <span className="mini-label">Run status</span>
                <StatusPill status={status} busy={busy} />
              </div>
              <p>{statusDescription}</p>
              <div className="hero-status-meta">
                <span>Keyword expansion trace included</span>
                <span>CSV export preserved</span>
              </div>
            </div>
          </div>
        </header>

        <div className="top-grid">
          <section className="panel card yaml-card">
            <div className="panel-header">
              <div>
                <p className="card-title">Search Configuration</p>
                <h2>YAML editor</h2>
              </div>
              <span className={`yaml-status ${yamlValid ? 'valid' : 'invalid'}`}>
                <span className="yaml-dot" />
                {yamlValid ? 'Valid YAML' : 'Needs fixing'}
              </span>
            </div>

            <p className="panel-description">
              Tune broad NIH search terms, MeSH expansion, and local topic filters. The editor keeps
              the full workflow transparent and reproducible.
            </p>

            <div className="yaml-toolbar">
              <label className="file-label">
                Upload YAML
                <input type="file" accept=".yaml,.yml" onChange={onUploadYaml} />
              </label>
              <button className="btn btn-soft" onClick={() => setShowSuggestModal(true)} type="button">
                Suggest Keywords
              </button>
              <button className="btn btn-ghost" onClick={() => setConfigYaml(DEFAULT_YAML)} type="button">
                Reset Example
              </button>
            </div>

            <div className="editor-shell">
              <div className="editor-toolbar">
                <span className="editor-pill">config.example-inspired</span>
                <span className="editor-hint">Stage 1 query + Stage 2 topic rules</span>
              </div>
              <textarea value={configYaml} onChange={(e) => setConfigYaml(e.target.value)} spellCheck={false} />
            </div>
          </section>

          <aside className="run-panel">
            <section className="panel card">
              <div className="panel-header">
                <div>
                  <p className="card-title">Run Settings</p>
                  <h2>Launch a search</h2>
                </div>
                <StatusPill status={status} busy={busy} />
              </div>

              <p className="panel-description">
                Start with a modest page count for faster iteration, then scale up once the topic
                rules look right.
              </p>

              <label className="field-label">Max pages (500 projects per page)</label>
              <input
                type="number"
                min={1}
                value={maxPages}
                onChange={(e) => setMaxPages(Number(e.target.value))}
              />
              <p className="field-hint">
                NIH rate-limit is about one request per second. Small pilot runs are easiest to inspect.
              </p>

              <div className="action-stack">
                <button className="btn btn-primary" onClick={startRun} disabled={!yamlValid || busy}>
                  {busy ? (
                    <>
                      <span className="spinner spinner-light" />
                      Searching
                    </>
                  ) : (
                    'Start Search'
                  )}
                </button>

                <button
                  className="btn btn-accent"
                  onClick={() => setShowSuggestModal(true)}
                  type="button"
                >
                  Suggest Keywords
                </button>

                <button
                  className="btn btn-secondary"
                  onClick={downloadCsv}
                  disabled={!runId || status !== 'completed'}
                >
                  Export CSV
                </button>
              </div>

              {message && !busy ? <div className="notice notice-error">{message}</div> : null}

              <div className="status-detail-card">
                <span className="mini-label">Progress note</span>
                <p>{statusDescription}</p>
              </div>
            </section>

            {keywordExpansions && Object.keys(keywordExpansions).length > 0 ? (
              <section className="panel card">
                <div className="panel-header compact">
                  <div>
                    <p className="card-title">AI Expansion</p>
                    <h2>Keyword additions</h2>
                  </div>
                </div>

                <div className="trace-grid">
                  {Object.entries(keywordExpansions).map(([original, expanded]) => (
                    <div key={original} className="trace-item-card">
                      <div className="trace-item-head">
                        <span className="trace-item-label">{original}</span>
                        <span className="trace-item-arrow">AI</span>
                      </div>
                      <KeywordChips terms={expanded} variant="accent" emptyLabel="No AI additions" />
                    </div>
                  ))}
                </div>
              </section>
            ) : null}

            {expansionTrace ? (
              <section className="panel card">
                <div className="panel-header compact">
                  <div>
                    <p className="card-title">Search Expansion Trace</p>
                    <h2>How your search evolved</h2>
                  </div>
                </div>

                <div className="trace-layout">
                  <TraceGroup
                    title="Original keywords"
                    subtitle="Starting broad terms from your YAML configuration."
                    terms={expansionTrace.original_keywords}
                    variant="neutral"
                  />

                  <div className="trace-group">
                    <div className="trace-group-head">
                      <h4>Lexical MeSH expansion</h4>
                      <p>{expansionTrace.mesh?.enabled ? 'Enabled' : 'Disabled'}</p>
                    </div>
                    <div className="trace-grid">
                      {Object.entries(expansionTrace.mesh?.terms_by_keyword || {}).map(([original, expanded]) => (
                        <div key={original} className="trace-item-card">
                          <div className="trace-item-head">
                            <span className="trace-item-label">{original}</span>
                            <span className="trace-item-arrow">MeSH</span>
                          </div>
                          <KeywordChips
                            terms={Array.isArray(expanded) ? expanded : []}
                            variant="teal"
                            emptyLabel="No additional MeSH terms"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="trace-group">
                    <div className="trace-group-head">
                      <h4>Semantic MeSH expansion</h4>
                      <p>{expansionTrace.semantic?.enabled ? 'Enabled' : 'Disabled'}</p>
                    </div>
                    {expansionTrace.semantic?.query ? (
                      <div className="semantic-query">Query: {expansionTrace.semantic.query}</div>
                    ) : null}
                    {expansionTrace.semantic?.error ? (
                      <div className="notice notice-warning">{expansionTrace.semantic.error}</div>
                    ) : null}
                    <KeywordChips
                      terms={expansionTrace.semantic?.expanded_terms || []}
                      variant="sage"
                      emptyLabel="No semantic expansion terms"
                    />
                    {expansionTrace.semantic?.concepts?.length ? (
                      <div className="concept-list">
                        {expansionTrace.semantic.concepts.map((concept, index) => (
                          <div key={`${concept.mesh_id}-${index}`} className="concept-card">
                            <div className="concept-top">
                              <span className="concept-name">{concept.preferred_name}</span>
                              <span className="concept-score">
                                {typeof concept.score === 'number' ? concept.score.toFixed(2) : concept.score}
                              </span>
                            </div>
                            <div className="concept-meta">
                              <span>{concept.mesh_id}</span>
                              {Array.isArray(concept.tree_numbers) && concept.tree_numbers.length ? (
                                <span>{concept.tree_numbers.join(', ')}</span>
                              ) : null}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </div>

                  <TraceGroup
                    title="Final NIH search terms"
                    subtitle="These are the terms sent to the NIH RePORTER API."
                    terms={expansionTrace.final_keywords}
                    variant="coral"
                  />
                </div>
              </section>
            ) : null}

            {summary ? (
              <section className="panel card">
                <div className="panel-header compact">
                  <div>
                    <p className="card-title">Results Summary</p>
                    <h2>Run snapshot</h2>
                  </div>
                </div>

                <div className="stats-grid">
                  {summaryCards.map((item) => (
                    <div key={item.label} className="stat-card">
                      <div className="stat-value">{item.value}</div>
                      <div className="stat-label">{item.label}</div>
                    </div>
                  ))}
                </div>

                {summary.counts_by_topic && Object.keys(summary.counts_by_topic).length > 0 ? (
                  <div className="topic-summary-list">
                    {Object.entries(summary.counts_by_topic).map(([name, count]) => {
                      const c = topicColor(name)
                      return (
                        <div key={name} className="topic-summary-row">
                          <span
                            className="topic-tag"
                            style={{ background: c.bg, color: c.color, borderColor: c.border }}
                          >
                            {name}
                          </span>
                          <span className="topic-summary-count">{count} projects</span>
                        </div>
                      )
                    })}
                  </div>
                ) : null}
              </section>
            ) : null}
          </aside>
        </div>

        <section className="results-section">
          <div className="results-header">
            <div className="results-heading">
              <p className="card-title">Search Results</p>
              <h2>Outreach-ready PI results</h2>
              <p>
                Review matched investigators, institutions, topic assignments, and project context
                before exporting.
              </p>
            </div>

            <div className="results-meta">
              {total > 0 ? <span className="total-badge">{total} PIs found</span> : null}
              {total > 0 ? (
                <div className="pagination">
                  <span className="page-info">
                    Page {page} of {pages}
                  </span>
                  <button className="btn-page" disabled={!canPrev} onClick={() => loadResults(runId, offset - limit)}>
                    Prev
                  </button>
                  <button className="btn-page" disabled={!canNext} onClick={() => loadResults(runId, offset + limit)}>
                    Next
                  </button>
                </div>
              ) : null}
            </div>
          </div>

          <div className="table-card">
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
                          <div className="empty-state-icon">🔎</div>
                          <div className="empty-state-title">
                            {status === 'completed' ? 'No matching results yet' : 'Ready when you are'}
                          </div>
                          <div className="empty-state-text">
                            {status === 'completed'
                              ? 'No results matched the current topic rules. Try broadening Stage 1 keywords or relaxing local filters.'
                              : 'Refine the YAML configuration above, then start a search to populate the table.'}
                          </div>
                        </div>
                      </td>
                    </tr>
                  ) : (
                    rows.map((r, idx) => {
                      const location = [r.organization_city, r.organization_state, r.organization_country]
                        .filter(Boolean)
                        .join(', ')
                      const startD = fmtDate(r.project_start_date)
                      const endD = fmtDate(r.project_end_date)
                      const rowKey = r.pi_profile_id || `${r.pi_name}-${r.organization_name}-${idx}`

                      return (
                        <tr key={rowKey}>
                          <td className="col-pi">
                            <div className="pi-name">
                              {r.pi_last_name && r.pi_first_name
                                ? `${r.pi_last_name}, ${r.pi_first_name}`
                                : r.pi_name || <span className="empty-val">—</span>}
                            </div>
                            {r.pi_profile_id ? (
                              <div className="pi-id">
                                <a
                                  href={`https://reporter.nih.gov/pi-details/${r.pi_profile_id}`}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="subtle-link"
                                >
                                  Profile ↗
                                </a>
                              </div>
                            ) : null}
                          </td>

                          <td className="col-email">
                            {r.pi_email ? (
                              <a href={`mailto:${r.pi_email}`} className="email-link">
                                {r.pi_email}
                              </a>
                            ) : (
                              <span className="empty-val">—</span>
                            )}
                          </td>

                          <td className="col-org">
                            <div className="org-name">
                              {r.organization_name || <span className="empty-val">—</span>}
                            </div>
                            {location ? <div className="org-loc">{location}</div> : null}
                          </td>

                          <td className="col-ic">
                            {r.admin_ic ? <span className="ic-tag">{r.admin_ic}</span> : <span className="empty-val">—</span>}
                          </td>

                          <td className="col-fys">
                            {Array.isArray(r.fiscal_years) && r.fiscal_years.length ? (
                              r.fiscal_years.map((fy) => (
                                <span key={fy} className="fy-pill">
                                  {fy}
                                </span>
                              ))
                            ) : (
                              <span className="empty-val">—</span>
                            )}
                          </td>

                          <td className="col-num">
                            {Array.isArray(r.project_numbers) && r.project_numbers.length ? (
                              r.project_numbers.map((n, i) => {
                                const appl = r.project_ids?.[i]
                                return appl ? (
                                  <a
                                    key={i}
                                    href={`https://reporter.nih.gov/project-details/${appl}`}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="proj-num"
                                  >
                                    {n} ↗
                                  </a>
                                ) : (
                                  <span key={i} className="proj-num muted-text">
                                    {n}
                                  </span>
                                )
                              })
                            ) : (
                              <span className="empty-val">—</span>
                            )}
                          </td>

                          <td className="col-date">
                            {startD || endD ? (
                              <div className="date-range">
                                <span>{startD || '—'}</span>
                                <span className="date-sep">to</span>
                                <span>{endD || '—'}</span>
                              </div>
                            ) : (
                              <span className="empty-val">—</span>
                            )}
                          </td>

                          <td className="col-topic">
                            {Array.isArray(r.matched_topics) && r.matched_topics.length ? (
                              r.matched_topics.map((t, i) => {
                                const c = topicColor(t)
                                return (
                                  <span
                                    key={i}
                                    className="topic-tag"
                                    style={{ background: c.bg, color: c.color, borderColor: c.border }}
                                  >
                                    {t}
                                  </span>
                                )
                              })
                            ) : (
                              <span className="empty-val">—</span>
                            )}
                          </td>

                          <td className="col-title">
                            {Array.isArray(r.sample_project_titles) && r.sample_project_titles.length ? (
                              r.sample_project_titles.map((t, i) => (
                                <div key={i} className="proj-title">
                                  {t}
                                </div>
                              ))
                            ) : (
                              <span className="empty-val">—</span>
                            )}
                          </td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {total > 0 ? (
            <div className="results-footer">
              <div className="results-footnote">
                Keep Stage 1 broad and use local topic rules to sharpen relevance before export.
              </div>
              <div className="pagination">
                <span className="page-info">
                  Page {page} of {pages} · {total} total
                </span>
                <button className="btn-page" disabled={!canPrev} onClick={() => loadResults(runId, offset - limit)}>
                  Prev
                </button>
                <button className="btn-page" disabled={!canNext} onClick={() => loadResults(runId, offset + limit)}>
                  Next
                </button>
              </div>
            </div>
          ) : null}
        </section>

        <p className="footer-tip">
          <strong>Tip:</strong> Broad NIH terms improve recall, while topic rules and MeSH traces keep the final list explainable.
        </p>

        {showSuggestModal ? (
          <div className="modal-overlay" onClick={() => setShowSuggestModal(false)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <div>
                  <p className="card-title">AI Keyword Suggester</p>
                  <h3>Describe the outreach topic</h3>
                </div>
                <button className="icon-button" onClick={() => setShowSuggestModal(false)} type="button">
                  ×
                </button>
              </div>

              <p className="modal-copy">
                Generate broad NIH search terms and local topic matching phrases from a short natural-language prompt.
              </p>

              <label className="field-label">Research topic description</label>
              <textarea
                value={topicDescription}
                onChange={(e) => setTopicDescription(e.target.value)}
                placeholder="e.g., AI and machine learning applications in reducing health disparities"
                className="modal-textarea"
              />

              <div className="modal-actions">
                <button
                  className="btn btn-primary"
                  onClick={suggestKeywords}
                  disabled={!topicDescription.trim() || suggestingKeywords}
                >
                  {suggestingKeywords ? 'Generating' : 'Generate Keywords'}
                </button>
                <button className="btn btn-ghost" onClick={() => setShowSuggestModal(false)} type="button">
                  Cancel
                </button>
              </div>

              {suggestedConfig && (suggestedConfig.broad_keywords?.length > 0 || suggestedConfig.topic_terms?.length > 0) ? (
                <div className="modal-results">
                  {suggestedConfig.broad_keywords?.length > 0 ? (
                    <div className="suggestion-block">
                      <h4>Broad keywords for NIH search</h4>
                      <KeywordChips terms={suggestedConfig.broad_keywords} variant="teal" />
                    </div>
                  ) : null}

                  {suggestedConfig.topic_terms?.length > 0 ? (
                    <div className="suggestion-block">
                      <h4>Topic matching terms for local filtering</h4>
                      <KeywordChips terms={suggestedConfig.topic_terms} variant="coral" />
                    </div>
                  ) : null}

                  <button className="btn btn-primary" onClick={applyKeywords}>
                    Apply to Config
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
