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

const TOPIC_COLORS = [
  { bg: 'rgba(92, 158, 149, 0.16)', color: '#2f6f69', border: 'rgba(92, 158, 149, 0.24)' },
  { bg: 'rgba(135, 170, 132, 0.18)', color: '#557451', border: 'rgba(135, 170, 132, 0.26)' },
  { bg: 'rgba(241, 170, 137, 0.18)', color: '#a65d3d', border: 'rgba(241, 170, 137, 0.28)' },
  { bg: 'rgba(142, 186, 201, 0.18)', color: '#4b7488', border: 'rgba(142, 186, 201, 0.28)' },
  { bg: 'rgba(217, 192, 140, 0.2)', color: '#8b6b24', border: 'rgba(217, 192, 140, 0.28)' },
]

const SEARCH_FIELD_OPTIONS = [
  { value: 'all', label: 'All searchable text' },
  { value: 'projecttitle', label: 'Project title' },
  { value: 'abstracttext', label: 'Abstract text' },
  { value: 'terms', label: 'Terms' },
]

const OPERATOR_OPTIONS = [
  { value: 'or', label: 'Broad match (recommended)' },
  { value: 'and', label: 'Strict match' },
]

const DEFAULT_AI_CONTEXT = 'biomedical research and health disparities'

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

function parseCommaSeparated(input) {
  return input
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function dedupeByLabel(concepts) {
  const seen = new Set()
  const unique = []

  concepts.forEach((concept) => {
    const label = String(concept?.label || '').trim()
    if (!label) return
    const key = label.toLowerCase()
    if (seen.has(key)) return
    seen.add(key)
    unique.push({
      label,
      source: concept?.source || 'manual',
      mesh_id: concept?.mesh_id ?? null,
      score: concept?.score ?? null,
    })
  })

  return unique
}

function parseFiscalYears(input) {
  return parseCommaSeparated(input)
    .map((item) => Number(item))
    .filter((item) => Number.isInteger(item) && item > 0)
}

function buildTopicName(question, concepts) {
  const normalizedQuestion = String(question || '').trim()
  if (normalizedQuestion) return normalizedQuestion
  if (concepts.length > 0) return concepts.map((item) => item.label).join(' + ').slice(0, 80)
  return 'Research Topic'
}

function simpleFallbackConcepts(question, maxConcepts = 5) {
  const cleaned = String(question || '')
    .replace(/[^\w\s/-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  if (!cleaned) return []

  const phrases = []
  const parts = cleaned.split(/\b(?:and|or|for|with|in|on|to|of|the|a|an|using|via)\b/i)
  parts.forEach((part) => {
    const normalized = part.trim()
    if (normalized.length >= 3) phrases.push(normalized)
  })

  const tokens = cleaned
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 2)

  for (let index = 0; index < tokens.length; index += 1) {
    phrases.push(tokens[index])
    if (tokens[index + 1]) phrases.push(`${tokens[index]} ${tokens[index + 1]}`)
  }

  return dedupeByLabel(
    phrases.slice(0, maxConcepts).map((label) => ({
      label,
      source: 'fallback',
      mesh_id: null,
      score: null,
    }))
  ).slice(0, maxConcepts)
}

function createConfigFromBuilder(builder) {
  const broadKeywords = builder.concepts.length > 0
    ? builder.concepts.map((concept) => concept.label)
    : simpleFallbackConcepts(builder.researchQuestion, 5).map((concept) => concept.label)

  return {
    query: {
      fiscal_years: parseFiscalYears(builder.fiscalYearsText),
      broad_keywords: broadKeywords,
      text_search_field: builder.textSearchField,
      text_search_operator: builder.textSearchOperator,
      mesh_expansion: {
        enabled: builder.meshExpansionEnabled,
        max_terms_per_keyword: 15,
        include_entry_terms: true,
        include_tree_children: true,
        max_tree_depth: 1,
        fallback_to_original: true,
        cache_enabled: true,
      },
      semantic_expansion: {
        enabled: builder.semanticExpansionEnabled,
        top_k: 10,
        max_terms: 30,
        min_score: null,
        include_synonyms: true,
        require_existing_index: false,
      },
      ai_expansion: {
        enabled: builder.aiExpansionEnabled,
        openai_api_key: null,
        model: 'gpt-4o-mini',
        max_expansions_per_keyword: 5,
        context: builder.researchQuestion.trim() || DEFAULT_AI_CONTEXT,
      },
    },
    topics: [
      {
        name: buildTopicName(builder.researchQuestion, builder.concepts),
        include_any: broadKeywords,
        include_all: [],
        exclude_any: [],
        co_require_groups: [],
      },
    ],
  }
}

function normalizeConceptsFromYaml(config) {
  const broadKeywords = Array.isArray(config?.query?.broad_keywords) ? config.query.broad_keywords : []
  return dedupeByLabel(
    broadKeywords.map((label) => ({
      label,
      source: 'manual',
      mesh_id: null,
      score: null,
    }))
  )
}

function builderFromConfigObject(config) {
  const aiContext = config?.query?.ai_expansion?.context
  return {
    researchQuestion:
      typeof aiContext === 'string' && aiContext !== DEFAULT_AI_CONTEXT
        ? aiContext
        : config?.topics?.[0]?.name || '',
    concepts: normalizeConceptsFromYaml(config),
    fiscalYearsText: Array.isArray(config?.query?.fiscal_years)
      ? config.query.fiscal_years.join(', ')
      : '',
    textSearchField: config?.query?.text_search_field || 'all',
    textSearchOperator: config?.query?.text_search_operator || 'or',
    meshExpansionEnabled: Boolean(config?.query?.mesh_expansion?.enabled),
    semanticExpansionEnabled: Boolean(config?.query?.semantic_expansion?.enabled),
    aiExpansionEnabled: Boolean(config?.query?.ai_expansion?.enabled),
  }
}

function builderFromYaml(yamlText) {
  try {
    const parsed = yaml.load(yamlText)
    return builderFromConfigObject(parsed || {})
  } catch {
    return builderFromConfigObject({})
  }
}

function serializeBuilder(builder) {
  return yaml.dump(createConfigFromBuilder(builder), {
    indent: 2,
    lineWidth: -1,
    noRefs: true,
  })
}

function conceptSourceLabel(source) {
  if (source === 'semantic_mesh') return 'Semantic match'
  if (source === 'mesh_lookup') return 'Official MeSH'
  if (source === 'fallback') return 'Fallback'
  return 'Manual'
}

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

function ToggleField({ label, description, checked, onChange }) {
  return (
    <label className="toggle-card">
      <div className="toggle-copy">
        <span className="toggle-label">{label}</span>
        <span className="toggle-description">{description}</span>
      </div>
      <span className={`toggle-switch ${checked ? 'checked' : ''}`}>
        <input type="checkbox" checked={checked} onChange={onChange} />
        <span className="toggle-slider" />
      </span>
    </label>
  )
}

export default function App() {
  const initialBuilder = useMemo(() => builderFromYaml(DEFAULT_YAML), [])

  const [builderState, setBuilderState] = useState(initialBuilder)
  const [configYaml, setConfigYaml] = useState(DEFAULT_YAML)
  const [conceptInput, setConceptInput] = useState('')
  const [advancedOpen, setAdvancedOpen] = useState(false)
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
  const [generatingConcepts, setGeneratingConcepts] = useState(false)
  const [conceptStatus, setConceptStatus] = useState({ tone: 'idle', text: 'Generate concepts from a research question to start.' })

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

  const selectedConceptLabels = useMemo(
    () => builderState.concepts.map((concept) => concept.label),
    [builderState.concepts]
  )

  function topicColor(name) {
    const idx = topicNames.indexOf(name)
    return TOPIC_COLORS[(idx >= 0 ? idx : 0) % TOPIC_COLORS.length]
  }

  function updateBuilder(updater) {
    setBuilderState((current) => {
      const next = typeof updater === 'function' ? updater(current) : { ...current, ...updater }
      setConfigYaml(serializeBuilder(next))
      return next
    })
  }

  function addManualConcept(rawValue) {
    const terms = parseCommaSeparated(String(rawValue || ''))
    if (terms.length === 0) return

    updateBuilder((current) => ({
      ...current,
      concepts: dedupeByLabel([
        ...current.concepts,
        ...terms.map((label) => ({ label, source: 'manual', mesh_id: null, score: null })),
      ]),
    }))
    setConceptInput('')
  }

  function removeConcept(labelToRemove) {
    updateBuilder((current) => ({
      ...current,
      concepts: current.concepts.filter((concept) => concept.label !== labelToRemove),
    }))
  }

  function clearAllConcepts() {
    updateBuilder((current) => ({
      ...current,
      concepts: [],
    }))
    setConceptStatus({ tone: 'idle', text: 'Concepts cleared. Generate again or add your own.' })
  }

  async function generateConcepts() {
    const question = builderState.researchQuestion.trim()
    if (!question) {
      setConceptStatus({ tone: 'warning', text: 'Add a research question first.' })
      return
    }

    setGeneratingConcepts(true)
    setConceptStatus({ tone: 'loading', text: 'Generating concepts...' })

    try {
      const resp = await fetch('/api/concepts/suggest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 8 }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      const concepts = dedupeByLabel(data.concepts || [])

      if (concepts.length === 0) {
        setConceptStatus({ tone: 'warning', text: 'No concepts found; add manually or edit the question.' })
        updateBuilder((current) => ({ ...current, concepts: [] }))
        return
      }

      updateBuilder((current) => ({
        ...current,
        concepts,
      }))

      const onlyFallback = concepts.every((concept) => concept.source === 'fallback')
      const usedMeshFallback = concepts.some((concept) => concept.source === 'mesh_lookup')
      if (onlyFallback) {
        setConceptStatus({
          tone: 'warning',
          text: 'Could not generate MeSH-grounded concepts; using simple fallback suggestions.',
        })
      } else if (usedMeshFallback || data.fallback_used) {
        setConceptStatus({
          tone: 'success',
          text: 'Suggested concepts ready. Some concepts came from local MeSH lookup fallback.',
        })
      } else {
        setConceptStatus({ tone: 'success', text: 'Suggested concepts ready.' })
      }
    } catch (_error) {
      const fallback = simpleFallbackConcepts(question, 5)
      updateBuilder((current) => ({
        ...current,
        concepts: fallback,
      }))
      if (fallback.length > 0) {
        setConceptStatus({
          tone: 'warning',
          text: 'Could not generate MeSH-grounded concepts; using simple fallback suggestions.',
        })
      } else {
        setConceptStatus({ tone: 'error', text: 'No concepts found; add manually or edit the question.' })
      }
    } finally {
      setGeneratingConcepts(false)
    }
  }

  async function startRun() {
    let yamlToRun = configYaml
    if (builderState.concepts.length === 0 && builderState.researchQuestion.trim()) {
      const fallbackConcepts = simpleFallbackConcepts(builderState.researchQuestion, 5)
      if (fallbackConcepts.length > 0) {
        const nextBuilder = {
          ...builderState,
          concepts: fallbackConcepts,
        }
        setBuilderState(nextBuilder)
        yamlToRun = serializeBuilder(nextBuilder)
        setConfigYaml(yamlToRun)
        setConceptStatus({
          tone: 'warning',
          text: 'No generated concepts were selected, so the search used a simple fallback concept list.',
        })
      }
    }

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
        body: JSON.stringify({ config_yaml: yamlToRun, max_pages: maxPages || null }),
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
    for (let i = 0; i < 600; i += 1) {
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
      await new Promise((resolve) => setTimeout(resolve, 1000))
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
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const nextYaml = String(reader.result || '')
      setConfigYaml(nextYaml)
      try {
        const parsed = yaml.load(nextYaml)
        setBuilderState(builderFromConfigObject(parsed || {}))
      } catch {
        // Preserve the existing builder while invalid YAML is being reviewed.
      }
    }
    reader.readAsText(file)
  }

  function onYamlChange(nextYaml) {
    setConfigYaml(nextYaml)
    try {
      const parsed = yaml.load(nextYaml)
      setBuilderState(builderFromConfigObject(parsed || {}))
    } catch {
      // Preserve current builder state while YAML is invalid.
    }
  }

  function fmtDate(value) {
    if (!value) return ''
    return value.slice(0, 10)
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
    const combinedConcepts = dedupeByLabel([
      ...(suggestedConfig?.broad_keywords || []).map((label) => ({
        label,
        source: 'manual',
        mesh_id: null,
        score: null,
      })),
      ...(suggestedConfig?.topic_terms || []).map((label) => ({
        label,
        source: 'manual',
        mesh_id: null,
        score: null,
      })),
    ])

    updateBuilder((current) => ({
      ...current,
      researchQuestion: topicDescription.trim() || current.researchQuestion,
      concepts: combinedConcepts.length > 0 ? combinedConcepts : current.concepts,
    }))

    setConceptStatus({ tone: 'success', text: 'Suggested concepts ready. You can edit them before searching.' })
    setShowSuggestModal(false)
    setSuggestedConfig(null)
    setTopicDescription('')
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
              Ask a research question, generate grounded concepts, review them quickly, and launch an NIH
              search without touching YAML unless you want extra control.
            </p>
            <div className="hero-metrics">
              <div className="hero-metric">
                <span className="hero-metric-value">Question</span>
                <span className="hero-metric-label">Start with plain language</span>
              </div>
              <div className="hero-metric">
                <span className="hero-metric-value">Concepts</span>
                <span className="hero-metric-label">MeSH-grounded when available</span>
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
                <span>Search by question first</span>
                <span>Advanced YAML still available</span>
              </div>
            </div>
          </div>
        </header>

        <div className="top-grid">
          <div className="builder-column">
            <section className="panel card builder-card">
              <div className="panel-header">
                <div>
                  <p className="card-title">Search Builder</p>
                  <h2>Start with a research question</h2>
                </div>
                <span className="builder-mode-pill">No YAML required</span>
              </div>

              <p className="panel-description">
                Type a research topic, generate suggested concepts, adjust the concept chips if needed, and
                then run the NIH search.
              </p>

              <div className="form-card question-card">
                <label className="field-label">Research topic or question</label>
                <textarea
                  className="builder-textarea"
                  value={builderState.researchQuestion}
                  onChange={(e) =>
                    updateBuilder((current) => ({
                      ...current,
                      researchQuestion: e.target.value,
                    }))
                  }
                  placeholder="AI for diabetes care in underserved populations"
                />
                <div className="question-actions">
                  <button className="btn btn-primary" type="button" onClick={generateConcepts} disabled={generatingConcepts}>
                    {generatingConcepts ? (
                      <>
                        <span className="spinner spinner-light" />
                        Generating Concepts
                      </>
                    ) : (
                      'Generate Concepts'
                    )}
                  </button>
                  <button
                    className="btn btn-soft"
                    type="button"
                    onClick={() => setShowSuggestModal(true)}
                  >
                    Suggest Keywords
                  </button>
                </div>
              </div>

              <div className="concept-panel">
                <div className="panel-header compact">
                  <div>
                    <p className="card-title">Suggested Concepts</p>
                    <h2>Review and edit before search</h2>
                  </div>
                  {selectedConceptLabels.length > 0 ? (
                    <span className="concept-count-pill">{selectedConceptLabels.length} selected</span>
                  ) : null}
                </div>

                <div className={`notice concept-notice concept-notice-${conceptStatus.tone}`}>
                  {conceptStatus.text}
                </div>

                {builderState.concepts.length > 0 ? (
                  <div className="selected-concepts-wrap">
                    {builderState.concepts.map((concept) => (
                      <span key={concept.label} className="concept-chip">
                        <span className="concept-chip-label">{concept.label}</span>
                        <span className="concept-chip-source">{conceptSourceLabel(concept.source)}</span>
                        <button type="button" onClick={() => removeConcept(concept.label)} aria-label={`Remove ${concept.label}`}>
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                ) : (
                  <div className="chips-empty chips-empty-large">
                    No concepts yet. Generate suggestions from the question above or add your own manually.
                  </div>
                )}

                <div className="manual-concept-row">
                  <div className="chip-input-shell">
                    <div className="chip-input-list">
                      <input
                        type="text"
                        value={conceptInput}
                        onChange={(e) => setConceptInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ',') {
                            e.preventDefault()
                            addManualConcept(conceptInput)
                          }
                        }}
                        onBlur={() => addManualConcept(conceptInput)}
                        placeholder="telemedicine, health disparities, artificial intelligence"
                      />
                    </div>
                  </div>
                  <button className="btn btn-secondary" type="button" onClick={() => addManualConcept(conceptInput)}>
                    Add Concept
                  </button>
                </div>

                <div className="concept-toolbar">
                  <button className="btn btn-ghost" type="button" onClick={generateConcepts} disabled={generatingConcepts}>
                    Regenerate Concepts
                  </button>
                  <button className="btn btn-ghost" type="button" onClick={clearAllConcepts}>
                    Clear All
                  </button>
                </div>
              </div>
            </section>

            <section className="panel card advanced-card">
              <button
                className="advanced-toggle"
                type="button"
                onClick={() => setAdvancedOpen((current) => !current)}
                aria-expanded={advancedOpen}
              >
                <div>
                  <p className="card-title">Advanced Mode</p>
                  <h2>Advanced Search Options</h2>
                </div>
                <span className={`advanced-chevron ${advancedOpen ? 'open' : ''}`}>⌄</span>
              </button>

              <p className="panel-description advanced-description">
                Adjust NIH matching behavior, fiscal years, expansion settings, and the raw YAML only if you need
                more control.
              </p>

              {advancedOpen ? (
                <div className="advanced-content">
                  <div className="form-grid two-up">
                    <div className="form-card">
                      <label className="field-label">Where should NIH search?</label>
                      <select
                        value={builderState.textSearchField}
                        onChange={(e) =>
                          updateBuilder((current) => ({
                            ...current,
                            textSearchField: e.target.value,
                          }))
                        }
                      >
                        {SEARCH_FIELD_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="form-card">
                      <label className="field-label">Match style</label>
                      <select
                        value={builderState.textSearchOperator}
                        onChange={(e) =>
                          updateBuilder((current) => ({
                            ...current,
                            textSearchOperator: e.target.value,
                          }))
                        }
                      >
                        {OPERATOR_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="form-card">
                      <label className="field-label">Fiscal years</label>
                      <input
                        type="text"
                        value={builderState.fiscalYearsText}
                        onChange={(e) =>
                          updateBuilder((current) => ({
                            ...current,
                            fiscalYearsText: e.target.value,
                          }))
                        }
                        placeholder="2024, 2025"
                      />
                    </div>

                    <div className="form-card">
                      <label className="field-label">Max pages</label>
                      <input type="number" min={1} value={maxPages} onChange={(e) => setMaxPages(Number(e.target.value))} />
                    </div>
                  </div>

                  <div className="toggle-stack">
                    <ToggleField
                      label="Expand using medical terminology"
                      description="Adds official medical synonyms and related National Library of Medicine terms."
                      checked={builderState.meshExpansionEnabled}
                      onChange={(e) =>
                        updateBuilder((current) => ({
                          ...current,
                          meshExpansionEnabled: e.target.checked,
                        }))
                      }
                    />
                    <ToggleField
                      label="Find conceptually similar topics"
                      description="Uses semantic matching over MeSH concepts when a local index is available."
                      checked={builderState.semanticExpansionEnabled}
                      onChange={(e) =>
                        updateBuilder((current) => ({
                          ...current,
                          semanticExpansionEnabled: e.target.checked,
                        }))
                      }
                    />
                    <ToggleField
                      label="AI-assisted query improvement"
                      description="Suggests extra search phrasing using a language model when configured."
                      checked={builderState.aiExpansionEnabled}
                      onChange={(e) =>
                        updateBuilder((current) => ({
                          ...current,
                          aiExpansionEnabled: e.target.checked,
                        }))
                      }
                    />
                  </div>

                  <div className="yaml-toolbar">
                    <label className="file-label">
                      Upload YAML
                      <input type="file" accept=".yaml,.yml" onChange={onUploadYaml} />
                    </label>
                    <button className="btn btn-ghost" onClick={() => {
                      setBuilderState(initialBuilder)
                      setConfigYaml(DEFAULT_YAML)
                      setConceptStatus({ tone: 'idle', text: 'Builder reset to the example configuration.' })
                    }} type="button">
                      Reset Example
                    </button>
                    <span className={`yaml-status ${yamlValid ? 'valid' : 'invalid'}`}>
                      <span className="yaml-dot" />
                      {yamlValid ? 'Valid YAML' : 'Needs fixing'}
                    </span>
                  </div>

                  <div className="editor-shell">
                    <div className="editor-toolbar">
                      <span className="editor-pill">Backend config preview</span>
                      <span className="editor-hint">If you edit YAML here, it becomes the source of truth for this run.</span>
                    </div>
                    <textarea value={configYaml} onChange={(e) => onYamlChange(e.target.value)} spellCheck={false} />
                  </div>
                </div>
              ) : null}
            </section>
          </div>

          <aside className="run-panel">
            <section className="panel card">
              <div className="panel-header">
                <div>
                  <p className="card-title">Actions</p>
                  <h2>Run and export</h2>
                </div>
                <StatusPill status={status} busy={busy} />
              </div>

              <p className="panel-description">
                When the concept chips look right, start the NIH search. You can export CSV after the run completes.
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

                <button className="btn btn-accent" onClick={() => setShowSuggestModal(true)} type="button">
                  Suggest Keywords
                </button>

                <button className="btn btn-secondary" onClick={downloadCsv} disabled={!runId || status !== 'completed'}>
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
                    subtitle="Starting broad terms from your configuration."
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
                          <span className="topic-tag" style={{ background: c.bg, color: c.color, borderColor: c.border }}>
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
                Review matched investigators, institutions, topic assignments, and project context before exporting.
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
                              ? 'No results matched the current topic rules. Try editing the concepts or relaxing local filters.'
                              : 'Generate concepts from a research question above, then start a search to populate this table.'}
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
                                  <span key={i} className="topic-tag" style={{ background: c.bg, color: c.color, borderColor: c.border }}>
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
                Broad concept chips improve recall; local filtering and traces keep the result set explainable.
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
          <strong>Tip:</strong> Use Generate Concepts first, then remove or add chips until the search intent feels right.
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
                placeholder="AI for diabetes care in underserved populations"
                className="modal-textarea"
              />

              <div className="modal-actions">
                <button className="btn btn-primary" onClick={suggestKeywords} disabled={!topicDescription.trim() || suggestingKeywords}>
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
                    Apply to Search Builder
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
