# AI-Powered Keyword Expansion

## Overview

The NIH RePORTER Outreach List Builder now includes **AI-powered keyword expansion** to improve search recall by automatically generating synonyms, related terms, acronyms, and common variations of your search keywords.

## Why Use AI Expansion?

When searching for research projects, you might miss relevant results because:
- Researchers use different terminology (e.g., "diabetes" vs "diabetic" vs "T2D")
- Acronyms and full forms are used interchangeably
- Synonyms and related concepts vary across fields
- Typos or spelling variations exist in project descriptions

AI expansion solves this by intelligently expanding your keywords before querying the NIH API.

## How It Works

1. **You provide** broad keywords (e.g., "health disparities", "technology")
2. **AI generates** related terms using GPT-4o-mini (or your chosen model)
3. **Expanded query** searches NIH with all variations
4. **Results show** which expansions were used for transparency

## Configuration

Add the `ai_expansion` section to your YAML config:

```yaml
query:
  fiscal_years: [2024, 2025]
  broad_keywords:
    - health disparities
    - diabetes
  
  ai_expansion:
    enabled: true  # Set to false to disable
    openai_api_key: "sk-..."  # Or set OPENAI_API_KEY env var
    model: gpt-4o-mini  # Options: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
    max_expansions_per_keyword: 5  # Max synonyms per keyword
    context: "biomedical research and health disparities"
```

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable/disable AI expansion |
| `openai_api_key` | string | `null` | OpenAI API key (or use env var) |
| `model` | string | `"gpt-4o-mini"` | OpenAI model to use |
| `max_expansions_per_keyword` | integer | `5` | Max expansions per keyword |
| `context` | string | `"biomedical research and health disparities"` | Domain context for better expansions |

## API Key Setup

### Option 1: Environment Variable (Recommended)
```bash
export OPENAI_API_KEY="sk-..."
```

### Option 2: Config File
```yaml
ai_expansion:
  enabled: true
  openai_api_key: "sk-..."
```

⚠️ **Security Note**: Never commit API keys to version control. Use environment variables or `.env` files (gitignored).

## Example Expansions

**Input:** `["health disparities", "diabetes"]`

**AI Output:**
```json
{
  "health disparities": [
    "health disparities",
    "health inequities",
    "health inequalities",
    "healthcare disparities",
    "health equity"
  ],
  "diabetes": [
    "diabetes",
    "diabetic",
    "diabetes mellitus",
    "T2D",
    "type 2 diabetes"
  ]
}
```

**NIH Query:** Searches for all 10 terms combined

## Cost Estimates

Using `gpt-4o-mini` (cheapest option):
- **~$0.01-0.03** per expansion request
- Expansions are **cached** for 30 days
- Typical search: **1 expansion request** (all keywords at once)

### Cost Comparison
| Model | Cost per 1M input tokens | Typical expansion cost |
|-------|-------------------------|----------------------|
| gpt-4o-mini | $0.15 | $0.01-0.03 |
| gpt-4o | $2.50 | $0.10-0.30 |
| gpt-3.5-turbo | $0.50 | $0.02-0.05 |

## Caching

Expansions are automatically cached to:
- **Reduce costs** (avoid re-running same expansions)
- **Improve speed** (instant retrieval from cache)
- **Cache location**: `.cache/` directory
- **Cache duration**: 30 days

## Frontend Display

When AI expansion is enabled, the frontend shows:
- 🤖 **AI Keyword Expansions** panel
- Original keyword → expanded terms mapping
- Full transparency of what was searched

## CLI Usage

```bash
# With AI expansion enabled in config
python -m app.cli \
  --config config_with_ai.yaml \
  --out-dir output/ \
  --max-pages 5

# Output includes keyword_expansions.json
ls output/
# results.json
# summary.json
# keyword_expansions.json  ← AI expansion details
# outreach.csv
```

## Best Practices

1. **Start with broad keywords** - AI will expand them
2. **Use domain context** - Improves expansion quality
3. **Monitor costs** - Check OpenAI usage dashboard
4. **Review expansions** - Frontend shows what was expanded
5. **Adjust max_expansions** - Balance recall vs query complexity

## Troubleshooting

### No expansions showing
- Check `enabled: true` in config
- Verify API key is set correctly
- Check backend logs for errors

### Unexpected expansions
- Adjust `context` field for better domain relevance
- Reduce `max_expansions_per_keyword`
- Try different model (gpt-4o for better quality)

### High costs
- Use `gpt-4o-mini` instead of `gpt-4o`
- Reduce `max_expansions_per_keyword`
- Cache is working automatically (check `.cache/`)

## Disabling AI Expansion

Set `enabled: false` or remove the `ai_expansion` section entirely:

```yaml
query:
  fiscal_years: [2024]
  broad_keywords:
    - health disparities
  # No ai_expansion section = disabled
```

## Technical Details

- **Implementation**: `backend/app/keyword_expander.py`
- **Model**: OpenAI Chat Completions API
- **Temperature**: 0.3 (balanced creativity)
- **Caching**: Diskcache with 30-day expiry
- **Deduplication**: Case-insensitive, preserves order
