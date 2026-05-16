import json
from typing import List, Optional, Dict, Any
from openai import OpenAI
from .cache import make_cache

class KeywordSuggester:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.cache = make_cache()
    
    def suggest_config(
        self, 
        topic_description: str,
        context: str = "biomedical research and health disparities",
        max_broad_keywords: int = 3,
        max_topic_terms: int = 10
    ) -> Dict[str, Any]:
        """
        Given a topic description, suggest a complete search configuration.
        
        Args:
            topic_description: Description of what you're looking for
            context: Domain context
            max_broad_keywords: Maximum number of broad keywords for NIH query
            max_topic_terms: Maximum number of terms for topic matching
            
        Returns:
            Dict with 'broad_keywords' and 'topic_terms'
        """
        if not self.client:
            return {"broad_keywords": [], "topic_terms": []}
        
        cache_key = f"config_suggestion_v2:{self.model}:{context}:{max_broad_keywords}:{max_topic_terms}:{topic_description}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        prompt = f"""You are a biomedical research expert. Based on the topic description, suggest search terms for the NIH RePORTER database.

Topic: {topic_description}
Context: {context}

The search has TWO stages:
1. BROAD KEYWORDS: Used to query NIH API (keep these general, 2-3 terms max)
2. TOPIC TERMS: Used to filter results locally (can be more specific, include variations)

Return a JSON object with this structure:
{{
  "broad_keywords": ["general term 1", "general term 2"],
  "topic_terms": ["specific term 1", "variation 1", "acronym 1", "related term 1", ...]
}}

Guidelines:
- broad_keywords: Core concepts only (e.g., "health disparities", "artificial intelligence")
- topic_terms: Include synonyms, acronyms, related methodologies, specific terms
- topic_terms should be diverse enough to catch relevant projects

Return ONLY the JSON object, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that suggests biomedical research configuration. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            config = json.loads(content)
            
            if not isinstance(config, dict):
                return {"broad_keywords": [], "topic_terms": []}
            
            # Ensure structure
            result = {
                "broad_keywords": config.get("broad_keywords", [])[:max_broad_keywords],
                "topic_terms": config.get("topic_terms", [])[:max_topic_terms]
            }
            
            # Cache for 30 days
            self.cache.set(cache_key, result, expire=86400 * 30)
            return result
            
        except Exception as e:
            print(f"Config suggestion failed: {e}")
            return {"broad_keywords": [], "topic_terms": []}
