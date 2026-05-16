from __future__ import annotations

import json
from openai import OpenAI
from .cache import make_cache


class KeywordExpander:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        # OpenAI() automatically reads OPENAI_API_KEY from the environment when
        # api_key is None, so we always create the client and let the SDK raise
        # a clear AuthenticationError if no key is available at call time.
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache = make_cache()

    def expand_keywords(
        self,
        keywords: list[str],
        context: str = "biomedical research and health disparities",
        max_expansions: int = 5
    ) -> dict[str, list[str]]:
        if not self.client:
            return {kw: [kw] for kw in keywords}
        
        cache_key = f"keyword_expansion:{self.model}:{context}:{max_expansions}:{','.join(sorted(keywords))}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        prompt = f"""You are a biomedical research expert. For each keyword below, generate up to {max_expansions} related terms, synonyms, acronyms, and common variations relevant to {context}.

Keywords: {', '.join(keywords)}

Return a JSON object where each key is an original keyword and the value is a list of expanded terms (including the original). Focus on:
- Common synonyms and related terms
- Acronyms and their expansions
- Common misspellings or variations
- Related concepts that researchers might use

Example format:
{{
  "diabetes": ["diabetes", "diabetic", "diabetes mellitus", "T2D", "type 2 diabetes"],
  "disparities": ["disparities", "disparity", "inequities", "inequality", "inequalities"]
}}

Return ONLY valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that expands biomedical research keywords. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            expansions = json.loads(content)
            
            for kw in keywords:
                if kw not in expansions:
                    expansions[kw] = [kw]
                elif kw not in expansions[kw]:
                    expansions[kw].insert(0, kw)
            
            self.cache.set(cache_key, expansions, expire=86400 * 30)
            return expansions
            
        except Exception as e:
            print(f"Keyword expansion failed: {e}")
            return {kw: [kw] for kw in keywords}
    
    def expand_query_keywords(
        self,
        keywords: list[str],
        enabled: bool = True,
        context: str = "biomedical research and health disparities",
        max_expansions: int = 5
    ) -> tuple[list[str], dict[str, list[str]]]:
        if not enabled:
            return keywords, {kw: [kw] for kw in keywords}
        
        expansions = self.expand_keywords(keywords, context, max_expansions)
        
        expanded_keywords = []
        for kw in keywords:
            expanded_keywords.extend(expansions.get(kw, [kw]))
        
        unique_expanded = []
        seen = set()
        for kw in expanded_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_expanded.append(kw)
        
        return unique_expanded, expansions
