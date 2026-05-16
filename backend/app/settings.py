from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OUTREACH_", env_file=".env", extra="ignore")

    reporter_api_base_url: str = "https://api.reporter.nih.gov/v2"
    reporter_rate_limit_seconds: float = 1.0

    cache_dir: str = ".cache"
    cache_ttl_seconds: int = 60 * 60 * 24 * 30

    server_base_url: str = "http://localhost:8000"


settings = Settings()
