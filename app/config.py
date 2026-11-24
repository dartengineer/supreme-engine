"""App settings, loaded from the .env file in the project root."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env also holds settings for other tools (e.g. POSTGRES_PASSWORD for Docker Compose).
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI CV Generator"
    database_url: str = "postgresql+psycopg://postgres:password@localhost:5432/cv_generator"
    secret_key: str = "change-me"  # used to sign login tokens; set a long random value in .env
    access_token_expire_minutes: int = 60 * 24  # 1 day

    # AI (Claude). Put ANTHROPIC_API_KEY in .env, never in code.
    anthropic_api_key: str | None = None
    ai_model: str = "claude-haiku-5-5"  # cheap and fast while developing

    # Load the embedding model in the background when the server starts, so the first request doesn't wait.
    warm_up_embeddings: bool = True

    # HTML -> PDF renderer (WeasyPrint in Docker, see pdf_renderer/). Listens on this machine only.
    pdf_renderer_url: str = "http://127.0.0.1:5001"


settings = Settings()