"""App settings, loaded from the .env file in the project root."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "AI CV Generator"
    database_url: str = "postgresql+psycopg://postgres:password@localhost:5432/cv_generator"
    secret_key: str = "change-me"  # used to sign login tokens; set a long random value in .env
    access_token_expire_minutes: int = 60 * 24  # 1 day


settings = Settings()