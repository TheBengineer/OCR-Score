"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Typed application settings.

    All values are sourced from environment variables or a .env file.
    """

    database_url: str = "postgresql+asyncpg://ocrscore:ocrscore@localhost:5432/ocrscore"
    echo_sql: bool = False
    storage_path: str = "./store"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"

    model_config = {"env_prefix": "OCRSCORE_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
