from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file='../environments/.env', env_file_encoding='utf-8', extra='allow')

    REDIS_HOST: str
    REDIS_PORT: str
    RATE_LIMIT_MAX_REQUESTS: str
    RATE_LIMIT_WINDOW: str

    # POSTGRES_DB: str
    # POSTGRES_USER: str
    # POSTGRES_PASSWORD: str
    # POSTGRES_HOST: str
    # POSTGRES_PORT: str

@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()