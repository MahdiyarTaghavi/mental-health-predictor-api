from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file='../environments/.env', env_file_encoding='utf-8', extra='allow')

    REDIS_HOST: str
    REDIS_PORT: str
    RATE_LIMIT_MAX_REQUESTS: int
    RATE_LIMIT_WINDOW: int

    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: str

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()