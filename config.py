from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api: SecretStr   
    llm_model: str            


class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int
    postgres_docker_port: int

    @property
    def database_url_gen(self) -> str:
        """postgresql://usuario:password@host:puerto/nombre_db"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Constante de código, no de .env: cambiarla obliga a re-embeber todo el corpus.
# La usan embed.py, load.py y retrieve().
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

db_settings = DBSettings()
ai_settings = AISettings()