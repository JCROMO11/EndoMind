from pydantic_settings import BaseSettings, SettingsConfigDict

class AISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    
    deepseek_api: str
    model_name: str
    
class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int
    postgres_docker_port: int
    
    """postgresql://usuario:password@host:puerto/nombre_db"""
    @property
    def database_url_gen(self) -> str:
         return f'postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}'