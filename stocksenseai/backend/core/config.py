import base64
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ENCRYPTION_KEY: str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "stocksense"
    POSTGRES_USER: str = "stocksense"
    POSTGRES_PASSWORD: str = "stocksense"
    DATABASE_URL: str = "postgresql+asyncpg://stocksense:stocksense@localhost:5432/stocksense"

    REDIS_URL: str = "redis://localhost:6379/0"

    SENDGRID_API_KEY: str = ""
    FROM_EMAIL: str = "noreply@stocksense.ai"

    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/google/callback"

    CORS_ORIGINS: str = "http://localhost:5173"
    FRONTEND_URL: str = "http://localhost:5173"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    SYSTEM_KIS_APP_KEY: str = ""
    SYSTEM_KIS_APP_SECRET: str = ""
    SYSTEM_KIS_MODE: str = "paper"
    SYSTEM_KIS_ACCOUNT_NO: str = ""  # 형식: 12345678-01
    EMAIL_VERIFY_TOKEN_EXPIRE_MINUTES: int = 30
    ML_UPLOAD_KEY: str = ""  # 로컬 예측 업로드용 시크릿 키 (.env에서 설정)

    @field_validator("ENCRYPTION_KEY")
    @classmethod
    def validate_encryption_key(cls, v: str) -> str:
        try:
            raw = base64.b64decode(v)
        except Exception:
            raise ValueError(
                "ENCRYPTION_KEY must be valid base64. "
                "Generate with: python -c \"import base64,os; print(base64.b64encode(os.urandom(32)).decode())\""
            )
        if len(raw) != 32:
            raise ValueError(
                f"ENCRYPTION_KEY must decode to exactly 32 bytes for AES-256 (got {len(raw)}). "
                "Generate with: python -c \"import base64,os; print(base64.b64encode(os.urandom(32)).decode())\""
            )
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @model_validator(mode="after")
    def assemble_db_url(self) -> "Settings":
        if not self.DATABASE_URL or self.DATABASE_URL == "postgresql+asyncpg://stocksense:stocksense@localhost:5432/stocksense":
            self.DATABASE_URL = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        if self.APP_ENV == "production" and self.SECRET_KEY == "dev-secret-key-change-in-production":
            raise ValueError(
                "SECRET_KEY must be changed from the default value in production. "
                "Set a strong random SECRET_KEY in your .env file."
            )
        return self

    model_config = {
        "env_file": PROJECT_ROOT / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
