from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deriv_ws_url: str = "wss://api.derivws.com/trading/v1/options/ws/public"
    deriv_app_id: int | None = None
    deriv_api_token: str | None = None
    firebase_database_url: str | None = None
    market_symbol: str = "frxEURUSD"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
