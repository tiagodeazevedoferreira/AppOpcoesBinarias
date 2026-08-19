from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deriv_ws_url: str = "wss://ws.derivws.com/websockets/v3"
    deriv_app_id: int | None = None
    market_symbol: str = "frxEURUSD"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
