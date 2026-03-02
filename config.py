"""
Configuration management for the Autonomous Risk-Aware Strategy Generator.
Uses Pydantic for validation and environment variable loading.
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

load_dotenv()

class RiskConfig(BaseSettings):
    """Risk management configuration"""
    max_position_size_pct: float = Field(0.05, ge=0, le=1)  # 5% max per position
    max_portfolio_risk_pct: float = Field(0.20, ge=0, le=1)  # 20% max portfolio risk
    stop_loss_pct: float = Field(0.02, ge=0, le=0.1)  # 2% stop loss
    take_profit_pct: float = Field(0.04, ge=0, le=0.2)  # 4% take profit
    var_confidence_level: float = Field(0.95, ge=0.9, le=0.99)  # 95% VaR confidence
    max_drawdown_pct: float = Field(0.15, ge=0, le=0.5)  # 15% max drawdown
    
    class Config:
        env_prefix = "RISK_"

class StrategyConfig(BaseSettings):
    """Strategy generation configuration"""
    rl_algorithm: str = Field("PPO", regex="^(PPO|A2C|DQN|SAC)$")
    learning_rate: float = Field(3e-4, ge=1e-5, le=1e-2)
    gamma: float = Field(0.99, ge=0.9, le=0.999)
    timesteps_per_training: int = Field(10000, ge=1000, le=100000)
    lookback_window: int = Field(50, ge=10, le=200)
    
    class Config:
        env_prefix = "STRATEGY_"

class FirebaseConfig(BaseSettings):
    """Firebase configuration"""
    project_id: str = Field(..., description="Firebase project ID")
    credentials_path: Optional[str] = Field(None, description="Path to service account key")
    database_url: Optional[str] = Field(None, description="Firebase database URL")
    
    class Config:
        env_prefix = "FIREBASE_"
    
    @validator('credentials_path')
    def validate_credentials_path(cls, v):
        if v and not os.path.exists(v):
            raise FileNotFoundError(f"Firebase credentials file not found: {v}")
        return v

class ExchangeConfig(BaseSettings):
    """Exchange configuration"""
    exchange_id: str = Field("binance", regex="^(binance|coinbase|kraken)$")
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox_mode: bool = Field(True, description="Use sandbox/testnet")
    symbols: List[str] = Field(["BTC/USDT", "ETH/USDT"])
    
    class Config:
        env_prefix = "EXCHANGE_"

class AppConfig(BaseSettings):
    """Main application configuration"""
    risk: RiskConfig = RiskConfig()
    strategy: StrategyConfig = StrategyConfig()
    firebase: FirebaseConfig = FirebaseConfig()
    exchange: ExchangeConfig = ExchangeConfig()
    log_level: str = Field("INFO", regex="^(DEBUG|INFO|WARNING|ERROR)$")
    env: str = Field("development", regex="^(development|staging|production)$")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.env == "production":
            self.exchange.sandbox_mode = False

config = AppConfig()