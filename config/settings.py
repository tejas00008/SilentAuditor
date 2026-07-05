"""Global configuration settings for SilentAuditor."""

import os
from pathlib import Path

PROJECT_NAME: str = "SilentAuditor"
VERSION: str = "0.1.0"

DEFAULT_CONFIG_PATH: Path = Path(__file__).parent / "thresholds.yaml"
DATABASE_PATH: Path = Path.home() / ".silentauditor" / "cache.db"

LOG_LEVEL: str = "INFO"

LLM_MODEL: str = "claude-sonnet-4-20250514"
LLM_MAX_MONTHLY_COST_PER_CUSTOMER: float = 200.0

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
