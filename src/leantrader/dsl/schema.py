
# Restored: these names were used below but never imported, so this
# module raised NameError on import.
from pydantic import BaseModel

# Restored: these names were used below but never imported, so this
# module raised NameError on import.
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
Timeframe = Literal["D1", "H4", "H1", "M15", "M5", "M1"]

class Condition(BaseModel):
    feature: str
    timeframe: Timeframe
    lookback_ok: int = 0

class SignalRule(BaseModel):
    name: str
    side: Literal["long", "short"]
    entry: List[Condition]
    confirm_any: Optional[List[Condition]] = None
    exit: List[Condition] = []
    stop: Optional[str] = None
    take: Optional[str] = None

class StrategySpec(BaseModel):
    meta: Dict[str, str] = {}
    timeframes: List[Timeframe] = ["D1", "H4", "H1", "M15"]
    signals: List[SignalRule]
