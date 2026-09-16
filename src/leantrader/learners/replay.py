
# Restored: these names were used below but never imported, so this
# module raised NameError on import.
from typing import Any
from typing import Dict
from typing import List
from dataclasses import dataclass
from dataclasses import field
@dataclass
class TradeRecord:
    ts: str
    pair: str
    session: str
    regime: str
    policy: str
    features: Dict[str, float]
    outcome: float  # pnl or R multiple
    costs: float
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReplayBuffer:
    limit: int = 10000
    buf: List[TradeRecord] = field(default_factory=list)

    def add(self, rec: TradeRecord):
        self.buf.append(rec)
        if len(self.buf) > self.limit:
            self.buf = self.buf[-self.limit :]

    def query(self, **filters) -> List[TradeRecord]:
        res = self.buf
        for k, v in filters.items():
            res = [r for r in res if getattr(r, k) == v]
        return res
