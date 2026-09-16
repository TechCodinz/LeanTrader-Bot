
# Restored: these names were used below but never imported, so this
# module raised NameError on import.
from typing import Dict
import pandas as pd
from ..dsl.compiler import compile_strategy, load_strategy

def load_house():
    spec = load_strategy("src/leantrader/dsl/examples/smc_trend_breakout.yaml")
    return compile_strategy(spec)

def run(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return load_house()(frames)
