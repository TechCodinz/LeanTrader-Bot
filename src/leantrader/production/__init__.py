"""The mature fast trading lane, restored from LeanTrader-Bot.

This package holds the historical fast path that produced the profitable
Testnet run: a seconds-scale collective lane (``fast_collective_testnet``), the
multi-position sentinel layer on top of it (``fast_collective_hyper``), the
sub-second velocity qualifier (``velocity_sniper_testnet``), the compounding
governor (``capital_growth``) and the engine registry (``engine_control``).

Deliberately NOT restored: the historical ``__init__`` here executed nineteen
``install_testnet_*`` monkeypatches at import time, from v1.60.7 through
v1.60.30. Those were written after the fast lane was already running, each one
patching the previous one's behaviour at runtime, and every one of them is
Bybit-Testnet-specific by construction. Importing that chain would both
re-import the runner stack it patches and make this package unable to carry
Live execution -- the opposite of what the restoration is for.

The lane's execution seam is an injected executor, so the patches that governed
order submission are not needed here: order authority lives in
``leantrader.execution`` (preflight -> route_order -> BrokerCCXT), which is
mode-neutral across paper, testnet and live.
"""
