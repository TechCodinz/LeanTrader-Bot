from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _base(symbol: str) -> str:
    return symbol.upper().split('/', 1)[0]


def _usd(value: Any) -> float:
    if isinstance(value, dict):
        return _finite(value.get('peggedUSD'))
    return _finite(value)


def _change(values: list[float], periods: int) -> float:
    if not values:
        return 0.0
    current = values[-1]
    index = max(0, len(values) - 1 - periods)
    prior = values[index]
    if current <= 0 or prior <= 0:
        return 0.0
    return current / prior - 1.0


class _AnyHttp:
    def __init__(self, timeout: float = 10.0) -> None:
        self.session = requests.Session()
        self.timeout = timeout

    def get(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(url, params=params or {}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def post(self, url: str, *, json: dict[str, Any]) -> Any:
        response = self.session.post(url, json=json, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


class GlassnodeExchangeFlowSensor:
    """Optional labelled exchange/whale flow sensor for BTC/ETH.

    Glassnode requires an API key.  The adapter remains unconfigured without one
    and never loads trading credentials.  Positive `net_to_exchanges` means
    labelled exchange inflows exceed outflows; negative means net withdrawals.
    """

    VERSION = '1.0'
    BASE_URL = 'https://api.glassnode.com/v1/metrics'
    SUPPORTED = {'BTC', 'ETH'}

    def __init__(
        self,
        api_key_file: Path,
        refresh_seconds: int = 900,
        http: _AnyHttp | None = None,
    ) -> None:
        self.api_key_file = api_key_file
        self.refresh_seconds = max(300, int(refresh_seconds))
        self.http = http or _AnyHttp()
        self.cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.calls = self.successes = self.failures = 0
        self.last_error: str | None = None

    def _key(self) -> str:
        env = os.getenv('GLASSNODE_API_KEY', '').strip()
        if env:
            return env
        try:
            return self.api_key_file.read_text(encoding='utf-8').strip()
        except OSError:
            return ''

    @staticmethod
    def _series(payload: Any) -> list[tuple[float, float]]:
        rows = payload if isinstance(payload, list) else []
        parsed: list[tuple[float, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            timestamp = _finite(row.get('t'))
            value = row.get('v')
            if isinstance(value, dict):
                # Some Glassnode metrics expose a nested value object.
                numeric = next((_finite(v, math.nan) for v in value.values() if math.isfinite(_finite(v, math.nan))), math.nan)
            else:
                numeric = _finite(value, math.nan)
            if timestamp > 0 and math.isfinite(numeric):
                parsed.append((timestamp, numeric))
        return sorted(parsed)

    def _metric(
        self, endpoint: str, asset: str, now: float, *, interval: str = '1h', lookback_hours: int = 72
    ) -> list[tuple[float, float]]:
        payload = self.http.get(
            f'{self.BASE_URL}/{endpoint}',
            params={
                'a': asset,
                'i': interval,
                's': int(now - lookback_hours * 3600),
                'u': int(now),
                'api_key': self._key(),
            },
        )
        return self._series(payload)

    def collect(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        asset = _base(symbol)
        now = time.time()
        if asset not in self.SUPPORTED:
            return {
                'status': 'not_applicable', 'sensor': 'exchange_onchain_flows', 'symbol': symbol,
                'values': {}, 'read_only': True, 'execution_authority': False,
            }
        key = self._key()
        if not key:
            return {
                'status': 'unconfigured', 'sensor': 'exchange_onchain_flows', 'symbol': symbol,
                'values': {}, 'read_only': True, 'execution_authority': False,
            }
        cached = self.cache.get(asset)
        if cached and time.monotonic() - cached[0] < self.refresh_seconds:
            return dict(cached[1])
        self.calls += 1
        try:
            inflow = self._metric('transactions/transfers_volume_to_exchanges_sum', asset, now)
            outflow = self._metric('transactions/transfers_volume_from_exchanges_sum', asset, now)
            if not inflow or not outflow:
                raise ValueError('Glassnode exchange flow series is empty')
            latest_in = inflow[-1][1]
            latest_out = outflow[-1][1]
            gross = latest_in + latest_out
            net_to_exchanges = latest_in - latest_out
            pressure = net_to_exchanges / gross if gross > 0 else 0.0
            inflow_values = [row[1] for row in inflow]
            outflow_values = [row[1] for row in outflow]
            latest_ts = max(inflow[-1][0], outflow[-1][0])

            whale_outflow = 0.0
            whale_status = 'unavailable'
            top_1pct_supply_share = 0.0
            top_1pct_supply_share_change = 0.0
            herfindahl_index = 0.0
            herfindahl_change = 0.0
            concentration_status = 'unavailable'
            try:
                whale = self._metric('distribution/exchange_whales_outflow', asset, now)
                if whale:
                    whale_outflow = whale[-1][1]
                    whale_status = 'available'
                    latest_ts = max(latest_ts, whale[-1][0])
            except Exception:
                # Optional enrichment cannot downgrade the core exchange-flow reading.
                pass
            try:
                top1 = self._metric(
                    'distribution/balance_1pct_holders', asset, now, interval='24h', lookback_hours=24 * 35
                )
                herfindahl = self._metric(
                    'distribution/herfindahl', asset, now, interval='24h', lookback_hours=24 * 35
                )
                if top1:
                    top_1pct_supply_share = top1[-1][1]
                    top_1pct_supply_share_change = top1[-1][1] - top1[0][1]
                    latest_ts = max(latest_ts, top1[-1][0])
                if herfindahl:
                    herfindahl_index = herfindahl[-1][1]
                    herfindahl_change = herfindahl[-1][1] - herfindahl[0][1]
                    latest_ts = max(latest_ts, herfindahl[-1][0])
                if top1 or herfindahl:
                    concentration_status = 'available'
            except Exception:
                # Distribution metrics are additional context and may have a
                # different product entitlement from exchange-flow metrics.
                pass

            result = {
                'source': 'Glassnode labelled on-chain metrics',
                'sensor': 'exchange_onchain_flows',
                'symbol': symbol,
                'status': 'available',
                'observed_at': now,
                'source_timestamp': latest_ts,
                'freshness_seconds': max(0.0, now - latest_ts),
                'confidence': 0.90,
                'provenance': 'configured_read_only_api',
                'values': {
                    'exchange_inflow': latest_in,
                    'exchange_outflow': latest_out,
                    'net_to_exchanges': net_to_exchanges,
                    'exchange_flow_pressure': _clip(pressure),
                    'inflow_change_24h_proxy': _change(inflow_values, 24),
                    'outflow_change_24h_proxy': _change(outflow_values, 24),
                    'whale_outflow': whale_outflow,
                    'whale_metric_status': whale_status,
                    'top_1pct_supply_share': top_1pct_supply_share,
                    'top_1pct_supply_share_change_35d': top_1pct_supply_share_change,
                    'herfindahl_index': herfindahl_index,
                    'herfindahl_change_35d': herfindahl_change,
                    'whale_concentration_status': concentration_status,
                },
                'read_only': True,
                'execution_authority': False,
            }
            self.successes += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 - optional sensor isolation
            self.failures += 1
            self.last_error = f'{type(exc).__name__}: {exc}'
            result = {
                'status': 'degraded', 'sensor': 'exchange_onchain_flows', 'symbol': symbol,
                'values': {}, 'error': self.last_error, 'read_only': True, 'execution_authority': False,
            }
        self.cache[asset] = (time.monotonic(), result)
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {
            'version': self.VERSION,
            'configured': bool(self._key()),
            'supported_assets': sorted(self.SUPPORTED),
            'calls': self.calls,
            'successes': self.successes,
            'failures': self.failures,
            'last_error': self.last_error,
            'read_only': True,
            'execution_authority': False,
        }


class DefiLlamaChainLiquiditySensor:
    """Free cross-chain stablecoin + TVL migration sensor.

    This is deliberately called a liquidity-migration *proxy*: changes in chain
    stablecoin supply and TVL are evidence of capital movement, not proof that a
    specific token will move next.
    """

    VERSION = '1.0'
    STABLE_URL = 'https://stablecoins.llama.fi/stablecoincharts'
    TVL_URL = 'https://api.llama.fi/v2/historicalChainTvl'
    DEFAULT_CHAINS = (
        'Ethereum', 'Tron', 'Solana', 'BSC', 'Arbitrum', 'Base',
        'Avalanche', 'Optimism', 'Polygon', 'Hyperliquid L1',
    )

    def __init__(
        self,
        refresh_seconds: int = 1800,
        chains: tuple[str, ...] | None = None,
        http: _AnyHttp | None = None,
    ) -> None:
        self.refresh_seconds = max(300, int(refresh_seconds))
        self.chains = chains or self.DEFAULT_CHAINS
        self.http = http or _AnyHttp()
        self.cache: tuple[float, dict[str, Any]] | None = None
        self.calls = self.successes = self.failures = 0
        self.last_error: str | None = None

    @staticmethod
    def _stable_values(payload: Any) -> list[float]:
        rows = payload if isinstance(payload, list) else (payload.get('data') or payload.get('chart') or []) if isinstance(payload, dict) else []
        result: list[float] = []
        for row in rows:
            if isinstance(row, dict):
                value = _usd(row.get('totalCirculatingUSD') or row.get('totalCirculating'))
                if value > 0:
                    result.append(value)
        return result

    @staticmethod
    def _tvl_values(payload: Any) -> list[float]:
        rows = payload if isinstance(payload, list) else []
        result: list[float] = []
        for row in rows:
            if isinstance(row, dict):
                value = _finite(row.get('tvl'))
                if value > 0:
                    result.append(value)
        return result

    def _chain(self, chain: str) -> tuple[str, dict[str, Any]]:
        stable = self.http.get(f'{self.STABLE_URL}/{chain}', params={})
        tvl = self.http.get(f'{self.TVL_URL}/{chain}', params={})
        svals = self._stable_values(stable)
        tvals = self._tvl_values(tvl)
        s1 = _change(svals, 1)
        s7 = _change(svals, 7)
        t1 = _change(tvals, 1)
        t7 = _change(tvals, 7)
        # Bounded evidence score.  Values around a few percent over a week are
        # already material for chain-level liquidity migration.
        score = _clip(
            0.55 * math.tanh(s7 / 0.03)
            + 0.30 * math.tanh(t7 / 0.08)
            + 0.10 * math.tanh(s1 / 0.01)
            + 0.05 * math.tanh(t1 / 0.025)
        )
        return chain, {
            'stablecoin_usd': svals[-1] if svals else 0.0,
            'stablecoin_change_1d': s1,
            'stablecoin_change_7d': s7,
            'tvl_usd': tvals[-1] if tvals else 0.0,
            'tvl_change_1d': t1,
            'tvl_change_7d': t7,
            'liquidity_migration_score': score,
            'stablecoin_history_points': len(svals),
            'tvl_history_points': len(tvals),
        }

    def collect(self) -> dict[str, Any]:
        if self.cache and time.monotonic() - self.cache[0] < self.refresh_seconds:
            return dict(self.cache[1])
        self.calls += 1
        now = time.time()
        rows: dict[str, dict[str, Any]] = {}
        failures: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(5, len(self.chains)), thread_name_prefix='chain-liquidity') as pool:
            futures = {pool.submit(self._chain, chain): chain for chain in self.chains}
            for future in as_completed(futures):
                chain = futures[future]
                try:
                    name, row = future.result()
                    rows[name] = row
                except Exception as exc:  # noqa: BLE001
                    failures[chain] = f'{type(exc).__name__}: {exc}'
        if rows:
            ranked = sorted(rows.items(), key=lambda item: item[1]['liquidity_migration_score'], reverse=True)
            result = {
                'source': 'DefiLlama free stablecoin and chain TVL APIs',
                'sensor': 'chain_liquidity_flows',
                'symbol': 'GLOBAL',
                'status': 'available' if not failures else 'partial',
                'observed_at': now,
                'source_timestamp': now,
                'freshness_seconds': 0.0,
                'confidence': 0.78 if not failures else 0.65,
                'values': {
                    'chains': rows,
                    'rotation_leaders': [name for name, _ in ranked[:3]],
                    'rotation_laggards': [name for name, _ in ranked[-3:]],
                    'cross_chain_dispersion': (
                        max(row['liquidity_migration_score'] for row in rows.values())
                        - min(row['liquidity_migration_score'] for row in rows.values())
                    ) if len(rows) >= 2 else 0.0,
                    'failures': failures,
                },
                'read_only': True,
                'execution_authority': False,
            }
            self.successes += 1
            self.last_error = None
        else:
            self.failures += 1
            self.last_error = '; '.join(f'{k}={v}' for k, v in failures.items())[:1200] or 'no chain data'
            result = {
                'status': 'degraded', 'sensor': 'chain_liquidity_flows', 'symbol': 'GLOBAL',
                'values': {'failures': failures}, 'error': self.last_error,
                'read_only': True, 'execution_authority': False,
            }
        self.cache = (time.monotonic(), result)
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {
            'version': self.VERSION,
            'chains': list(self.chains),
            'calls': self.calls,
            'successes': self.successes,
            'failures': self.failures,
            'last_error': self.last_error,
            'read_only': True,
            'execution_authority': False,
        }


class DefiLlamaProFlowSensor:
    """Optional institutional ETF + bridge-flow context using DefiLlama Pro."""

    VERSION = '1.0'
    BASE_URL = 'https://pro-api.llama.fi'
    BRIDGE_CHAINS = ('Ethereum', 'Solana', 'Arbitrum', 'Base', 'Avalanche', 'BSC', 'Optimism')

    def __init__(
        self,
        api_key_file: Path,
        refresh_seconds: int = 1800,
        http: _AnyHttp | None = None,
    ) -> None:
        self.api_key_file = api_key_file
        self.refresh_seconds = max(300, int(refresh_seconds))
        self.http = http or _AnyHttp()
        self.cache: tuple[float, dict[str, Any]] | None = None
        self.calls = self.successes = self.failures = 0
        self.last_error: str | None = None

    def _key(self) -> str:
        env = os.getenv('DEFILLAMA_API_KEY', '').strip()
        if env:
            return env
        try:
            return self.api_key_file.read_text(encoding='utf-8').strip()
        except OSError:
            return ''

    def _url(self, path: str) -> str:
        return f'{self.BASE_URL}/{self._key()}{path}'

    @staticmethod
    def _etf_flows(payload: Any) -> dict[str, dict[str, float]]:
        rows = payload if isinstance(payload, list) else []
        grouped: dict[str, list[tuple[str, float]]] = {'bitcoin': [], 'ethereum': []}
        for row in rows:
            if not isinstance(row, dict):
                continue
            asset = str(row.get('gecko_id') or '').lower()
            if asset not in grouped:
                continue
            grouped[asset].append((str(row.get('day') or ''), _finite(row.get('total_flow_usd'))))
        result: dict[str, dict[str, float]] = {}
        for asset, values in grouped.items():
            values.sort(key=lambda item: item[0])
            flows = [v for _, v in values]
            result[asset] = {
                'latest_flow_usd': flows[-1] if flows else 0.0,
                'flow_7d_usd': sum(flows[-7:]),
                'flow_30d_usd': sum(flows[-30:]),
            }
        return result

    def _bridge_chain(self, chain: str) -> tuple[str, dict[str, float]]:
        payload = self.http.get(self._url(f'/bridges/bridgevolume/{chain}'), params={})
        rows = payload if isinstance(payload, list) else []
        rows = [row for row in rows if isinstance(row, dict)]
        rows.sort(key=lambda row: _finite(row.get('date')))
        recent = rows[-7:]
        latest = rows[-1] if rows else {}
        dep = _finite(latest.get('depositUSD'))
        wd = _finite(latest.get('withdrawUSD'))
        net = dep - wd
        return chain, {
            'deposit_usd_1d': dep,
            'withdraw_usd_1d': wd,
            'net_bridge_flow_usd_1d': net,
            'net_bridge_flow_usd_7d': sum(_finite(row.get('depositUSD')) - _finite(row.get('withdrawUSD')) for row in recent),
        }

    def collect(self) -> dict[str, Any]:
        if self.cache and time.monotonic() - self.cache[0] < self.refresh_seconds:
            return dict(self.cache[1])
        if not self._key():
            result = {
                'status': 'unconfigured', 'sensor': 'institutional_bridge_flows', 'values': {},
                'read_only': True, 'execution_authority': False,
            }
            self.cache = (time.monotonic(), result)
            return dict(result)
        self.calls += 1
        now = time.time()
        try:
            etfs = self.http.get(self._url('/etfs/flows'), params={})
            institutional = self._etf_flows(etfs)
            bridges: dict[str, Any] = {}
            bridge_errors: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix='bridge-flow') as pool:
                futures = {pool.submit(self._bridge_chain, chain): chain for chain in self.BRIDGE_CHAINS}
                for future in as_completed(futures):
                    chain = futures[future]
                    try:
                        name, row = future.result()
                        bridges[name] = row
                    except Exception as exc:  # noqa: BLE001
                        bridge_errors[chain] = f'{type(exc).__name__}: {exc}'
            result = {
                'source': 'DefiLlama Pro ETF and bridge APIs',
                'sensor': 'institutional_bridge_flows',
                'symbol': 'GLOBAL',
                'status': 'available' if bridges and not bridge_errors else 'partial',
                'observed_at': now,
                'source_timestamp': now,
                'freshness_seconds': 0.0,
                'confidence': 0.82 if bridges else 0.70,
                'values': {
                    'institutional_etf_flows': institutional,
                    'bridge_chains': bridges,
                    'bridge_errors': bridge_errors,
                },
                'read_only': True,
                'execution_authority': False,
            }
            self.successes += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = f'{type(exc).__name__}: {exc}'
            result = {
                'status': 'degraded', 'sensor': 'institutional_bridge_flows', 'values': {},
                'error': self.last_error, 'read_only': True, 'execution_authority': False,
            }
        self.cache = (time.monotonic(), result)
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {
            'version': self.VERSION,
            'configured': bool(self._key()),
            'calls': self.calls,
            'successes': self.successes,
            'failures': self.failures,
            'last_error': self.last_error,
            'read_only': True,
            'execution_authority': False,
        }


class EvmChainCongestionSensor:
    """Optional generic EVM JSON-RPC fee/congestion sensor.

    No public RPC URL is hard-coded. Operators may provide a read-only endpoint
    via ETHEREUM_RPC_URL or a secret file. JSON-RPC methods are read-only.
    """

    VERSION = '1.0'

    def __init__(
        self, rpc_url_file: Path, refresh_seconds: int = 300, http: _AnyHttp | None = None
    ) -> None:
        self.rpc_url_file = rpc_url_file
        self.refresh_seconds = max(60, int(refresh_seconds))
        self.http = http or _AnyHttp()
        self.cache: tuple[float, dict[str, Any]] | None = None
        self.calls = self.successes = self.failures = 0
        self.last_error: str | None = None

    def _url(self) -> str:
        env = os.getenv('ETHEREUM_RPC_URL', '').strip()
        if env:
            return env
        try:
            return self.rpc_url_file.read_text(encoding='utf-8').strip()
        except OSError:
            return ''

    def _rpc(self, method: str, params: list[Any]) -> Any:
        payload = self.http.post(
            self._url(),
            json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params},
        )
        if not isinstance(payload, dict):
            raise ValueError('Ethereum JSON-RPC response is not an object')
        if payload.get('error'):
            raise RuntimeError(f"Ethereum JSON-RPC {method} error: {payload['error']}")
        return payload.get('result')

    @staticmethod
    def _hex_int(value: Any) -> int:
        if isinstance(value, int):
            return value
        text = str(value or '0')
        return int(text, 16) if text.startswith('0x') else int(float(text))

    def collect(self) -> dict[str, Any]:
        if self.cache and time.monotonic() - self.cache[0] < self.refresh_seconds:
            return dict(self.cache[1])
        if not self._url():
            result = {
                'status': 'unconfigured', 'sensor': 'evm_network_congestion', 'chain': 'Ethereum',
                'values': {}, 'read_only': True, 'execution_authority': False,
            }
            self.cache = (time.monotonic(), result)
            return dict(result)
        self.calls += 1
        now = time.time()
        try:
            gas_price = self._hex_int(self._rpc('eth_gasPrice', []))
            fee_history = self._rpc('eth_feeHistory', ['0x20', 'latest', [25, 50, 75]]) or {}
            if not isinstance(fee_history, dict):
                raise ValueError('eth_feeHistory result is not an object')
            base_fees = [self._hex_int(value) for value in fee_history.get('baseFeePerGas') or []]
            gas_used = [_finite(value) for value in fee_history.get('gasUsedRatio') or []]
            if not base_fees or not gas_used:
                raise ValueError('Ethereum fee history is empty')
            latest_base = base_fees[-1]
            oldest_base = base_fees[0]
            base_change = latest_base / oldest_base - 1.0 if oldest_base > 0 else 0.0
            mean_util = sum(gas_used) / len(gas_used)
            high_util_fraction = sum(1 for value in gas_used if value >= 0.85) / len(gas_used)
            congestion = _clip(
                0.55 * _clip((mean_util - 0.50) / 0.50, 0.0, 1.0)
                + 0.25 * high_util_fraction
                + 0.20 * _clip(math.tanh(max(0.0, base_change) / 0.50), 0.0, 1.0),
                0.0, 1.0,
            )
            result = {
                'source': 'configured Ethereum JSON-RPC',
                'sensor': 'evm_network_congestion',
                'chain': 'Ethereum',
                'status': 'available',
                'observed_at': now,
                'source_timestamp': now,
                'freshness_seconds': 0.0,
                'confidence': 0.86,
                'provenance': 'configured_read_only_rpc',
                'values': {
                    'gas_price_gwei': gas_price / 1e9,
                    'latest_base_fee_gwei': latest_base / 1e9,
                    'base_fee_change_window': base_change,
                    'mean_gas_used_ratio': mean_util,
                    'high_utilization_fraction': high_util_fraction,
                    'network_congestion_score': congestion,
                    'fee_history_blocks': len(gas_used),
                },
                'read_only': True,
                'execution_authority': False,
            }
            self.successes += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = f'{type(exc).__name__}: {exc}'
            result = {
                'status': 'degraded', 'sensor': 'evm_network_congestion', 'chain': 'Ethereum',
                'values': {}, 'error': self.last_error, 'read_only': True, 'execution_authority': False,
            }
        self.cache = (time.monotonic(), result)
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {
            'version': self.VERSION, 'configured': bool(self._url()), 'calls': self.calls,
            'successes': self.successes, 'failures': self.failures, 'last_error': self.last_error,
            'read_only': True, 'execution_authority': False,
        }


class EthereumStablecoinIssuanceSensor:
    """Read-only USDC/USDT mint/burn event sensor over configured Ethereum RPC.

    The sensor scans a bounded recent block window for ERC-20 Transfer events
    whose source or destination is the zero address. It treats those events as
    issuance/redemption evidence only; it does not assume the newly issued
    stablecoins immediately enter risk markets.
    """

    VERSION = '1.0'
    TRANSFER_TOPIC = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
    ZERO_TOPIC = '0x' + '0' * 64
    TOKENS = {
        'USDC': {'address': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48', 'decimals': 6},
        'USDT': {'address': '0xdAC17F958D2ee523a2206206994597C13D831ec7', 'decimals': 6},
    }

    def __init__(
        self, rpc_url_file: Path, refresh_seconds: int = 300, lookback_blocks: int = 600,
        http: _AnyHttp | None = None,
    ) -> None:
        self.rpc_url_file = rpc_url_file
        self.refresh_seconds = max(60, int(refresh_seconds))
        self.lookback_blocks = max(50, min(5000, int(lookback_blocks)))
        self.http = http or _AnyHttp()
        self.cache: tuple[float, dict[str, Any]] | None = None
        self.calls = self.successes = self.failures = 0
        self.last_error: str | None = None

    def _url(self) -> str:
        env = os.getenv('ETHEREUM_RPC_URL', '').strip()
        if env:
            return env
        try:
            return self.rpc_url_file.read_text(encoding='utf-8').strip()
        except OSError:
            return ''

    def _rpc(self, method: str, params: list[Any]) -> Any:
        payload = self.http.post(
            self._url(),
            json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params},
        )
        if not isinstance(payload, dict):
            raise ValueError('Ethereum JSON-RPC response is not an object')
        if payload.get('error'):
            raise RuntimeError(f"Ethereum JSON-RPC {method} error: {payload['error']}")
        return payload.get('result')

    @staticmethod
    def _quantity(value: Any, decimals: int) -> float:
        try:
            raw = int(str(value or '0x0'), 16)
        except (TypeError, ValueError):
            return 0.0
        return raw / (10 ** decimals)

    def _logs(self, address: str, from_block: int, to_block: int, *, mint: bool) -> list[dict[str, Any]]:
        topics: list[Any] = [self.TRANSFER_TOPIC]
        if mint:
            topics.append(self.ZERO_TOPIC)
        else:
            topics.extend([None, self.ZERO_TOPIC])
        result = self._rpc(
            'eth_getLogs',
            [{
                'address': address,
                'fromBlock': hex(from_block),
                'toBlock': hex(to_block),
                'topics': topics,
            }],
        ) or []
        return [row for row in result if isinstance(row, dict)]

    def collect(self) -> dict[str, Any]:
        if self.cache and time.monotonic() - self.cache[0] < self.refresh_seconds:
            return dict(self.cache[1])
        if not self._url():
            result = {
                'status': 'unconfigured', 'sensor': 'ethereum_stablecoin_issuance', 'chain': 'Ethereum',
                'values': {}, 'read_only': True, 'execution_authority': False,
            }
            self.cache = (time.monotonic(), result)
            return dict(result)
        self.calls += 1
        now = time.time()
        try:
            latest_raw = self._rpc('eth_blockNumber', [])
            latest = int(str(latest_raw), 16)
            first = max(0, latest - self.lookback_blocks + 1)
            tokens: dict[str, Any] = {}
            total_minted = total_burned = 0.0
            for symbol, meta in self.TOKENS.items():
                minted_logs = self._logs(meta['address'], first, latest, mint=True)
                burned_logs = self._logs(meta['address'], first, latest, mint=False)
                minted = sum(self._quantity(row.get('data'), meta['decimals']) for row in minted_logs)
                burned = sum(self._quantity(row.get('data'), meta['decimals']) for row in burned_logs)
                total_minted += minted
                total_burned += burned
                tokens[symbol] = {
                    'minted_units': minted,
                    'burned_units': burned,
                    'net_issuance_units': minted - burned,
                    'mint_events': len(minted_logs),
                    'burn_events': len(burned_logs),
                }
            gross = total_minted + total_burned
            net = total_minted - total_burned
            issuance_pressure = _clip(net / gross) if gross > 0 else 0.0
            result = {
                'source': 'configured Ethereum JSON-RPC ERC-20 Transfer logs',
                'sensor': 'ethereum_stablecoin_issuance',
                'chain': 'Ethereum',
                'status': 'available',
                'observed_at': now,
                'source_timestamp': now,
                'freshness_seconds': 0.0,
                'confidence': 0.84,
                'provenance': 'configured_read_only_rpc',
                'values': {
                    'tokens': tokens,
                    'minted_usd_nominal_window': total_minted,
                    'burned_usd_nominal_window': total_burned,
                    'net_issuance_usd_nominal_window': net,
                    'issuance_pressure': issuance_pressure,
                    'from_block': first,
                    'to_block': latest,
                    'lookback_blocks': self.lookback_blocks,
                },
                'read_only': True,
                'execution_authority': False,
            }
            self.successes += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = f'{type(exc).__name__}: {exc}'
            result = {
                'status': 'degraded', 'sensor': 'ethereum_stablecoin_issuance', 'chain': 'Ethereum',
                'values': {}, 'error': self.last_error, 'read_only': True, 'execution_authority': False,
            }
        self.cache = (time.monotonic(), result)
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {
            'version': self.VERSION, 'configured': bool(self._url()), 'tokens': sorted(self.TOKENS),
            'lookback_blocks': self.lookback_blocks, 'calls': self.calls, 'successes': self.successes,
            'failures': self.failures, 'last_error': self.last_error, 'read_only': True,
            'execution_authority': False,
        }


class SolanaNetworkCongestionSensor:
    """Optional Solana activity/priority-fee sensor through configured JSON-RPC."""

    VERSION = '1.0'

    def __init__(
        self, rpc_url_file: Path, refresh_seconds: int = 300, http: _AnyHttp | None = None
    ) -> None:
        self.rpc_url_file = rpc_url_file
        self.refresh_seconds = max(60, int(refresh_seconds))
        self.http = http or _AnyHttp()
        self.cache: tuple[float, dict[str, Any]] | None = None
        self.calls = self.successes = self.failures = 0
        self.last_error: str | None = None

    def _url(self) -> str:
        env = os.getenv('SOLANA_RPC_URL', '').strip()
        if env:
            return env
        try:
            return self.rpc_url_file.read_text(encoding='utf-8').strip()
        except OSError:
            return ''

    def _rpc(self, method: str, params: list[Any]) -> Any:
        payload = self.http.post(
            self._url(),
            json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params},
        )
        if not isinstance(payload, dict):
            raise ValueError('Solana JSON-RPC response is not an object')
        if payload.get('error'):
            raise RuntimeError(f"Solana JSON-RPC {method} error: {payload['error']}")
        return payload.get('result')

    def collect(self) -> dict[str, Any]:
        if self.cache and time.monotonic() - self.cache[0] < self.refresh_seconds:
            return dict(self.cache[1])
        if not self._url():
            result = {
                'status': 'unconfigured', 'sensor': 'solana_network_congestion', 'chain': 'Solana',
                'values': {}, 'read_only': True, 'execution_authority': False,
            }
            self.cache = (time.monotonic(), result)
            return dict(result)
        self.calls += 1
        now = time.time()
        try:
            samples = self._rpc('getRecentPerformanceSamples', [12]) or []
            fees = self._rpc('getRecentPrioritizationFees', []) or []
            samples = [row for row in samples if isinstance(row, dict)]
            fees = [row for row in fees if isinstance(row, dict)]
            if not samples:
                raise ValueError('Solana performance samples are empty')
            recent_secs = sum(max(1.0, _finite(row.get('samplePeriodSecs'), 60.0)) for row in samples)
            non_vote = sum(_finite(row.get('numNonVoteTransactions')) for row in samples)
            slots = sum(_finite(row.get('numSlots')) for row in samples)
            non_vote_tps = non_vote / recent_secs if recent_secs > 0 else 0.0
            slots_per_second = slots / recent_secs if recent_secs > 0 else 0.0
            fee_values = sorted(max(0.0, _finite(row.get('prioritizationFee'))) for row in fees)
            median_fee = fee_values[len(fee_values) // 2] if fee_values else 0.0
            p90_fee = fee_values[min(len(fee_values) - 1, int(0.90 * (len(fee_values) - 1)))] if fee_values else 0.0
            # Priority fee is a network-demand signal, not a universal capacity
            # gauge. Keep the score conservative and pair it with slot cadence.
            fee_pressure = _clip(math.tanh(p90_fee / 10_000.0), 0.0, 1.0)
            slot_stress = _clip((0.40 - slots_per_second) / 0.20, 0.0, 1.0)
            congestion = _clip(0.75 * fee_pressure + 0.25 * slot_stress, 0.0, 1.0)
            result = {
                'source': 'configured Solana JSON-RPC',
                'sensor': 'solana_network_congestion',
                'chain': 'Solana',
                'status': 'available',
                'observed_at': now,
                'source_timestamp': now,
                'freshness_seconds': 0.0,
                'confidence': 0.76,
                'provenance': 'configured_read_only_rpc',
                'values': {
                    'non_vote_tps': non_vote_tps,
                    'slots_per_second': slots_per_second,
                    'median_prioritization_fee_microlamports': median_fee,
                    'p90_prioritization_fee_microlamports': p90_fee,
                    'network_congestion_score': congestion,
                    'performance_samples': len(samples),
                    'priority_fee_samples': len(fees),
                },
                'read_only': True,
                'execution_authority': False,
            }
            self.successes += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = f'{type(exc).__name__}: {exc}'
            result = {
                'status': 'degraded', 'sensor': 'solana_network_congestion', 'chain': 'Solana',
                'values': {}, 'error': self.last_error, 'read_only': True, 'execution_authority': False,
            }
        self.cache = (time.monotonic(), result)
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {
            'version': self.VERSION, 'configured': bool(self._url()), 'calls': self.calls,
            'successes': self.successes, 'failures': self.failures, 'last_error': self.last_error,
            'read_only': True, 'execution_authority': False,
        }


class FlowIntelligenceSynthesizer:
    """Fuses independent flow evidence without granting execution authority."""

    VERSION = '1.1'
    CHAIN_BY_ASSET = {
        'ETH': 'Ethereum', 'SOL': 'Solana', 'BNB': 'BSC', 'AVAX': 'Avalanche',
        'ARB': 'Arbitrum', 'OP': 'Optimism', 'POL': 'Polygon', 'MATIC': 'Polygon',
        'TRX': 'Tron', 'HYPE': 'Hyperliquid L1',
    }

    def synthesize(
        self,
        symbol: str,
        *,
        exchange_onchain: dict[str, Any] | None,
        chain_liquidity: dict[str, Any] | None,
        stablecoins: dict[str, Any] | None,
        pro_flows: dict[str, Any] | None,
        evm_congestion: dict[str, Any] | None = None,
        solana_congestion: dict[str, Any] | None = None,
        stablecoin_issuance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        base = _base(symbol)
        onchain_values = (exchange_onchain or {}).get('values') or {}
        chain_values = (chain_liquidity or {}).get('values') or {}
        stable_values = (stablecoins or {}).get('values') or {}
        pro_values = (pro_flows or {}).get('values') or {}
        evm_values = (evm_congestion or {}).get('values') or {}
        solana_values = (solana_congestion or {}).get('values') or {}
        issuance_values = (stablecoin_issuance or {}).get('values') or {}

        exchange_pressure = _clip(_finite(onchain_values.get('exchange_flow_pressure')))
        global_supply_change = _finite(stable_values.get('recent_supply_change'))
        stable_impulse = _clip(math.tanh(global_supply_change / 0.005))

        chain_name = self.CHAIN_BY_ASSET.get(base)
        chain_row = ((chain_values.get('chains') or {}).get(chain_name) or {}) if chain_name else {}
        chain_rotation = _clip(_finite(chain_row.get('liquidity_migration_score')))

        institution = (pro_values.get('institutional_etf_flows') or {}).get('bitcoin' if base == 'BTC' else 'ethereum' if base == 'ETH' else '') or {}
        flow7 = _finite(institution.get('flow_7d_usd'))
        # $1bn weekly ETF flow is treated as strong evidence, but still bounded.
        institutional_pressure = _clip(math.tanh(flow7 / 1_000_000_000.0)) if institution else 0.0

        bridge = ((pro_values.get('bridge_chains') or {}).get(chain_name) or {}) if chain_name else {}
        bridge7 = _finite(bridge.get('net_bridge_flow_usd_7d'))
        bridge_pressure = _clip(math.tanh(bridge7 / 250_000_000.0)) if bridge else 0.0

        issuance_pressure = _clip(_finite(issuance_values.get('issuance_pressure')))
        concentration_status = str(onchain_values.get('whale_concentration_status') or 'unavailable')
        top1_change = _finite(onchain_values.get('top_1pct_supply_share_change_35d'))
        # Concentration is treated as a risk/context dimension rather than a
        # directional alpha signal. Percent-vs-fraction provider units are both
        # bounded through tanh instead of assuming a fixed scale.
        concentration_change = _clip(math.tanh(top1_change / 0.01)) if concentration_status == 'available' else 0.0

        network_congestion = 0.0
        network_name = chain_name
        if chain_name == 'Ethereum' and (evm_congestion or {}).get('status') == 'available':
            network_congestion = _clip(_finite(evm_values.get('network_congestion_score')), 0.0, 1.0)
        elif chain_name == 'Solana' and (solana_congestion or {}).get('status') == 'available':
            network_congestion = _clip(_finite(solana_values.get('network_congestion_score')), 0.0, 1.0)

        components: list[tuple[float, float]] = []
        if (exchange_onchain or {}).get('status') == 'available':
            # Positive net-to-exchange pressure is supply pressure, so invert it
            # for a directional liquidity score.
            components.append((-exchange_pressure, 0.35))
        if chain_row:
            components.append((chain_rotation, 0.25))
        if stable_values:
            components.append((stable_impulse, 0.17))
        if (stablecoin_issuance or {}).get('status') == 'available':
            # Actual mint/burn logs are useful, but new issuance is not assumed
            # to be immediately deployed into risk assets. Keep the weight low.
            components.append((issuance_pressure, 0.08))
        if institution:
            components.append((institutional_pressure, 0.12))
        if bridge:
            components.append((bridge_pressure, 0.08))
        total_weight = sum(weight for _, weight in components)
        score = sum(value * weight for value, weight in components) / total_weight if total_weight else 0.0

        contradictions: list[str] = []
        if exchange_pressure >= 0.45 and stable_impulse >= 0.45:
            contradictions.append('exchange_inflows_despite_global_liquidity_expansion')
        if exchange_pressure <= -0.45 and stable_impulse <= -0.45:
            contradictions.append('exchange_withdrawals_despite_global_liquidity_contraction')
        if institutional_pressure and exchange_pressure and institutional_pressure * (-exchange_pressure) < -0.20:
            contradictions.append('institutional_exchange_flow_conflict')
        if chain_rotation and bridge_pressure and chain_rotation * bridge_pressure < -0.20:
            contradictions.append('chain_liquidity_bridge_flow_conflict')

        return {
            'source': 'LeanTrader governed flow synthesizer',
            'sensor': 'flow_intelligence',
            'symbol': symbol,
            'status': 'available' if components else 'insufficient_evidence',
            'observed_at': time.time(),
            'confidence': min(0.90, 0.35 + 0.12 * len(components)),
            'values': {
                'flow_score': _clip(score),
                'exchange_flow_pressure': exchange_pressure,
                'stablecoin_liquidity_impulse': stable_impulse,
                'stablecoin_issuance_pressure': issuance_pressure,
                'chain_liquidity_rotation': chain_rotation,
                'institutional_flow_pressure': institutional_pressure,
                'bridge_flow_pressure': bridge_pressure,
                'whale_concentration_change': concentration_change,
                'network_congestion_score': network_congestion,
                'network': network_name,
                'components_available': len(components),
                'chain': chain_name,
                'contradictions': contradictions,
            },
            'read_only': True,
            'execution_authority': False,
            'can_increase_risk': False,
        }
