from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leantrader.production.market_world_model import MarketWorldModel
from leantrader.production.onchain_flow_sensors import (
    DefiLlamaChainLiquiditySensor,
    FlowIntelligenceSynthesizer,
    GlassnodeExchangeFlowSensor,
)


class FakeGlassnodeHttp:
    def get(self, url, *, params=None):
        base = 1_700_000_000
        if 'transfers_volume_to_exchanges_sum' in url:
            return [{'t': base + i * 3600, 'v': 100.0 + i} for i in range(30)]
        if 'transfers_volume_from_exchanges_sum' in url:
            return [{'t': base + i * 3600, 'v': 80.0 + i * 0.5} for i in range(30)]
        if 'exchange_whales_outflow' in url:
            return [{'t': base + i * 3600, 'v': 15.0 + i * 0.1} for i in range(30)]
        if 'balance_1pct_holders' in url:
            return [{'t': base + i * 86400, 'v': 0.44 + i * 0.0005} for i in range(30)]
        if 'distribution/herfindahl' in url:
            return [{'t': base + i * 86400, 'v': 0.08 + i * 0.0001} for i in range(30)]
        raise AssertionError(url)


class FakeChainHttp:
    def get(self, url, *, params=None):
        if 'stablecoincharts' in url:
            # Strong but plausible one-week stablecoin growth.
            return [
                {'date': str(i), 'totalCirculatingUSD': {'peggedUSD': 1000.0 + i * 8.0}}
                for i in range(12)
            ]
        if 'historicalChainTvl' in url:
            return [{'date': i, 'tvl': 5000.0 + i * 20.0} for i in range(12)]
        raise AssertionError(url)


def _frame(n=320):
    close = np.linspace(100.0, 104.0, n)
    return pd.DataFrame({
        'open': close,
        'high': close * 1.002,
        'low': close * 0.998,
        'close': close,
        'volume': np.full(n, 10.0),
    })


def test_glassnode_exchange_flow_adapter_is_read_only_and_directional(tmp_path: Path):
    key = tmp_path / 'glassnode.key'
    key.write_text('test-key')
    sensor = GlassnodeExchangeFlowSensor(key, http=FakeGlassnodeHttp())
    row = sensor.collect('BTC/USDT')
    assert row['status'] == 'available'
    assert row['values']['net_to_exchanges'] > 0
    assert row['values']['exchange_flow_pressure'] > 0
    assert row['values']['whale_metric_status'] == 'available'
    assert row['values']['whale_concentration_status'] == 'available'
    assert row['values']['top_1pct_supply_share_change_35d'] > 0
    assert row['execution_authority'] is False
    assert sensor.health()['configured'] is True


def test_glassnode_adapter_stays_explicitly_unconfigured_without_key(tmp_path: Path):
    sensor = GlassnodeExchangeFlowSensor(tmp_path / 'missing.key', http=FakeGlassnodeHttp())
    row = sensor.collect('BTC/USDT')
    assert row['status'] == 'unconfigured'
    assert row['execution_authority'] is False


def test_defillama_chain_liquidity_sensor_builds_rotation_proxy():
    sensor = DefiLlamaChainLiquiditySensor(chains=('Ethereum', 'Solana'), http=FakeChainHttp())
    row = sensor.collect()
    assert row['status'] == 'available'
    assert set(row['values']['chains']) == {'Ethereum', 'Solana'}
    assert row['values']['chains']['Ethereum']['liquidity_migration_score'] > 0
    assert row['execution_authority'] is False


def test_flow_synthesizer_fuses_independent_evidence_without_authority():
    synth = FlowIntelligenceSynthesizer()
    row = synth.synthesize(
        'ETH/USDT',
        exchange_onchain={
            'status': 'available',
            'values': {'exchange_flow_pressure': -0.6},
        },
        chain_liquidity={
            'status': 'available',
            'values': {'chains': {'Ethereum': {'liquidity_migration_score': 0.7}}},
        },
        stablecoins={
            'status': 'available',
            'values': {'recent_supply_change': 0.01},
        },
        pro_flows={
            'status': 'available',
            'values': {
                'institutional_etf_flows': {'ethereum': {'flow_7d_usd': 500_000_000.0}},
                'bridge_chains': {'Ethereum': {'net_bridge_flow_usd_7d': 200_000_000.0}},
            },
        },
    )
    assert row['status'] == 'available'
    assert row['values']['flow_score'] > 0
    assert row['values']['components_available'] == 5
    assert row['execution_authority'] is False
    assert row['can_increase_risk'] is False


def test_world_model_promotes_flow_evidence_to_research_patterns_only(tmp_path: Path):
    model = MarketWorldModel(tmp_path / 'world.json')
    result = model.observe_symbol(
        'BTC/USDT',
        _frame(),
        adaptive={'score': 0.2, 'confidence': 0.6, 'regime': 'trend'},
        advanced={
            'signals': [],
            'swarm': {'score': 0.2, 'confidence': 0.6},
            'liquidity': {'imbalance': 0.1, 'spread_bps': 2.0, 'available': True},
        },
        public_context={'available': True, 'score': 0.0, 'confidence': 0.5},
        timeframe_signals={'15m': 0.2, '1h': 0.1},
        timeframe_coverage=1.0,
        external_sensors={
            'flow_intelligence': {
                'status': 'available',
                'confidence': 0.8,
                'values': {
                    'flow_score': -0.7,
                    'exchange_flow_pressure': 0.7,
                    'stablecoin_liquidity_impulse': -0.5,
                    'chain_liquidity_rotation': 0.0,
                    'institutional_flow_pressure': 0.0,
                    'bridge_flow_pressure': 0.0,
                    'contradictions': [],
                },
            },
        },
    )
    assert 'exchange_inflow_supply_pressure' in result['latent_patterns']
    assert 'stablecoin_liquidity_contraction' in result['latent_patterns']
    assert result['specialist_scores']['onchain_flow_intelligence'] == -0.7
    assert result['execution_authority'] is False

class FakeEvmRpcHttp:
    def post(self, url, *, json):
        method = json['method']
        if method == 'eth_gasPrice':
            return {'jsonrpc': '2.0', 'id': 1, 'result': hex(45_000_000_000)}
        if method == 'eth_feeHistory':
            base = [hex(int(20_000_000_000 * (1 + i * 0.04))) for i in range(33)]
            return {
                'jsonrpc': '2.0', 'id': 1,
                'result': {
                    'baseFeePerGas': base,
                    'gasUsedRatio': [0.91 if i % 2 == 0 else 0.82 for i in range(32)],
                    'reward': [],
                },
            }
        raise AssertionError(method)


class FakeSolanaRpcHttp:
    def post(self, url, *, json):
        method = json['method']
        if method == 'getRecentPerformanceSamples':
            return {
                'jsonrpc': '2.0', 'id': 1,
                'result': [
                    {'slot': 1000 - i, 'numTransactions': 70000, 'numNonVoteTransactions': 52000,
                     'numSlots': 20, 'samplePeriodSecs': 60}
                    for i in range(12)
                ],
            }
        if method == 'getRecentPrioritizationFees':
            return {
                'jsonrpc': '2.0', 'id': 1,
                'result': [{'slot': 900 + i, 'prioritizationFee': 12000 + i * 1000} for i in range(20)],
            }
        raise AssertionError(method)


def test_evm_congestion_sensor_is_optional_read_only_and_detects_pressure(tmp_path: Path):
    from leantrader.production.onchain_flow_sensors import EvmChainCongestionSensor

    secret = tmp_path / 'ethereum_rpc_url'
    secret.write_text('https://rpc.example.invalid')
    sensor = EvmChainCongestionSensor(secret, http=FakeEvmRpcHttp())
    row = sensor.collect()
    assert row['status'] == 'available'
    assert row['values']['network_congestion_score'] > 0.5
    assert row['values']['mean_gas_used_ratio'] > 0.8
    assert row['execution_authority'] is False


def test_evm_congestion_sensor_is_explicitly_unconfigured_without_rpc(tmp_path: Path):
    from leantrader.production.onchain_flow_sensors import EvmChainCongestionSensor

    sensor = EvmChainCongestionSensor(tmp_path / 'missing')
    row = sensor.collect()
    assert row['status'] == 'unconfigured'
    assert row['execution_authority'] is False


def test_solana_congestion_sensor_uses_performance_and_priority_fee_evidence(tmp_path: Path):
    from leantrader.production.onchain_flow_sensors import SolanaNetworkCongestionSensor

    secret = tmp_path / 'solana_rpc_url'
    secret.write_text('https://solana.example.invalid')
    sensor = SolanaNetworkCongestionSensor(secret, http=FakeSolanaRpcHttp())
    row = sensor.collect()
    assert row['status'] == 'available'
    assert row['values']['non_vote_tps'] > 0
    assert row['values']['network_congestion_score'] > 0
    assert row['execution_authority'] is False


def test_world_model_marks_chain_congestion_as_research_only_pattern(tmp_path: Path):
    model = MarketWorldModel(tmp_path / 'world-congestion.json')
    result = model.observe_symbol(
        'ETH/USDT',
        _frame(),
        adaptive={'score': 0.1, 'confidence': 0.6, 'regime': 'trend'},
        advanced={
            'signals': [],
            'swarm': {'score': 0.1, 'confidence': 0.6},
            'liquidity': {'imbalance': 0.0, 'spread_bps': 2.0, 'available': True},
        },
        public_context={'available': True, 'score': 0.0, 'confidence': 0.5},
        timeframe_signals={'15m': 0.1, '1h': 0.1},
        timeframe_coverage=1.0,
        external_sensors={
            'evm_network_congestion': {'status': 'available', 'values': {'network_congestion_score': 0.9}},
            'flow_intelligence': {
                'status': 'available', 'confidence': 0.8,
                'values': {
                    'flow_score': 0.0,
                    'exchange_flow_pressure': 0.0,
                    'stablecoin_liquidity_impulse': 0.0,
                    'chain_liquidity_rotation': 0.0,
                    'institutional_flow_pressure': 0.0,
                    'bridge_flow_pressure': 0.0,
                    'whale_concentration_change': 0.0,
                    'network_congestion_score': 0.9,
                    'contradictions': [],
                },
            },
        },
    )
    assert 'chain_congestion_stress' in result['latent_patterns']
    assert result['senses']['flow_stress'] > 0
    assert result['execution_authority'] is False

class FakeStablecoinIssuanceRpcHttp:
    def post(self, url, *, json):
        method = json['method']
        if method == 'eth_blockNumber':
            return {'jsonrpc': '2.0', 'id': 1, 'result': hex(20_000_000)}
        if method == 'eth_getLogs':
            flt = json['params'][0]
            topics = flt['topics']
            is_mint = len(topics) == 2
            # USDC/USDT both use 6 decimals in this bounded sensor.
            amount = 2_000_000 * 10**6 if is_mint else 500_000 * 10**6
            return {
                'jsonrpc': '2.0', 'id': 1,
                'result': [{'data': hex(amount), 'topics': topics, 'blockNumber': hex(19_999_999)}],
            }
        raise AssertionError(method)


def test_ethereum_stablecoin_mint_burn_sensor_tracks_bounded_event_window(tmp_path: Path):
    from leantrader.production.onchain_flow_sensors import EthereumStablecoinIssuanceSensor

    secret = tmp_path / 'ethereum_rpc_url'
    secret.write_text('https://rpc.example.invalid')
    sensor = EthereumStablecoinIssuanceSensor(secret, http=FakeStablecoinIssuanceRpcHttp())
    row = sensor.collect()
    assert row['status'] == 'available'
    assert row['values']['minted_usd_nominal_window'] == 4_000_000
    assert row['values']['burned_usd_nominal_window'] == 1_000_000
    assert row['values']['net_issuance_usd_nominal_window'] == 3_000_000
    assert row['values']['issuance_pressure'] > 0
    assert row['execution_authority'] is False
