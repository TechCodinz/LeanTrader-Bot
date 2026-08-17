from __future__ import annotations

from pathlib import Path

from leantrader.production.market_sensor_fabric import BybitDerivativesSensor, DefiLlamaStablecoinSensor, MarketSensorFabric


class FakeHttp:
    def get(self, url, *, params):
        if url.endswith('/v5/market/tickers'):
            return {'retCode': 0, 'result': {'list': [{'fundingRate':'0.0006','markPrice':'101','indexPrice':'100','openInterest':'1200','nextFundingTime':'999'}]}}
        if url.endswith('/v5/market/funding/history'):
            return {'retCode':0,'result':{'list':[{'fundingRate':'0.0006'},{'fundingRate':'0.0002'}]}}
        if url.endswith('/v5/market/open-interest'):
            return {'retCode':0,'result':{'list':[{'openInterest':'1200','timestamp':'1000000'},{'openInterest':'1000','timestamp':'900000'}]}}
        if url.endswith('/v5/market/account-ratio'):
            return {'retCode':0,'result':{'list':[{'buyRatio':'0.60','sellRatio':'0.40','timestamp':'1000000'}]}}
        raise AssertionError(url)


def test_bybit_derivatives_sensor_builds_positioning_features():
    sensor = BybitDerivativesSensor(http=FakeHttp())
    row = sensor.collect('BTC/USDT')
    assert row['status'] == 'available'
    assert row['values']['funding_rate'] == 0.0006
    assert abs(row['values']['open_interest_change_15m_window'] - 0.2) < 1e-12
    assert row['values']['positioning_skew'] > 0
    assert row['execution_authority'] is False


def test_fabric_source_status_marks_missing_adapters_explicitly(tmp_path: Path):
    fabric = MarketSensorFabric(tmp_path/'sensors.json', fred_api_key_file=tmp_path/'missing')
    status = fabric.source_status(
        {'BTC/USDT': {
            'derivatives': {'status':'available'},
            'liquidations': {'status':'available'},
            'options': {'status':'available'},
        }},
        {'status':'unconfigured'},
        {'status':'available'},
    )
    assert status['derivatives_funding'] == 'available'
    assert status['open_interest'] == 'available'
    assert status['liquidations'] == 'available'
    assert status['options_surface'] == 'available'
    assert status['onchain_flows'] == 'unconfigured_or_unavailable'
    assert status['stablecoin_liquidity'] == 'available'


class StablecoinListHttp:
    def get_any(self, url, *, params):
        if url.endswith('/stablecoins'):
            return [
                {'symbol':'USDT','circulating':{'peggedUSD':1000},'price':1.0},
                {'symbol':'USDC','circulating':{'peggedUSD':500},'price':0.999},
            ]
        if url.endswith('/stablecoincharts/all'):
            return [
                {'totalCirculatingUSD':{'peggedUSD':1400}},
                {'totalCirculatingUSD':{'peggedUSD':1500}},
            ]
        raise AssertionError(url)

    def get(self, url, *, params):
        value = self.get_any(url, params=params)
        if not isinstance(value, dict):
            raise ValueError('JSON response is not an object')
        return value


def test_defillama_stablecoin_sensor_accepts_list_responses():
    sensor = DefiLlamaStablecoinSensor(http=StablecoinListHttp())
    row = sensor.collect()
    assert row['status'] == 'available'
    assert row['values']['total_circulating_usd'] == 1500
    assert row['values']['recent_supply_change'] > 0
    assert sensor.health()['failures'] == 0
