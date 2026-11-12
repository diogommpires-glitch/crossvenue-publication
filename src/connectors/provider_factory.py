"""Build configured read-only market-data adapters for local paper accounts."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import List, Tuple

from ..config.cross_asset_config import CrossAssetConfig
from .binance_provider import BinancePublicDataProvider
from .coinbase_provider import CoinbasePublicQuoteProvider
from .databento_provider import DatabentoMarketDataProvider
from .market_data import MarketDataProvider
from .oanda_provider import OandaPricingProvider


ProviderBinding = Tuple[str, MarketDataProvider, Tuple[str, ...]]


def build_market_data_provider_factories(config: CrossAssetConfig, environ=None):
    """Return lazy independent bindings so one missing entitlement is isolated.

    Unlike the eager compatibility helper below, this does not construct an
    SDK or inspect credentials. Each supervisor invokes only its own factory.
    """
    bindings = []
    for provider_id, provider in config.providers.items():
        groups = ((instrument_id,) for instrument_id in provider.instrument_ids) if (
            provider.provider_type == "binance") else (provider.instrument_ids,)
        for instrument_ids in groups:
            isolated_provider = replace(provider, instrument_ids=tuple(instrument_ids))
            isolated_config = replace(config, providers={provider_id: isolated_provider})
            binding_id = "{}:{}".format(provider_id, instrument_ids[0]) if (
                provider.provider_type == "binance") else provider_id

            def factory(selected=isolated_config):
                return build_market_data_providers(selected, environ=environ)[0][1]

            bindings.append((binding_id, factory, tuple(instrument_ids)))
    return bindings


def build_market_data_providers(config: CrossAssetConfig,
                                environ=None) -> List[ProviderBinding]:
    """Construct adapters from config and environment-only credentials.

    The returned provider contract exposes start/stop and normalized events
    only. No order, cancel, account, or execution endpoint is constructed.
    """
    env = os.environ if environ is None else environ
    bindings: List[ProviderBinding] = []
    for provider_id, provider_config in config.providers.items():
        instruments = [config.instruments[instrument_id]
                       for instrument_id in provider_config.instrument_ids]
        if provider_config.provider_type == "binance":
            for item in instruments:
                bindings.append((
                    "{}:{}".format(provider_id, item.spec.instrument_id),
                    BinancePublicDataProvider(item.spec, symbol=item.provider_symbol),
                    (item.spec.instrument_id,),
                ))
        elif provider_config.provider_type == "coinbase":
            item = instruments[0]
            bindings.append((provider_id, CoinbasePublicQuoteProvider(
                item.spec, symbol=item.provider_symbol,
                # This valuation feed must not relax stricter account limits.
                max_source_age_seconds=min(config.market_data_stale_after, 10.0),
                max_clock_skew_seconds=min(config.max_clock_skew_seconds, 3.0),
            ), (item.spec.instrument_id,)))
        elif provider_config.provider_type == "databento":
            symbols = {}
            for item in instruments:
                symbol = item.provider_symbol.strip().upper()
                if symbol in symbols:
                    raise ValueError(
                        "Databento provider {} maps raw symbol {} more than once".format(
                            provider_id, symbol))
                symbols[symbol] = item.spec
            if not provider_config.dataset:
                raise ValueError("Databento provider {} requires an explicit dataset".format(provider_id))
            schemas = provider_config.schemas or ("definition", "mbp-1", "trades", "status")
            provider = DatabentoMarketDataProvider(
                dataset=provider_config.dataset,
                symbol_map=symbols,
                schemas=schemas,
                stype_in=provider_config.stype_in,
            )
            bindings.append((provider_id, provider,
                             tuple(item.spec.instrument_id for item in instruments)))
        elif provider_config.provider_type == "oanda":
            account_id = env.get(provider_config.account_id_env, "")
            if not account_id.strip():
                raise RuntimeError(
                    "OANDA provider {} needs account ID in environment variable {}".format(
                        provider_id, provider_config.account_id_env))
            symbol_map = {}
            for item in instruments:
                symbol = item.provider_symbol.strip().upper()
                if not symbol:
                    raise ValueError("OANDA provider {} has an empty symbol mapping".format(
                        provider_id))
                if symbol in symbol_map:
                    raise ValueError(
                        "OANDA provider {} maps symbol {} more than once".format(
                            provider_id, symbol))
                symbol_map[symbol] = item.spec
            provider = OandaPricingProvider(
                account_id=account_id,
                instrument_map=symbol_map,
                environment=provider_config.environment,
            )
            bindings.append((provider_id, provider,
                             tuple(item.spec.instrument_id for item in instruments)))
        else:
            raise ValueError("unsupported market-data provider type {}".format(
                provider_config.provider_type))
    return bindings
