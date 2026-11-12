"""Advisory market lead/lag research; never a trading or risk decision.

Arrival-time buckets, not provider timestamps, align streams. Only closed
buckets participate. Correlation is descriptive, not a Granger test or proof
of causation. Positive returns are not execution success or expected profit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, deque
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..market.events import MarketDataEvent, MarketEventKind
from .recording import canonical_json, event_from_dict, iter_records


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def entropy(probabilities):
    values = [_number(p, "probability") for p in probabilities]
    if not values or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    return -sum(p * math.log2(p) for p in values if p)


def information_gain(prior, posterior, cost=1.0):
    cost = _number(cost, "cost")
    if cost == 0:
        raise ValueError("cost must be positive")
    return (_number(prior, "prior entropy") - _number(posterior, "posterior entropy")) / cost


def cross_entropy(prior, posterior):
    entropy(prior)
    entropy(posterior)
    if len(prior) != len(posterior):
        raise ValueError("distributions must have the same length")
    return sum(-p * math.log2(q) if q else math.inf for p, q in zip(prior, posterior) if p)


def kl_divergence(prior, posterior):
    return cross_entropy(prior, posterior) - entropy(prior)


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta, evaluated on its stable branch."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 / (1.0 - qab * x / qap)
    value = d
    for m in range(1, 201):
        for numerator, denominator in (
            (m * (b - m) * x, (qam + 2 * m) * (a + 2 * m)),
            (-(a + m) * (qab + m) * x, (a + 2 * m) * (qap + 2 * m)),
        ):
            coefficient = numerator / denominator
            d = 1.0 + coefficient * d
            c = 1.0 + coefficient / c
            if abs(d) < 1e-300:
                d = math.copysign(1e-300, d)
            if abs(c) < 1e-300:
                c = math.copysign(1e-300, c)
            d = 1.0 / d
            delta = d * c
            value *= delta
        if abs(delta - 1.0) < 3e-14:
            return value
    raise ArithmeticError("incomplete beta fraction did not converge")


def _pearson_p_value(correlation: float, degrees: float) -> float:
    """Student/Pearson two-sided tail; effective serial degrees are approximate."""
    x = max(0.0, 1.0 - correlation * correlation)
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    a, b = degrees / 2.0, 0.5
    scale = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    result = (
        scale * _beta_fraction(a, b, x) / a
        if x < (a + 1) / (a + b + 2)
        else 1.0 - scale * _beta_fraction(b, a, 1.0 - x) / b
    )
    return max(0.0, min(1.0, result))


def _by_rejections(pvalues: list[float], q: float = 0.01) -> set[int]:
    """Exact Benjamini-Yekutieli step-up over the complete prespecified family."""
    if not pvalues:
        return set()
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("discovery p-values must be finite fractions")
    count = len(pvalues)
    harmonic = math.fsum(1 / rank for rank in range(1, count + 1))
    ordered = sorted(range(count), key=pvalues.__getitem__)
    last = max(
        (
            rank
            for rank, index in enumerate(ordered, 1)
            if pvalues[index] <= q * rank / (count * harmonic)
        ),
        default=0,
    )
    return set(ordered[:last])


def _pair_statistics(
    returns: dict[tuple[str, str], dict[int, float]], forgetting: float
) -> dict[str, NDArray[np.float64] | NDArray[np.int64] | NDArray[np.bool_]]:
    """Masked one-lag sufficient statistics; absent buckets never become returns."""
    keys = sorted(returns)
    buckets = [bucket for values in returns.values() for bucket in values]
    start, end = (min(buckets), max(buckets)) if buckets else (0, 0)
    values = np.zeros((end - start + 1, len(keys)))
    mask = np.zeros_like(values)
    for column, key in enumerate(keys):
        # Shift each stream before raw moments to avoid cancellation from a
        # nonzero level. Exact constant inputs then remain exactly zero.
        baseline = next(iter(returns[key].values()), 0.0)
        for bucket, value in returns[key].items():
            values[bucket - start, column] = value - baseline
            mask[bucket - start, column] = 1
    x, y, mx, my = values[:-1], values[1:], mask[:-1], mask[1:]
    samples = mx.T @ my
    count = np.maximum(samples, 1)
    sx, sy = x.T @ my, mx.T @ y
    sxx, syy = (x * x).T @ my, mx.T @ (y * y)
    covariance = x.T @ y - sx * sy / count
    variance_x = np.maximum(0, sxx - sx * sx / count)
    variance_y = np.maximum(0, syy - sy * sy / count)
    denominator = np.sqrt(variance_x * variance_y)
    correlation = np.divide(
        covariance, denominator, out=np.zeros_like(samples), where=denominator > 0
    )
    weights = forgetting ** np.arange(len(x) - 1, -1, -1)
    total_weight = mx.T @ (weights[:, None] * my)
    safe_weight = np.maximum(total_weight, np.finfo(float).tiny)
    weight_squared = mx.T @ ((weights * weights)[:, None] * my)
    effective = np.divide(
        total_weight * total_weight,
        weight_squared,
        out=np.zeros_like(samples),
        where=weight_squared > 0,
    )
    mean_x = x.T @ (weights[:, None] * my) / safe_weight
    mean_y = mx.T @ (weights[:, None] * y) / safe_weight
    weighted_covariance = x.T @ (weights[:, None] * y) / safe_weight - mean_x * mean_y
    weighted_variance_x = np.maximum(
        0, (x * x).T @ (weights[:, None] * my) / safe_weight - mean_x * mean_x
    )
    weighted_variance_y = np.maximum(
        0, mx.T @ (weights[:, None] * (y * y)) / safe_weight - mean_y * mean_y
    )
    weighted_denominator = np.sqrt(weighted_variance_x * weighted_variance_y)
    weighted_correlation = np.divide(
        weighted_covariance,
        weighted_denominator,
        out=np.zeros_like(samples),
        where=weighted_denominator > 0,
    )
    # Autocorrelations use only pairs present at both bucket endpoints. Gaps
    # remain physical time gaps rather than compressed adjacent observations.
    serial_x = np.maximum(0, sxx - 2 * mean_x * sx + mean_x * mean_x * samples)
    serial_y = np.maximum(0, syy - 2 * mean_y * sy + mean_y * mean_y * samples)
    correction = np.zeros_like(samples)
    for lag in range(1, min(10, len(x) - 1) + 1):
        valid_x, valid_y = mx[:-lag] * mx[lag:], my[:-lag] * my[lag:]
        paired = valid_x.T @ valid_y
        cross_x = (x[:-lag] * x[lag:]).T @ valid_y
        cross_y = valid_x.T @ (y[:-lag] * y[lag:])
        sum_x = ((x[:-lag] + x[lag:]) * valid_x).T @ valid_y
        sum_y = valid_x.T @ ((y[:-lag] + y[lag:]) * valid_y)
        numerator_x = cross_x - mean_x * sum_x + mean_x * mean_x * paired
        numerator_y = cross_y - mean_y * sum_y + mean_y * mean_y * paired
        rho_x = np.divide(numerator_x, serial_x, out=np.zeros_like(samples), where=serial_x > 0)
        rho_y = np.divide(numerator_y, serial_y, out=np.zeros_like(samples), where=serial_y > 0)
        correction += np.maximum(0, 1 - lag / count) * rho_x * rho_y
    effective /= np.maximum(1, 1 + 2 * correction)
    last_bucket = np.full(samples.shape, -1, dtype=np.int64)
    for offset in range(len(x)):
        np.copyto(
            last_bucket,
            start + offset + 1,
            where=(mx[offset, :, None] * my[offset, None, :]).astype(bool),
        )
    return {
        "samples": samples,
        "correlation": np.clip(correlation, -1, 1),
        "weighted_correlation": np.clip(weighted_correlation, -1, 1),
        "effective_samples": effective,
        "last_bucket": last_bucket,
        "valid": (denominator > 0) & (weighted_denominator > 0),
    }


class CausalVenueScout:
    """Bounded single-writer research state keyed by (provider, instrument).

    All times are processing epoch seconds. No currency amounts are combined.
    Stored signals are simple price changes in basis points. Cross-asset edges
    are descriptive comparisons only; they do not assert fungible instruments.
    """

    def __init__(
        self,
        *,
        bucket_seconds=1.0,
        max_buckets=300,
        min_samples=10,
        max_streams=64,
        forgetting=0.95,
        price_policy="first",
        freshness_seconds=5.0,
    ):
        self.bucket_seconds = _number(bucket_seconds, "bucket_seconds")
        if self.bucket_seconds == 0:
            raise ValueError("bucket_seconds must be positive")
        for name, value, low in (
            ("max_buckets", max_buckets, 4),
            ("min_samples", min_samples, 3),
            ("max_streams", max_streams, 1),
        ):
            if type(value) is not int or value < low:
                raise ValueError(f"{name} must be an integer >= {low}")
        if min_samples > max_buckets - 2:
            raise ValueError("min_samples exceeds available lagged history")
        if max_buckets > 10000 or max_streams > 64:
            raise ValueError("research limits are max_buckets <= 10000 and max_streams <= 64")
        if price_policy not in {"first", "trade", "midpoint"}:
            raise ValueError("price_policy must be first, trade or midpoint")
        self.price_policy = price_policy
        self.freshness_seconds = _number(freshness_seconds, "freshness_seconds")
        if not self.freshness_seconds:
            raise ValueError("freshness_seconds must be positive")
        self.max_buckets = max_buckets
        self.min_samples = min_samples
        self.max_streams = max_streams
        self.forgetting = _number(forgetting, "forgetting")
        if not 0 < self.forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        self.reset()

    def reset(self):
        self._prices = {}
        self._clock = None
        self.skipped_capacity = 0
        self._kinds = {}
        self._sequences = {}
        self._generations = {}
        self._input_times = {}
        self.diagnostics = Counter()

    def invalidate_provider(self, provider_id):
        for key in list(self._prices):
            if key[0] == provider_id:
                self._prices.pop(key)
                self._kinds.pop(key, None)
                self._sequences.pop(key, None)
                self._input_times.pop(key, None)
        self.diagnostics["provider_invalidations"] += 1

    def provider_status(self, provider_id, generation, state):
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        previous = self._generations.get(provider_id)
        if previous is not None and generation < previous:
            self.diagnostics["old_generation"] += 1
            return False
        if previous != generation or state in {
            "invalidated",
            "retrying",
            "stopped",
            "access_denied",
            "configuration_error",
        }:
            self.invalidate_provider(provider_id)
        if provider_id not in self._generations and len(self._generations) >= self.max_streams:
            self.diagnostics["provider_capacity"] += 1
            self.skipped_capacity += 1
            return False
        self._generations[provider_id] = generation
        return True

    def observe(
        self, event: MarketDataEvent, processing_ts: float, *, provider_id=None, generation=0
    ):
        stamp = _number(processing_ts, "processing_ts")
        if self._clock is not None and stamp < self._clock:
            raise ValueError("processing clock moved backwards")
        self._clock = stamp
        provider_id = event.source if provider_id is None else provider_id
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be non-empty text")
        if not self.provider_status(provider_id, generation, "healthy"):
            return
        key = (provider_id, event.instrument_id)
        if event.kind in (MarketEventKind.STATUS, MarketEventKind.SESSION):
            if event.kind == MarketEventKind.STATUS and event.status in {
                "trading", "tradeable", "connected", "healthy"
            }:
                return
            if event.kind == MarketEventKind.SESSION and event.session == "open":
                return
            # A discontinuity must not become a synthetic cross-gap return.
            self._prices.pop(key, None)
            self._kinds.pop(key, None)
            self._sequences.pop(key, None)
            self._input_times.pop(key, None)
            self.diagnostics["stream_discontinuity"] += 1
            return
        if event.kind == MarketEventKind.TRADE:
            price = event.price
            kind = "trade"
        elif event.kind == MarketEventKind.BOOK_SNAPSHOT and event.bids and event.asks:
            bid, ask = max(p for p, _ in event.bids), min(p for p, _ in event.asks)
            if bid > ask:
                self.diagnostics["crossed_book"] += 1
                return
            price = (bid + ask) / 2
            kind = "midpoint"
        else:
            self.diagnostics["unsupported_event"] += 1
            return  # Deltas alone cannot reconstruct a best bid/ask.
        if price is None or price <= 0:
            self.diagnostics["nonpositive_price"] += 1
            return  # Signed/nonpositive futures prices need a different return model.
        expected = self._kinds.get(key) if self.price_policy == "first" else self.price_policy
        if expected is not None and expected != kind:
            self.diagnostics["price_policy"] += 1
            return
        if event.sequence is not None and event.sequence <= self._sequences.get(key, -1):
            self.diagnostics["duplicate_or_reordered_sequence"] += 1
            return
        if key not in self._prices and len(self._prices) >= self.max_streams:
            self.skipped_capacity += 1
            return
        bucket = math.floor(stamp / self.bucket_seconds)
        history = self._prices.setdefault(key, deque(maxlen=self.max_buckets))
        self._input_times[key] = stamp
        self._kinds[key] = kind
        if event.sequence is not None:
            self._sequences[key] = event.sequence
        if history and history[-1][0] == bucket:
            history[-1] = (bucket, price)
        else:
            history.append((bucket, price))

    def _returns(self, history, now):
        closed = [
            (b, p)
            for b, p in history
            if (b + 1) * self.bucket_seconds <= now
            and b * self.bucket_seconds >= now - self.max_buckets * self.bucket_seconds
        ]
        return {
            b: (price / previous - 1) * 10000
            for (a, previous), (b, price) in zip(closed, closed[1:])
            if b == a + 1
        }

    def summary(self, now=None):
        now = self._clock if now is None else _number(now, "now")
        if now is None:
            now = 0.0
        if self._clock is not None and now < self._clock:
            raise ValueError("summary cannot precede the latest observation")
        closed_returns = {key: self._returns(history, now) for key, history in self._prices.items()}
        closed_ages = {
            key: now - (max(values) + 1) * self.bucket_seconds if values else None
            for key, values in closed_returns.items()
        }
        returns = {
            key: values
            if closed_ages[key] is not None and closed_ages[key] <= self.freshness_seconds
            else {}
            for key, values in closed_returns.items()
        }
        edges = []
        pvalues = []
        influence = {key: [] for key in returns}
        pair_statistics = _pair_statistics(returns, self.forgetting)
        for source_index, source in enumerate(sorted(returns)):
            for target_index, target in enumerate(sorted(returns)):
                if source == target:
                    continue
                family_index = len(pvalues)
                pvalues.append(1.0)
                pair = source_index, target_index
                samples = int(pair_statistics["samples"][pair])
                if samples < self.min_samples or not pair_statistics["valid"][pair]:
                    continue
                correlation = float(pair_statistics["correlation"][pair])
                paired_age = (
                    now - (int(pair_statistics["last_bucket"][pair]) + 1) * self.bucket_seconds
                )
                paired_stale = paired_age > self.freshness_seconds
                effective = float(pair_statistics["effective_samples"][pair])
                weighted_correlation = float(pair_statistics["weighted_correlation"][pair])
                pvalue = (
                    _pearson_p_value(weighted_correlation, effective - 2)
                    if effective >= 32
                    else None
                )
                eligible = (
                    not paired_stale and samples >= 32 and effective >= 32 and pvalue is not None
                )
                if eligible:
                    pvalues[family_index] = pvalue
                edges.append(
                    {
                        "source": list(source),
                        "target": list(target),
                        "lag_buckets": 1,
                        "lag_seconds": self.bucket_seconds,
                        "samples": samples,
                        "correlation": correlation,
                        "paired_age_seconds": paired_age,
                        "paired_stale": paired_stale,
                        "effective_samples": effective,
                        "weighted_correlation": weighted_correlation,
                        "approximate_p_value": pvalue,
                        "discovery_eligible": eligible,
                        "discovery": False,
                        "actionable": False,
                        "family_index": family_index,
                    }
                )
                if not paired_stale:
                    influence[source].append(abs(correlation))
        rejected = _by_rejections(pvalues)
        for edge in edges:
            edge["discovery"] = edge.pop("family_index") in rejected and edge["discovery_eligible"]
        rows = []
        for key in sorted(returns):
            values = list(returns[key].values())
            n = len(values)
            counts = Counter(
                0 if x < -1 else 1 if x < 0 else 2 if x == 0 else 3 if x <= 1 else 4 for x in values
            )
            score = entropy([c / n for c in counts.values()]) / math.log2(5) if n else None
            alpha = beta = 1.0
            for value in values:
                alpha = 1 + (alpha - 1) * self.forgetting + (value > 0)
                beta = 1 + (beta - 1) * self.forgetting + (value <= 0)
            variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
            volatility = statistics.pstdev(values[-10:]) if n >= self.min_samples else None
            regime = (
                "INSUFFICIENT_DATA"
                if volatility is None
                else "FAST"
                if volatility > 10
                else "SLOW"
                if volatility < 1
                else "TRANSITION"
            )
            half_life = 30 if regime == "FAST" else 300 if regime == "SLOW" else 165
            last_bucket = max(returns[key], default=None)
            age = now - (last_bucket + 1) * self.bucket_seconds if last_bucket is not None else None
            decay = math.exp(-math.log(2) * age / half_life) if age is not None else 0.0
            lead = statistics.mean(influence[key]) if influence[key] else None
            # Direction-neutral exploration ranking, NOT an expected-profit score.
            rank = (
                decay * (0.4 * score + 0.3 * variance * 12 + 0.3 * (lead or 0))
                if n >= self.min_samples
                else None
            )
            rows.append(
                {
                    "provider": key[0],
                    "instrument_id": key[1],
                    "samples": n,
                    "generation": self._generations[key[0]],
                    "price_kind": self._kinds[key],
                    "age_seconds": closed_ages[key],
                    "closed_return_age_seconds": closed_ages[key],
                    "input_age_seconds": now - self._input_times[key]
                    if self._input_times.get(key) is not None
                    else None,
                    "stale": closed_ages[key] is None or closed_ages[key] > self.freshness_seconds,
                    "entropy": score,
                    "positive_return_probability": alpha / (alpha + beta),
                    "uncertainty": variance,
                    "regime": regime,
                    "decay_weight": decay,
                    "lead_strength": lead,
                    "research_priority": rank,
                }
            )
        rows.sort(
            key=lambda row: (
                -(row["research_priority"] or 0),
                row["provider"],
                row["instrument_id"],
            )
        )
        return {
            "schema_version": 2,
            "advisory_only": True,
            "relationship_method": "closed_bucket_lagged_pearson_not_granger",
            "processing_ts": now,
            "skipped_capacity": self.skipped_capacity,
            "diagnostics": dict(self.diagnostics),
            "streams": rows,
            "edges": edges,
            "discoveries": [edge for edge in edges if edge["discovery"]],
            "discovery_method": {
                "family": "all_retained_directed_stream_pairs_one_bucket_lag",
                "family_tests": len(pvalues),
                "q": 0.01,
                "multiplicity": "exact_BY_step_up",
                "minimum_effective_samples": 32,
                "null": "approximate_weighted_Pearson_Kish_Bartlett_10_bucket_lags",
                "calibrated": False,
                "fdr_guaranteed": False,
                "interpretation": "exploratory_discoveries_not_causal_or_actionable",
            },
        }

    def save(self, path):
        """Exclusive research-state creation; never overwrite an account file."""
        state = {
            "schema_version": 3,
            "parameters": {
                "bucket_seconds": self.bucket_seconds,
                "max_buckets": self.max_buckets,
                "min_samples": self.min_samples,
                "max_streams": self.max_streams,
                "forgetting": self.forgetting,
                "price_policy": self.price_policy,
                "freshness_seconds": self.freshness_seconds,
            },
            "clock": self._clock,
            "generations": self._generations,
            "diagnostics": dict(self.diagnostics),
            "kinds": [[list(key), kind] for key, kind in sorted(self._kinds.items())],
            "sequences": [[list(key), value] for key, value in sorted(self._sequences.items())],
            "skipped_capacity": self.skipped_capacity,
            "prices": [[list(key), list(history)] for key, history in sorted(self._prices.items())],
            "input_times": [
                [list(key), self._input_times.get(key)] for key in sorted(self._prices)
            ],
        }
        payload = canonical_json(state)
        envelope = {"state": state, "sha256": hashlib.sha256(payload.encode()).hexdigest()}
        path = Path(path)
        # A hard link publishes only the complete, flushed file, with no overwrite.
        fd, temporary = tempfile.mkstemp(prefix=".research-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(canonical_json(envelope) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            Path(temporary).unlink()

    @classmethod
    def load(cls, path):
        try:
            return cls._load(path)
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError("invalid research state structure") from exc

    @classmethod
    def _load(cls, path):
        path = Path(path)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("research state exceeds size limit")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        state = envelope["state"]
        if (
            state["schema_version"] not in (2, 3)
            or hashlib.sha256(canonical_json(state).encode()).hexdigest() != envelope["sha256"]
        ):
            raise ValueError("research state checksum/schema mismatch")
        scout = cls(**state["parameters"])
        if state["clock"] is not None:
            scout._clock = _number(state["clock"], "clock")
        if type(state["skipped_capacity"]) is not int or state["skipped_capacity"] < 0:
            raise ValueError("invalid capacity count")
        scout.skipped_capacity = state["skipped_capacity"]
        if len(state["prices"]) > scout.max_streams:
            raise ValueError("too many streams")
        for key, history in state["prices"]:
            if len(key) != 2 or any(not isinstance(v, str) or not v.strip() for v in key):
                raise ValueError("invalid stream key")
            key = tuple(key)
            if key in scout._prices or not history or len(history) > scout.max_buckets:
                raise ValueError("duplicate stream or excessive history")
            previous = -1
            for bucket, price in history:
                if (
                    type(bucket) is not int
                    or bucket <= previous
                    or scout._clock is None
                    or bucket > math.floor(scout._clock / scout.bucket_seconds)
                ):
                    raise ValueError("invalid bucket ordering")
                if _number(price, "price") == 0:
                    raise ValueError("price must be positive")
                previous = bucket
            scout._prices[key] = deque((tuple(item) for item in history), maxlen=scout.max_buckets)
        if state["schema_version"] == 2:
            # Old buckets cannot identify the exact last arrival time.
            scout._input_times = dict.fromkeys(scout._prices)
        else:
            for key, value in state["input_times"]:
                key = tuple(key)
                if key not in scout._prices or key in scout._input_times:
                    raise ValueError("invalid input-time stream")
                if value is not None:
                    value = _number(value, "input time")
                    bucket = scout._prices[key][-1][0]
                    if value > scout._clock or math.floor(value / scout.bucket_seconds) != bucket:
                        raise ValueError("input time does not match latest price bucket")
                scout._input_times[key] = value
            if scout._input_times.keys() != scout._prices.keys():
                raise ValueError("missing input time")
        generations = state["generations"]
        if not isinstance(generations, dict) or len(generations) > scout.max_streams:
            raise ValueError("invalid provider generations")
        for provider, value in generations.items():
            if (
                not isinstance(provider, str)
                or not provider.strip()
                or type(value) is not int
                or value < 0
            ):
                raise ValueError("invalid provider generation")
        scout._generations = generations
        for key, kind in state["kinds"]:
            key = tuple(key)
            if (
                key not in scout._prices
                or key in scout._kinds
                or kind not in {"trade", "midpoint"}
                or key[0] not in generations
                or scout.price_policy not in {"first", kind}
            ):
                raise ValueError("invalid price kind")
            scout._kinds[key] = kind
        if scout._kinds.keys() != scout._prices.keys():
            raise ValueError("missing price kind")
        for key, value in state["sequences"]:
            key = tuple(key)
            if (
                key not in scout._prices
                or key in scout._sequences
                or type(value) is not int
                or value < 0
            ):
                raise ValueError("invalid sequence")
            scout._sequences[key] = value
        if not isinstance(state["diagnostics"], dict) or len(state["diagnostics"]) > 32:
            raise ValueError("invalid diagnostics")
        for name, value in state["diagnostics"].items():
            if not isinstance(name, str) or type(value) is not int or value < 0:
                raise ValueError("invalid diagnostic counter")
        scout.diagnostics = Counter(state["diagnostics"])
        return scout

    def parameters(self):
        return {
            "schema_version": 2,
            "bucket_seconds": self.bucket_seconds,
            "max_buckets": self.max_buckets,
            "min_samples": self.min_samples,
            "max_streams": self.max_streams,
            "forgetting": self.forgetting,
            "price_policy": self.price_policy,
            "freshness_seconds": self.freshness_seconds,
        }

    @classmethod
    def from_parameters(cls, parameters):
        if not isinstance(parameters, dict) or parameters.get("schema_version") != 2:
            raise ValueError("research configuration requires schema_version 2")
        expected = cls().parameters().keys()
        if parameters.keys() != expected:
            raise ValueError("research configuration must specify exactly the versioned parameters")
        return cls(**{key: value for key, value in parameters.items() if key != "schema_version"})


def analyze_recording(directory, *, state_out=None, state_in=None, parameters=None):
    """Read validated recordings without instantiating an account or provider."""
    if state_in is not None and parameters is not None:
        raise ValueError(
            "restored state owns its parameters; do not combine state-in and research-config"
        )
    scout = (
        CausalVenueScout.load(state_in)
        if state_in is not None
        else (
            CausalVenueScout.from_parameters(parameters)
            if parameters is not None
            else CausalVenueScout()
        )
    )
    last = scout._clock
    for record in iter_records(directory):
        if last is not None and record["processing_ts"] < last:
            raise ValueError("processing clock moved backwards across recording continuation")
        last = record["processing_ts"]
        if record["kind"] == "market":
            scout.observe(
                event_from_dict(record["payload"]),
                last,
                provider_id=record.get("provider_id"),
                generation=record.get("generation", 0),
            )
        elif record["kind"] == "provider":
            scout.provider_status(
                record["provider_id"], record.get("generation", 0), record["payload"].get("state")
            )
    scout._clock = last
    result = scout.summary(last)
    if state_out is not None:
        scout.save(state_out)
    return result


def evaluate_direction_recording(directory):
    """Prequential direction calibration, not a simulated strategy or profit test."""
    bounds = [record["processing_ts"] for record in iter_records(directory)]
    if len(bounds) < 5 or bounds[-1] <= bounds[0]:
        raise ValueError("direction evaluation needs a chronological recording")
    start, end = bounds[0], bounds[-1]
    train_end = start + (end - start) * 0.6
    validation_end = start + (end - start) * 0.8
    scout = CausalVenueScout()
    pending = {}
    metrics = {
        name: {"samples": 0, "brier_sum": 0.0, "baseline_brier_sum": 0.0}
        for name in ("train", "validation", "test")
    }
    for record in iter_records(directory):
        now = record["processing_ts"]
        if record["kind"] == "provider":
            provider = record["provider_id"]
            before = dict(scout._generations)
            scout.provider_status(
                provider, record.get("generation", 0), record["payload"].get("state")
            )
            if before.get(provider) != scout._generations.get(provider) or record["payload"].get(
                "state"
            ) not in {"healthy", "connected"}:
                pending = {key: value for key, value in pending.items() if key[0] != provider}
        elif record["kind"] == "market":
            event = event_from_dict(record["payload"])
            scout.observe(
                event,
                now,
                provider_id=record.get("provider_id"),
                generation=record.get("generation", 0),
            )
            if event.kind in {MarketEventKind.STATUS, MarketEventKind.SESSION}:
                pending.pop((record.get("provider_id") or event.source, event.instrument_id), None)
        # Score only the next fully closed bucket against the previous forecast.
        for row in scout.summary(now)["streams"]:
            key = (row["provider"], row["instrument_id"])
            returns = scout._returns(scout._prices[key], now)
            bucket = max(returns, default=None)
            if bucket is None:
                continue
            previous = pending.get(key)
            if previous and bucket == previous[0]:
                continue
            if previous and bucket == previous[0] + 1:
                bucket_end = (bucket + 1) * scout.bucket_seconds
                bucket_start = bucket * scout.bucket_seconds
                split = (
                    "train"
                    if bucket_end <= train_end
                    else "validation"
                    if bucket_end <= validation_end
                    else "test"
                )
                boundary = (
                    start
                    if split == "train"
                    else train_end
                    if split == "validation"
                    else validation_end
                )
                if bucket_start >= boundary:
                    outcome = float(returns[bucket] > 0)
                    metrics[split]["samples"] += 1
                    metrics[split]["brier_sum"] += (previous[1] - outcome) ** 2
                    metrics[split]["baseline_brier_sum"] += (0.5 - outcome) ** 2
            if row["samples"] >= scout.min_samples and not row["stale"]:
                pending[key] = (bucket, row["positive_return_probability"])
    for value in metrics.values():
        count = value["samples"]
        value["brier"] = value.pop("brier_sum") / count if count else None
        value["baseline_brier"] = value.pop("baseline_brier_sum") / count if count else None
    return {
        "advisory_only": True,
        "method": "prequential_next_closed_bucket_direction",
        "parameters": scout.parameters(),
        "trials": 1,
        "train_end": train_end,
        "validation_end": validation_end,
        "end": end,
        "metrics": metrics,
        "execution_evaluated": False,
        "automatic_promotion": False,
    }
