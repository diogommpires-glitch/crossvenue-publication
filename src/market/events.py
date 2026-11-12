"""Immutable normalized events shared by market-data provider adapters."""

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral
from typing import Optional, Tuple

from ..utils.types import Side


class MarketEventKind(str, Enum):
    """Normalized event payload kinds."""

    BOOK_SNAPSHOT = "book_snapshot"
    BOOK_DELTA = "book_delta"
    TRADE = "trade"
    SESSION = "session"
    STATUS = "status"


BookLevel = Tuple[float, float]


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(name))
    return value.strip()


def _finite_number(value: float, name: str, positive: bool = False, non_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be finite".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{} must be finite".format(name))
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    if positive and result <= 0:
        raise ValueError("{} must be positive".format(name))
    if non_negative and result < 0:
        raise ValueError("{} must be non-negative".format(name))
    return result


def _levels(values: Tuple[BookLevel, ...], name: str, allow_zero_quantity: bool) -> Tuple[BookLevel, ...]:
    try:
        source = tuple(values)
    except TypeError:
        raise ValueError("{} must be an iterable of (price, quantity) levels".format(name))
    normalized = []
    for index, level in enumerate(source):
        if isinstance(level, (str, bytes)):
            raise ValueError("{} level {} must contain price and quantity".format(name, index))
        try:
            if len(level) != 2:
                raise ValueError
            price, quantity = level
        except (TypeError, ValueError):
            raise ValueError("{} level {} must contain price and quantity".format(name, index))
        normalized_price = _finite_number(price, "{} level price".format(name))
        normalized_quantity = _finite_number(
            quantity,
            "{} level quantity".format(name),
            positive=not allow_zero_quantity,
            non_negative=allow_zero_quantity,
        )
        normalized.append((normalized_price, normalized_quantity))
    return tuple(normalized)


@dataclass(frozen=True)
class MarketDataEvent:
    """Normalized immutable market event with event and local receive times.

    Timestamps are finite epoch seconds. They are deliberately not required to
    be equal or ordered because source clocks may differ; adapters must retain
    both values and a caller must apply its clock-skew policy. Book snapshot
    levels are normalized to descending bids and ascending asks. A zero-size
    level is only permitted in a book delta and denotes removal.
    """

    instrument_id: str
    source: str
    kind: MarketEventKind
    event_ts: float
    receive_ts: float
    sequence: Optional[int] = None
    bids: Tuple[BookLevel, ...] = ()
    asks: Tuple[BookLevel, ...] = ()
    price: Optional[float] = None
    quantity: Optional[float] = None
    side: Optional[Side] = None
    session: Optional[str] = None
    status: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _required_text(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "source", _required_text(self.source, "source"))
        if not isinstance(self.kind, MarketEventKind):
            raise ValueError("kind must be a MarketEventKind")

        object.__setattr__(self, "event_ts", _finite_number(self.event_ts, "event_ts", non_negative=True))
        object.__setattr__(self, "receive_ts", _finite_number(self.receive_ts, "receive_ts", non_negative=True))

        if self.sequence is not None:
            if isinstance(self.sequence, bool) or not isinstance(self.sequence, Integral) or self.sequence < 0:
                raise ValueError("sequence must be a non-negative integer or None")
            object.__setattr__(self, "sequence", int(self.sequence))

        allow_zero = self.kind == MarketEventKind.BOOK_DELTA
        bids = _levels(self.bids, "bids", allow_zero)
        asks = _levels(self.asks, "asks", allow_zero)
        if self.kind == MarketEventKind.BOOK_SNAPSHOT:
            bids = tuple(sorted(bids, key=lambda level: level[0], reverse=True))
            asks = tuple(sorted(asks, key=lambda level: level[0]))
            if len({price for price, _ in bids}) != len(bids):
                raise ValueError("book snapshot bids cannot contain duplicate prices")
            if len({price for price, _ in asks}) != len(asks):
                raise ValueError("book snapshot asks cannot contain duplicate prices")
        object.__setattr__(self, "bids", bids)
        object.__setattr__(self, "asks", asks)

        if self.price is not None:
            object.__setattr__(self, "price", _finite_number(self.price, "price"))
        if self.quantity is not None:
            object.__setattr__(self, "quantity", _finite_number(self.quantity, "quantity", positive=True))
        if self.side is not None and not isinstance(self.side, Side):
            raise ValueError("side must be a Side or None")

        if self.kind == MarketEventKind.TRADE:
            if self.price is None or self.quantity is None:
                raise ValueError("trade events require price and positive quantity")
        if self.kind == MarketEventKind.SESSION and self.session is None:
            raise ValueError("session events require a session value")
        if self.kind == MarketEventKind.STATUS and self.status is None:
            raise ValueError("status events require a status value")

        for name in ("session", "status"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _required_text(value, name).lower())


# Semantic alias for call sites that prefer the longer type name.
NormalizedMarketEvent = MarketDataEvent

