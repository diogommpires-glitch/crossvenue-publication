# Live USDT/USD account valuation

The optional `coinbase` provider reads Coinbase Exchange's public USDT-USD level-1 book once per second. It uses only the public GET endpoint, with no credentials, account access or order methods. See the [official product-book reference](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-book).

Configure one `coinbase` provider bound to a disabled deliverable-FX instrument with base/quantity currency USDT, quote/settlement currency USD, venue COINBASE, contract multiplier 1 and provider symbol `USDT-USD`. The fill model is `fx_quote_cross`, but `enabled: false` is mandatory. Both strategy and manual paper-order admission remain disabled for this valuation entry. The public best bid/ask midpoint values existing cash through the currency-conversion path. The quote entry holds zero exposure and creates no fills or deposits.

Native cash remains in USDT. Displayed USD equity and translation PnL move with the observed midpoint. Historical opening capital, funding, fills, action IDs and command history are retained. For an existing account, changing the configuration requires the usual stopped-account backup, preview and explicit migration. Its historical opening-conversion seed is retained for reconciliation; migration does not renew its timestamp. After a live quote has replaced that seed, an outage cannot restore parity as a fallback.

## Timestamp and freshness convention

The adapter preserves the publisher's RFC3339 book time and integer sequence alongside local receive time. It rejects missing/malformed times or sequences, auction, crossed, non-finite or nonpositive books, future timestamps beyond the configured skew and stale source data. Duplicate/lower sequences and regressed source times do not refresh the same provider instance. Reconnection also remains subject to the engine's source-age checks; no local sequence counter or timestamp substitution is used.

Factory limits are the stricter of the account settings and 10 seconds of source age plus 3 seconds of clock skew. The existing FX ledger measures mark age from local receipt and expires at the account's `max_mark_age_seconds`. With the active account's 30 seconds mark limit, the admission-plus-hold budget is at most approximately 43 seconds of publication age. This is not a guarantee of 30 seconds from publication. A healthy clock is required. Errors or expired marks block affected valuation/risk; the adapter does not widen tolerances or manufacture a 1.0 rate.

Requests have fixed connect/read timeouts of 3 seconds, a 64 KiB response bound and a body deadline of 6 seconds checked between bounded reads. Redirects are rejected. Owned public sessions disable ambient credential lookup. The provider supervisor handles retries and invalidation. Stop prevents new callback admissions and returns after a bounded join; an already-admitted external callback may finish afterwards. An in-flight worker cannot be restarted until it exits. Owned HTTP sessions close when the worker exits; injected test sessions remain caller-owned.

The provider does not fabricate public trades, session state or executable liquidity. The disabled valuation instrument is not a stablecoin redemption guarantee, and a successful quote is not acceptance evidence for other provider credentials or market entitlements.
