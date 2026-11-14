"""Thin Kalshi REST client: RSA-PSS request signing, retries, and the handful of
endpoints this bot needs. Public market-data calls work without credentials."""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

log = logging.getLogger(__name__)

PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiError(RuntimeError):
    """Raised for non-retryable API errors."""


class KalshiClient:
    def __init__(self, base_url: str = PROD_URL, key_id: Optional[str] = None,
                 private_key_path: Optional[str] = None, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self.timeout = timeout
        self._session = requests.Session()
        self._private_key = None
        if private_key_path:
            pem = Path(private_key_path).expanduser().read_bytes()
            self._private_key = serialization.load_pem_private_key(pem, password=None)
            if not isinstance(self._private_key, (rsa.RSAPrivateKey, ed25519.Ed25519PrivateKey)):
                raise KalshiError("Unsupported private key type: Kalshi uses RSA or Ed25519 keys.")

    # ------------------------------------------------------------------ core
    @property
    def can_trade(self) -> bool:
        return self._private_key is not None and bool(self.key_id)

    def _sign_headers(self, method: str, path: str) -> dict[str, str]:
        """Signature covers: timestamp_ms + METHOD + path (no query string).
        Kalshi verifies with the algorithm of the key you registered: Ed25519 signs the text
        directly; RSA uses PSS with SHA-256 (MGF1-SHA256, salt length = digest length)."""
        ts = str(int(time.time() * 1000))
        message = f"{ts}{method.upper()}{path}".encode()
        if isinstance(self._private_key, ed25519.Ed25519PrivateKey):
            signature = self._private_key.sign(message)
        else:
            signature = self._private_key.sign(
                message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    def _request(self, method: str, endpoint: str, *, params: Optional[dict] = None,
                 json: Optional[dict] = None, auth: bool = False, retries: int = 7) -> Any:
        url = f"{self.base_url}{endpoint}"
        for attempt in range(retries + 1):
            headers = {}
            if auth:
                if not self.can_trade:
                    raise KalshiError("This call needs API credentials (key id + private key).")
                # The signed path is the full URL path, e.g. /trade-api/v2/portfolio/orders
                headers = self._sign_headers(method, requests.utils.urlparse(url).path)
            try:
                resp = self._session.request(method, url, params=params, json=json,
                                             headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == retries:
                    raise KalshiError(f"network error: {exc}") from exc
                time.sleep(min(0.5 * 2 ** attempt, 8.0))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == retries:
                    raise KalshiError(f"{resp.status_code} after {retries} retries: {resp.text[:200]}")
                time.sleep(min(0.5 * 2 ** attempt, 8.0))
                continue
            if resp.status_code >= 400:
                raise KalshiError(f"{method} {endpoint} -> {resp.status_code}: {resp.text[:300]}")
            return resp.json() if resp.content else {}
        raise KalshiError("unreachable")

    # ----------------------------------------------------------- market data
    def get_series(self, series_ticker: str) -> dict:
        return self._request("GET", f"/series/{series_ticker}")["series"]

    def get_markets(self, **params) -> dict:
        return self._request("GET", "/markets", params=params)

    def iter_markets(self, **params):
        """Yield every market matching ``params``, following pagination cursors."""
        cursor = ""
        while True:
            page = self.get_markets(limit=1000, cursor=cursor, **params)
            yield from page["markets"]
            cursor = page.get("cursor", "")
            if not cursor:
                return

    def get_market(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}")["market"]

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        return self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_trades(self, ticker: str, min_ts: int, limit: int = 1000) -> list[dict]:
        """Public trades in a market since ``min_ts`` (unix seconds), newest first."""
        out, cursor = [], ""
        while True:
            page = self._request("GET", "/markets/trades",
                                 params={"ticker": ticker, "min_ts": min_ts, "limit": limit,
                                         "cursor": cursor})
            out += page["trades"]
            cursor = page.get("cursor", "")
            if not cursor:
                return out

    def get_candlesticks(self, series: str, ticker: str, start_ts: int, end_ts: int,
                         period_minutes: int = 1) -> list[dict]:
        data = self._request("GET", f"/series/{series}/markets/{ticker}/candlesticks",
                             params={"start_ts": start_ts, "end_ts": end_ts,
                                     "period_interval": period_minutes})
        return data.get("candlesticks", [])

    # -------------------------------------------------------------- portfolio
    def get_balance(self) -> dict:
        return self._request("GET", "/portfolio/balance", auth=True)

    def get_positions(self, **params) -> dict:
        return self._request("GET", "/portfolio/positions", params=params, auth=True)

    def get_order(self, order_id: str) -> dict:
        return self._request("GET", f"/portfolio/orders/{order_id}", auth=True)["order"]

    def create_order(self, **body) -> dict:
        return self._request("POST", "/portfolio/orders", json=body, auth=True)["order"]

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/portfolio/orders/{order_id}", auth=True)
