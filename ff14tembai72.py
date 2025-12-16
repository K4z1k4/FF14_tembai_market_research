from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import time

import requests

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, digamma
from scipy.stats import median_abs_deviation

# ----------------------------
# Universalis I/O
# ----------------------------

class UniversalisClient:
    """
    Minimal Universalis v2 client.

    Uses:
      - GET /api/v2/{world}/{itemId}
      - GET /api/v2/history/{world}/{itemId}?entries=... (default 50, max 200 per docs snippet)
    """
    def __init__(
        self,
        base_url: str = "https://universalis.app",
        timeout_sec: float = 10.0,
        max_retries: int = 3,
        backoff_sec: float = 0.8,
        user_agent: str = "ffxiv-flip-research/0.1",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_err: Optional[Exception] = None
        for i in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout_sec)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                # simple exponential-ish backoff
                time.sleep(self.backoff_sec * (1.5 ** i))
        raise RuntimeError(f"GET failed: {url} params={params} err={last_err}")

    def get_market(self, world: str, item_id: int) -> Dict[str, Any]:
        """
        Returns current listings + recentHistory + lastUploadTime, etc.
        Example fields observed:
          lastUploadTime (ms), listings[*].pricePerUnit, listings[*].quantity, ...
          recentHistory[*].timestamp (sec), recentHistory[*].quantity, ...
        """
        return self._get_json(f"/api/v2/{world}/{item_id}")

    def get_history(
        self,
        world: str,
        item_id: int,
        entries: int = 200,
        *,
        min_sale_price: Optional[int] = None,
        max_sale_price: Optional[int] = None,
        hq: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Returns sale history.
        Observed fields:
          lastUploadTime (ms), entries[*] (sales), worldName, etc.
        """
        # docs snippet indicates entries has default 50, max 200 -> clip defensively
        entries = int(max(1, min(entries, 200)))

        params: Dict[str, Any] = {"entries": entries}
        if min_sale_price is not None:
            params["minSalePrice"] = int(min_sale_price)
        if max_sale_price is not None:
            params["maxSalePrice"] = int(max_sale_price)
        if hq is not None:
            # Universalis commonly uses "hq" on query; keep as-is.
            params["hq"] = "true" if hq else "false"

        return self._get_json(f"/api/v2/history/{world}/{item_id}", params=params)


# ----------------------------
# Data model helpers
# ----------------------------

@dataclass(frozen=True)
class Sale:
    timestamp_sec: int
    quantity: int
    price_per_unit: int
    hq: Optional[bool] = None
    buyer_name: Optional[str] = None


def _coalesce(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def parse_sales_from_history(history_json: Dict[str, Any]) -> List[Sale]:
    """
    Parses history endpoint response:
      { ..., "entries": [ { "timestamp": ..., "quantity": ..., "pricePerUnit": ... , ...}, ... ] }
    """
    entries = history_json.get("entries", [])
    out: List[Sale] = []
    for e in entries:
        ts = e.get("timestamp")
        qty = e.get("quantity")
        ppu = e.get("pricePerUnit")
        if ts is None or qty is None or ppu is None:
            continue
        out.append(Sale(
            timestamp_sec=int(ts),
            quantity=int(qty),
            price_per_unit=int(ppu),
            hq=e.get("hq"),
            buyer_name=e.get("buyerName"),
        ))
    return out


def parse_last_upload_ms(j: Dict[str, Any]) -> Optional[int]:
    # Observed as "lastUploadTime" in ms :contentReference[oaicite:2]{index=2}
    v = j.get("lastUploadTime")
    return int(v) if v is not None else None


# ----------------------------
# Feature calculations: k72 Δh, μ
# ----------------------------

def now_epoch_sec() -> int:
    return int(time.time())


def delta_hours_from_last_upload(last_upload_ms: Optional[int], now_sec: Optional[int] = None) -> Optional[float]:
    if last_upload_ms is None:
        return None
    if now_sec is None:
        now_sec = now_epoch_sec()
    # lastUploadTime is ms; now is sec
    return max(0.0, (now_sec - (last_upload_ms / 1000.0)) / 3600.0)


def count_sold_quantity_in_window(sales: Iterable[Sale], window_hours: float, now_sec: Optional[int] = None) -> int:
    if now_sec is None:
        now_sec = now_epoch_sec()
    since = now_sec - int(window_hours * 3600.0)
    return sum(s.quantity for s in sales if s.timestamp_sec >= since)


def k72_and_delta_h(
    history_json: Dict[str, Any],
    *,
    now_sec: Optional[int] = None,
) -> Tuple[int, Optional[float]]:
    if now_sec is None:
        now_sec = now_epoch_sec()
    sales = parse_sales_from_history(history_json)
    k72 = count_sold_quantity_in_window(sales, 72.0, now_sec=now_sec)
    delta_h = delta_hours_from_last_upload(parse_last_upload_ms(history_json), now_sec=now_sec)
    return k72, delta_h


def effective_window_hours(window_hours: float, delta_h: Optional[float]) -> float:
    """
    Effective observation time for staleness.
    - delta_h = 0  -> tau
    - delta_h grows -> decays smoothly
    """
    if delta_h is None:
        return window_hours
    return float(window_hours * math.exp(-delta_h / max(1e-9, 24.0)))


# ----------------------------
# Bayesian core: Poisson-Gamma (λ is "mean sales per 24h")
# ----------------------------

def posterior_gamma_params(
    alpha: float,
    beta: float,
    k_obs: int,
    obs_hours_eff: float,
) -> Tuple[float, float]:
    """
    Model:
      λ ~ Gamma(alpha, beta)  [shape=alpha, rate=beta], where λ is mean sales per 24h
      k | λ ~ Poisson( λ * (obs_hours_eff / 24) )

    Posterior:
      λ | k ~ Gamma(alpha + k, beta + obs_hours_eff/24)
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha, beta must be > 0")
    if k_obs < 0:
        raise ValueError("k_obs must be >= 0")
    t = max(0.0, obs_hours_eff / 24.0)
    return (alpha + k_obs, beta + t)


def posterior_mean_lambda_per_24h(alpha_post: float, beta_post: float) -> float:
    # E[λ | data] = alpha_post / beta_post
    return alpha_post / beta_post


def world_mu_from_k72(
    alpha: float,
    beta: float,
    k72: int,
    delta_h: Optional[float],
    *,
    window_hours: float = 72.0,
) -> float:
    """
    Returns μ: expected sales quantity in the next 24h on that world.
    """
    obs_eff = effective_window_hours(window_hours, delta_h)
    a_post, b_post = posterior_gamma_params(alpha, beta, k72, obs_eff)
    return posterior_mean_lambda_per_24h(a_post, b_post)


# ----------------------------
# End-to-end per world: fetch -> (k72, Δh, μ)
# ----------------------------

def compute_world_metrics(
    client: UniversalisClient,
    world: str,
    item_id: int,
    *,
    alpha: float,
    beta: float,
    window_hours: float = 72.0,
    history_entries: int = 200,
    now_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Output keys (you said you want):
      - mu
      - k72
      - delta_h (Δh)
    (score is intentionally left out for now)
    """
    if now_sec is None:
        now_sec = now_epoch_sec()

    hist = client.get_history(world, item_id, entries=history_entries)
    k72, delta_h = k72_and_delta_h(hist, now_sec=now_sec)
    mu = world_mu_from_k72(alpha, beta, k72, delta_h, window_hours=window_hours)

    return {
        "world": world,
        "item_id": item_id,
        "k72": int(k72),
        "delta_h": float(delta_h) if delta_h is not None else None,
        "mu": float(mu),
        "last_upload_ms": parse_last_upload_ms(hist),
    }


def aggregate_item_mu(world_metrics: Iterable[Dict[str, Any]]) -> float:
    """
    Item-level expected sales in next 24h:
      μ_sum = Σ_world μ_world
    """
    s = 0.0
    for m in world_metrics:
        mu = m.get("mu")
        if mu is not None:
            s += float(mu)
    return s



@dataclass(frozen=True)
class AlphaBetaFit:
    alpha: float
    beta: float
    loglik: float
    n_obs: int
    success: bool
    message: str
    n_iter: int


def _mom_initial_alpha_beta(K: np.ndarray, W: float) -> Tuple[float, float]:
    """
    MoMの初期値（かなり効きます）
    モデル:
      λ ~ Gamma(α,β) (rate)
      K | λ ~ Poisson(W λ)
    周辺で:
      E[K] = W α / β
      Var[K] = E[K] + (E[K]^2)/α
    """
    m = float(np.mean(K))
    v = float(np.var(K, ddof=1)) if K.size >= 2 else max(m, 1e-6)

    # 過分散がない（v<=m）場合は α を大きくしてほぼポアソン寄せ
    if v <= m + 1e-12:
        alpha = 100.0
        beta = alpha * W / max(m, 1e-6)
        return alpha, max(beta, 1e-6)

    alpha = (m * m) / max(v - m, 1e-12)
    beta = alpha * W / max(m, 1e-6)

    # 極端値を軽くクリップ（最適化の発散防止）
    alpha = float(np.clip(alpha, 1e-3, 1e6))
    beta = float(np.clip(beta, 1e-6, 1e9))
    return alpha, beta


def fit_alpha_beta_mle_scipy(
    Ks: List[int],
    *,
    W_days: float = 3.0,
    init: Optional[Tuple[float, float]] = None,
    maxiter: int = 500,
) -> AlphaBetaFit:
    """
    DC内の world ごとの 3日合計売上 K_j を入力して、(α,β) を周辺尤度最大化で推定。
    - α>0,β>0 を確実に守るため、a=log α, b=log β で最適化します。
    - 目的関数は負の対数尤度（NLL）。勾配も解析的に入れて高速化。
    """
    if len(Ks) == 0:
        raise ValueError("Ks is empty")

    K = np.asarray(Ks, dtype=np.float64)
    if np.any(K < 0):
        raise ValueError("Ks must be >= 0")
    W = float(W_days)
    if W <= 0:
        raise ValueError("W_days must be > 0")

    # 初期値
    if init is None:
        a0, b0 = _mom_initial_alpha_beta(K, W)
    else:
        a0, b0 = init
        if a0 <= 0 or b0 <= 0:
            raise ValueError("init alpha,beta must be > 0")

    x0 = np.array([math.log(a0), math.log(b0)], dtype=np.float64)

    # 周辺対数尤度（定数項も含める）
    # ll = Σ [ lgamma(K+α)-lgamma(α)-lgamma(K+1) + K log W + α log β - (K+α) log(β+W) ]
    # これを最大化 <=> NLL = -ll を最小化
    def nll_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
        a = float(np.exp(x[0]))  # α
        b = float(np.exp(x[1]))  # β

        # log-likelihood
        ll = np.sum(
            gammaln(K + a) - gammaln(a) - gammaln(K + 1.0)
            + K * math.log(W)
            + a * math.log(b)
            - (K + a) * math.log(b + W)
        )
        nll = -float(ll)

        # dℓ/dα = Σ[ ψ(K+α) - ψ(α) + log β - log(β+W) ]
        dldalpha = float(np.sum(digamma(K + a) - digamma(a) + math.log(b) - math.log(b + W)))

        # dℓ/dβ = Σ[ α/β - (K+α)/(β+W) ]
        dldbeta = float(np.sum(a / b - (K + a) / (b + W)))

        # chain rule: α = exp(x0), β = exp(x1)
        # dℓ/dx0 = dℓ/dα * α
        # dℓ/dx1 = dℓ/dβ * β
        grad = np.array([-dldalpha * a, -dldbeta * b], dtype=np.float64)  # NLLなのでマイナス
        return nll, grad

    def fun(x: np.ndarray) -> float:
        return nll_and_grad(x)[0]

    def jac(x: np.ndarray) -> np.ndarray:
        return nll_and_grad(x)[1]

    res = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        jac=jac,
        options={"maxiter": int(maxiter)},
    )

    alpha_hat = float(np.exp(res.x[0]))
    beta_hat = float(np.exp(res.x[1]))
    loglik = -float(res.fun)

    usable = (
    np.isfinite(alpha_hat) and alpha_hat > 0 and
    np.isfinite(beta_hat)  and beta_hat  > 0 and
    np.isfinite(loglik)
    )

    usable = usable and (1e-3 <= alpha_hat <= 1e6) and (1e-6 <= beta_hat <= 1e9)

    return AlphaBetaFit(
        alpha=alpha_hat,
        beta=beta_hat,
        loglik=loglik,
        n_obs=int(K.size),
        success=bool(res.success) or usable,
        message=str(res.message),
        n_iter=int(res.nit) if hasattr(res, "nit") else -1,
    )

def sum_sold_quantity_in_days_from_history_json(history_json: dict, *, since_sec: int) -> int:
    K = 0
    for e in history_json.get("entries", []):
        ts = e.get("timestamp")
        qty = e.get("quantity")
        if ts is None or qty is None:
            continue
        if int(ts) >= since_sec:
            K += int(qty)
    return K


def fetch_K3d_per_world(
    client,
    world: str,
    item_id: int,
    *,
    window_days: int = 3,
    history_entries: int = 500,
    now_sec: Optional[int] = None,
) -> int:
    if now_sec is None:
        now_sec = int(time.time())
    since = now_sec - int(window_days * 24 * 3600)
    try:
        hist = client.get_history(world, item_id, entries=history_entries)
    except RuntimeError as e:
        return None
    return sum_sold_quantity_in_days_from_history_json(hist, since_sec=since)


def fit_alpha_beta_per_item_dc_scipy(
    client,
    *,
    item_id: int,
    worlds: List[str],
    world_to_dc: Dict[str, str],
    window_days: int = 3,
    history_entries: int = 500,
    now_sec: Optional[int] = None,
    min_worlds_in_dc: int = 3,
    maxiter: int = 500,
) -> Dict[str, AlphaBetaFit]:
    """
    1アイテムについて、DCごとに Ks を作って SciPy MLE で(α,β)推定。
    """
    if now_sec is None:
        now_sec = int(time.time())

    dc_to_Ks: Dict[str, List[int]] = {}
    for w in worlds:
        dc = world_to_dc.get(w)
        if dc is None:
            continue
        K = fetch_K3d_per_world(
            client, w, item_id,
            window_days=window_days,
            history_entries=history_entries,
            now_sec=now_sec,
        )
        dc_to_Ks.setdefault(dc, []).append(K)

    out: Dict[str, AlphaBetaFit] = {}
    for dc, Ks in dc_to_Ks.items():
        if len(Ks) < min_worlds_in_dc:
            continue
        fit = fit_alpha_beta_mle_scipy(
            Ks,
            W_days=float(window_days),
            init=None,          # MoM初期値（自動）
            maxiter=maxiter,
        )
        out[dc] = fit

    return out

def rotation_metrics_from_posterior(a_post: float, b_post: float) -> Dict[str, float]:
    """
    事後: λ|data ~ Gamma(a_post, b_post) (rate)
    ここで λ は「次の24hの平均販売個数」。

    次の24h販売個数 X の事後予測についてのモーメントを返す。
    """
    mu = a_post / b_post
    var = mu + (mu * mu) / a_post
    cv = math.sqrt(var) / mu

    # NB (r=a_post, p=b/(b+1)) の歪度
    p = b_post / (b_post + 1.0)
    skew = (2.0 - p) / math.sqrt(a_post * (1.0 - p))
    skew_sat = skew / (1.0 + skew)

    return {
        "mu": float(mu),
        "var": float(var),
        "cv": float(cv),
        "skew": float(skew),
        "skew_sat": float(skew_sat),
    }


def rotation_score_preset3(metrics: Dict[str, float], *, preset_lambda: float = 1.0, preset_kappa: float = 0.1) -> float:
    """
    これは「回転寄りのスコア（安定重視プリセット3）」。
    ※あなたの言う最終スコアではなく、回転の良さを1値に潰すための中間量。
    """
    mu = metrics["mu"]
    cv = metrics["cv"]
    skew_sat = metrics["skew_sat"]
    return float((mu / (1.0 + preset_lambda * cv)) * (1.0 + preset_kappa * skew_sat))


def compute_rotation_bundle_world_item_dc(
    client,
    *,
    world: str,
    item_id: int,
    world_to_dc: Dict[str, str],
    dc_params: Dict[str, Tuple[float, float]],  # dc -> (alpha, beta)
    window_hours: float = 72.0,
    history_entries: int = 200,
    now_sec: Optional[int] = None,
    preset_lambda: float = 1.0,
    preset_kappa: float = 0.1,
    fallback_params: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    (world,item) について「回転関連だけ」を返す:
      - k72, delta_h
      - mu（次24h期待販売個数）
      - rotation_score（回転寄り中間スコア）
      - cv, skew 等（Python内部用だが返しておく：統合スコアで使うなら便利）
    """
    if now_sec is None:
        import time
        now_sec = int(time.time())

    dc = world_to_dc.get(world)
    if dc is None:
        if fallback_params is None:
            raise KeyError(f"world_to_dc has no entry for world={world}")
        alpha, beta = fallback_params
        dc_used = None
    else:
        params = dc_params.get(dc)
        if params is None:
            if fallback_params is None:
                raise KeyError(f"dc_params has no entry for dc={dc} (world={world})")
            alpha, beta = fallback_params
        else:
            alpha, beta = params
        dc_used = dc

    hist = client.get_history(world, item_id, entries=history_entries)
    k72, delta_h = k72_and_delta_h(hist, now_sec=now_sec)

    obs_eff_hours = effective_window_hours(window_hours, delta_h)
    a_post, b_post = posterior_gamma_params(alpha, beta, k72, obs_eff_hours)

    m = rotation_metrics_from_posterior(a_post, b_post)
    rot_score = rotation_score_preset3(m, preset_lambda=preset_lambda, preset_kappa=preset_kappa)

    return {
        "world": world,
        "item_id": int(item_id),
        "dc": dc_used,
        "alpha_used": float(alpha),
        "beta_used": float(beta),
        "k72": int(k72),
        "delta_h": float(delta_h) if delta_h is not None else None,
        "mu": m["mu"],
        "rotation_score": float(rot_score),
        # ここから下は「統合スコア」に使うなら便利（不要なら捨ててOK）
        "cv": m["cv"],
        "skew_sat": m["skew_sat"],
        "updated_at_sec": int(now_sec),
    }

def fetch_market_v2_with_params(
    client,
    world_dc_region: str,
    item_id: int,
    *,
    listing_limit: int = 100,
    hq: Optional[bool] = None,
    tax: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    /api/v2/{worldDcRegion}/{itemId}?listingLimit=...&hq=...&tax=...
    - client に _get_json がある前提（あなたの UniversalisClient にはある）
    """
    params: Dict[str, Any] = {"listingLimit": int(listing_limit)}
    if hq is not None:
        params["hq"] = "true" if hq else "false"
    if tax is not None:
        params["tax"] = "true" if tax else "false"

    # 既存クラス実装に依存（再定義はしない）
    return client._get_json(f"/api/v2/{world_dc_region}/{int(item_id)}", params=params)
def robust_floor_mean_price_from_listings_mad(
    listings: List[Dict[str, Any]],
    *,
    scan_n: int = 60,
    near_ratio: float = 0.1,
    mad_k: float = 6.0,
    min_keep: int = 3,
    weight_by_quantity: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    MADで「異常に安すぎる」外れ値を落としてから、
    最安値近傍 (p <= p0*(1+near_ratio)) の平均を返す。

    - scan_n: 価格昇順で先頭から何件見るか
    - mad_k : 低すぎ判定の閾値（大きいほど緩い）
    - near_ratio: 最安値p0から何%までを「最安値付近」とするか
    - min_keep: 近傍が少なすぎたら先頭min_keep件を使う
    """
    pts: List[Tuple[int, int]] = []
    for x in listings:
        p = x.get("pricePerUnit")
        if p is None:
            continue
        p = int(p)
        if p <= 0:
            continue
        q = x.get("quantity", 1)
        try:
            q = int(q)
        except Exception:
            q = 1
        q = max(1, q)
        pts.append((p, q))

    if not pts:
        raise ValueError("No valid listings (pricePerUnit)")

    pts.sort(key=lambda t: t[0])
    pts = pts[: max(1, int(scan_n))]

    prices = np.array([p for p, _ in pts], dtype=np.float64)

    med = float(np.median(prices))
    mad = float(median_abs_deviation(prices, scale=1.0, nan_policy="omit"))

    # modified z-score: 0.6745*(x-med)/MAD
    # 「低すぎる側」だけ落とす
    if mad > 0.0:
        mz = 0.6745 * (prices - med) / mad
        keep = (mz >= -float(mad_k))
        filtered = [pt for pt, ok in zip(pts, keep.tolist()) if ok]
    else:
        filtered = pts

    if not filtered:
        filtered = pts

    p0 = min(p for p, _ in filtered)
    thr = p0 * (1.0 + float(near_ratio))

    near = [(p, q) for (p, q) in filtered if p <= thr]
    if len(near) < int(min_keep):
        near = filtered[: int(min_keep)]

    if weight_by_quantity:
        num = sum(p * q for p, q in near)
        den = sum(q for _, q in near)
        mean_price = num / max(1, den)
    else:
        mean_price = sum(p for p, _ in near) / len(near)

    debug = {
        "scan_n": int(scan_n),
        "near_ratio": float(near_ratio),
        "mad_k": float(mad_k),
        "median": float(med),
        "mad": float(mad),
        "p0": int(p0),
        "thr": float(thr),
        "used_n": int(len(near)),
        "used_prices": [int(p) for p, _ in near],
    }
    return float(mean_price), debug


def estimate_region_floor_price_mad(
    client,
    *,
    region: str,
    item_id: int,
    listing_limit: int = 120,
    hq: Optional[bool] = None,
    tax: Optional[bool] = None,
    scan_n: int = 60,
    near_ratio: float = 0.03,
    mad_k: float = 6.0,
    min_keep: int = 3,
    weight_by_quantity: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    リージョン全体の listings から、MADで外れ値を落として
    「最安値付近の頑健平均」を返す。
    """
    j = fetch_market_v2_with_params(
        client,
        region,
        item_id,
        listing_limit=listing_limit,
        hq=hq,
        tax=tax,
    )
    listings = j.get("listings", [])
    price, dbg = robust_floor_mean_price_from_listings_mad(
        listings,
        scan_n=scan_n,
        near_ratio=near_ratio,
        mad_k=mad_k,
        min_keep=min_keep,
        weight_by_quantity=weight_by_quantity,
    )
    dbg.update({"region": region, "item_id": int(item_id), "listing_limit": int(listing_limit)})
    return price, dbg


def compute_buy_sell_prices_by_region_min_mad(
    client,
    *,
    item_id: int,
    buy_region: str,
    sell_region: str,
    listing_limit: int = 120,
    scan_n: int = 60,
    near_ratio: float = 0.03,
    mad_k: float = 6.0,
    min_keep: int = 3,
    weight_by_quantity: bool = False,
) -> Dict[str, Any]:
    """
    buy_price / sell_price を「リージョン最安値付近（MADで外れ値除外）」で決める。
    """
    buy_price, buy_dbg = estimate_region_floor_price_mad(
        client,
        region=buy_region,
        item_id=item_id,
        listing_limit=listing_limit,
        scan_n=scan_n,
        near_ratio=near_ratio,
        mad_k=mad_k,
        min_keep=min_keep,
        weight_by_quantity=weight_by_quantity,
    )
    sell_price, sell_dbg = estimate_region_floor_price_mad(
        client,
        region=sell_region,
        item_id=item_id,
        listing_limit=listing_limit,
        scan_n=scan_n,
        near_ratio=near_ratio,
        mad_k=mad_k,
        min_keep=min_keep,
        weight_by_quantity=weight_by_quantity,
    )
    return {
        "item_id": int(item_id),
        "buy_region": buy_region,
        "sell_region": sell_region,
        "buy_price": float(buy_price) * 1.05,
        "sell_price": float(sell_price) * 0.95,
        "buy_debug": buy_dbg,
        "sell_debug": sell_dbg,
    }    
def compute_profit_rotation_score(
    *,
    mu_24h: float,
    buy_price: float,
    sell_price: float,
) -> Dict[str, float]:
    """
    最小の「本質」:
      spread = sell_price - buy_price
      expected_profit_24h = mu * spread

    ※手数料・税は除外（あなたの方針）
    """
    spread = float(sell_price) - float(buy_price)
    expected_profit_24h = float(mu_24h) * spread
    return {
        "spread": float(spread),
        "expected_profit_24h": float(expected_profit_24h),
    }


def compute_final_metrics_world_item(
    client,
    *,
    world: str,
    item_id: int,
    # 回転側（DC別αβ）
    world_to_dc: Dict[str, str],
    dc_params_for_item: Dict[str, tuple],  # dc -> (alpha, beta)
    # 価格側（リージョン最安付近）
    buy_region: str,
    sell_region: str,
    # universalis fetch knobs
    history_entries: int = 200,
    window_hours: float = 72.0,
    listing_limit: int = 120,
    # robust price knobs (MAD)
    scan_n: int = 60,
    near_ratio: float = 0.03,
    mad_k: float = 6.0,
    min_keep: int = 3,
    weight_by_quantity: bool = False,
    # misc
    now_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """
    返すもの（必要な最小構成）:
      - mu, k72, Δh（回転）
      - buy_price, sell_price, spread（価格）
      - expected_profit_24h（最終：利益×回転）

    ※最終スコアは expected_profit_24h を採用（現段階）
    """
    # 1) 回転（muなど）
    rot = compute_rotation_bundle_world_item_dc(
        client,
        world=world,
        item_id=item_id,
        world_to_dc=world_to_dc,
        dc_params=dc_params_for_item,
        history_entries=history_entries,
        window_hours=window_hours,
        now_sec=now_sec,
        preset_lambda=1.0,
        preset_kappa=0.1,
    )

    # 2) 価格（リージョン最安付近）
    px = compute_buy_sell_prices_by_region_min_mad(
        client,
        item_id=item_id,
        buy_region=buy_region,
        sell_region=sell_region,
        listing_limit=listing_limit,
        scan_n=scan_n,
        near_ratio=near_ratio,
        mad_k=mad_k,
        min_keep=min_keep,
        weight_by_quantity=weight_by_quantity,
    )

    # 3) 本質（利益×回転）
    pr = compute_profit_rotation_score(
        mu_24h=float(rot["mu"]),
        buy_price=float(px["buy_price"]),
        sell_price=float(px["sell_price"]),
    )

    return {
        "world": world,
        "item_id": int(item_id),
        "buy_region": buy_region,
        "sell_region": sell_region,
        # 回転
        "k72": rot["k72"],
        "delta_h": rot["delta_h"],
        "mu": rot["mu"],
        # 価格
        "buy_price": px["buy_price"],
        "sell_price": px["sell_price"],
        "spread": pr["spread"],
        # 最終
        "expected_profit_24h": pr["expected_profit_24h"],
        # デバッグ用（必要なら見る）
        "rotation_score": rot.get("rotation_score"),
        "price_debug_buy": px.get("buy_debug"),
        "price_debug_sell": px.get("sell_debug"),
    }
if __name__ == "__main__":
    uni = UniversalisClient()

    ITEM_ID = 47964
    TARGET_WORLD = "Siren"
    DC_NAME = "Aether"

    # Aether worlds（推定用の観測点）
    aether_worlds = [
        "Adamantoise",
        "Cactuar",
        "Faerie",
        "Gilgamesh",
        "Jenova",
        "Midgardsormr",
        "Sargatanas",
        "Siren",
    ]
    world_to_dc = {w: DC_NAME for w in aether_worlds}

    # 1) DC別(ここではAether)の α,β を推定（3日固定）
    fits = fit_alpha_beta_per_item_dc_scipy(
        uni,
        item_id=ITEM_ID,
        worlds=aether_worlds,
        world_to_dc=world_to_dc,
        window_days=3,
        history_entries=200,
        min_worlds_in_dc=3,
        maxiter=800,
    )
    fit = fits[DC_NAME]
    print(f"[FIT] item={ITEM_ID} dc={DC_NAME} alpha={fit.alpha:.6g} beta={fit.beta:.6g} loglik={fit.loglik:.6g} success={fit.success}")

    dc_params_for_item = {DC_NAME: (fit.alpha, fit.beta)}

    # 2) 最終指標まで（利益×回転）
    out = compute_final_metrics_world_item(
        uni,
        world=TARGET_WORLD,
        item_id=ITEM_ID,
        world_to_dc=world_to_dc,              # TARGET_WORLD が含まれていればOK
        dc_params_for_item=dc_params_for_item,
        buy_region="Japan",
        sell_region="North-America",
        history_entries=200,
        window_hours=72.0,
        listing_limit=120,
        scan_n=60,
        near_ratio=0.03,
        mad_k=6.0,
        min_keep=3,
    )

    print(f"[PRED] world={TARGET_WORLD} item={ITEM_ID}")
    print(f"  k72     = {out['k72']}")
    print(f"  Δh      = {out['delta_h']}")
    print(f"  mu      = {out['mu']:.6g}           # 次24h期待販売個数")
    print(f"  buy     = {out['buy_price']:.6g}    # Japan 最安付近(MAD)")
    print(f"  sell    = {out['sell_price']:.6g}   # NA 最安付近(MAD)")
    print(f"  spread  = {out['spread']:.6g}")
    print(f"  E[profit_24h] = {out['expected_profit_24h']:.6g}")
