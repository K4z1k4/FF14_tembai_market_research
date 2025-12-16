import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import math

import gspread
from google.oauth2.service_account import Credentials

import ff14tembai72 as core

JST = timezone(timedelta(hours=9))
SHEET_REGION_TO_UNI = {"JP": "Japan", "NA": "North-America", "EU": "Europe"}


def now_jst() -> datetime:
    return datetime.now(JST)


def dt_to_iso(dt: datetime) -> str:
    return dt.astimezone(JST).isoformat(timespec="seconds")


def parse_dt_maybe(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def open_sheet(spreadsheet_id: str, creds_json_path: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(creds_json_path, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(spreadsheet_id)


def get_headers(ws) -> List[str]:
    return [h.strip() for h in ws.row_values(1)]


def header_index(headers: List[str]) -> Dict[str, int]:
    return {h: i for i, h in enumerate(headers) if h}


def read_config_kv(ws) -> Dict[str, Any]:
    rows = ws.get_all_values()
    out: Dict[str, Any] = {}
    for r in rows[1:]:
        if len(r) < 2:
            continue
        k = str(r[0]).strip()
        v = str(r[1]).strip()
        if not k:
            continue
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except Exception:
            out[k] = v
    return out


def read_items_enabled(ws) -> List[Tuple[int, str]]:
    rows = ws.get_all_values()
    out: List[Tuple[int, str]] = []
    for r in rows[1:]:
        if len(r) < 3:
            continue
        try:
            item_id = int(str(r[0]).strip())
        except Exception:
            continue
        name = str(r[1]).strip()
        enable = str(r[2]).strip().upper() in ("TRUE", "1", "YES", "Y")
        if enable:
            out.append((item_id, name))
    return out


def read_world_master_all(ws) -> List[Dict[str, Any]]:
    # WORLD_MASTER: world, dc, region(JP/NA/EU), enable
    rows = ws.get_all_values()
    out: List[Dict[str, Any]] = []
    for r in rows[1:]:
        if len(r) < 4:
            continue
        world = str(r[0]).strip()
        dc = str(r[1]).strip()
        region = str(r[2]).strip()
        enable = str(r[3]).strip().upper() in ("TRUE", "1", "YES", "Y")
        if world and dc and region:
            out.append({"world": world, "dc": dc, "region": region, "enable": enable})
    return out


@dataclass(frozen=True)
class ABRow:
    item_id: int
    dc: str
    alpha: float
    beta: float
    fitted_at: str
    n_worlds: int
    loglik: float
    success: bool


def read_ab_params(ws) -> Dict[Tuple[int, str], ABRow]:
    rows = ws.get_all_values()
    out: Dict[Tuple[int, str], ABRow] = {}
    for r in rows[1:]:
        if len(r) < 8:
            continue
        try:
            item_id = int(str(r[0]).strip())
        except Exception:
            continue
        dc = str(r[1]).strip()
        if not dc:
            continue
        try:
            alpha = float(r[2]); beta = float(r[3])
        except Exception:
            continue
        fitted_at = str(r[4]).strip()
        try:
            n_worlds = int(float(r[5]))
        except Exception:
            n_worlds = 0
        try:
            loglik = float(r[6])
        except Exception:
            loglik = float("nan")
        success = str(r[7]).strip().upper() in ("TRUE", "1", "YES", "Y")
        out[(item_id, dc)] = ABRow(item_id, dc, alpha, beta, fitted_at, n_worlds, loglik, success)
    return out


def update_table_by_headers(ws, headers: List[str], rows: List[List[Any]]):
    # A2から全置換（ヘッダ行は触らない）
    n_cols = len(headers)
    if n_cols == 0:
        raise RuntimeError(f"Worksheet '{ws.title}' has empty header row.")
    if not rows:
        ws.update(f"A2:{gspread.utils.rowcol_to_a1(2, n_cols)}", [[""] * n_cols])
        return
    norm: List[List[Any]] = []
    for r in rows:
        if len(r) < n_cols:
            r = r + [""] * (n_cols - len(r))
        elif len(r) > n_cols:
            r = r[:n_cols]
        norm.append(r)
    end_a1 = gspread.utils.rowcol_to_a1(len(norm) + 1, n_cols)
    #ws.update(f"A2:{end_a1}", norm)
    norm = [[_sanitize_cell(x) for x in r] for r in norm]
    ws.update(range_name=f"A2:{end_a1}", values=norm)



def update_table_partial(ws, n_cols: int, rows: List[List[Any]]):
    # A2から「左n_cols列」だけ置換（右側は触らない）
    if n_cols <= 0:
        return
    if not rows:
        ws.update(f"A2:{gspread.utils.rowcol_to_a1(2, n_cols)}", [[""] * n_cols])
        return
    norm: List[List[Any]] = []
    for r in rows:
        r = r[:n_cols] if len(r) >= n_cols else (r + [""] * (n_cols - len(r)))
        norm.append(r)
    end_a1 = gspread.utils.rowcol_to_a1(len(norm) + 1, n_cols)
    norm = [[_sanitize_cell(x) for x in r] for r in norm]
    ws.update(range_name=f"A2:{end_a1}", values=norm)

   # ws.update(f"A2:{end_a1}", norm)


def main():
    spreadsheet_id = require_env("FF14_TEMBAI_SHEET_ID")
    creds_json_path = require_env("FF14_TEMBAI_GOOGLE_CREDENTIALS_JSON")

    sh = open_sheet(spreadsheet_id, creds_json_path)

    ws_config = sh.worksheet("CONFIG")
    ws_items = sh.worksheet("ITEMS")
    ws_world = sh.worksheet("WORLD_MASTER")
    ws_prices = sh.worksheet("ITEM_PRICES")
    ws_world_item = sh.worksheet("WORLD_ITEM")
    ws_ab = sh.worksheet("AB_PARAMS")

    cfg = read_config_kv(ws_config)
    items = read_items_enabled(ws_items)
    worlds_all = read_world_master_all(ws_world)
    ab_old = read_ab_params(ws_ab)

    # knobs（CONFIG）
    window_hours = float(cfg.get("window_hours", 72))
    fit_days = int(cfg.get("fit_days", 3))

    listing_limit = int(cfg.get("listing_limit", 120))
    scan_n = int(cfg.get("scan_n", 60))
    near_ratio = float(cfg.get("near_ratio", 0.03))
    mad_k = float(cfg.get("mad_k", 6))
    min_keep = int(cfg.get("min_keep", 3))

    history_entries_rot = int(cfg.get("history_entries_rot", 200))
    history_entries_fit = int(cfg.get("history_entries_fit", 500))
    min_worlds_in_dc = int(cfg.get("min_worlds_in_dc", 3))
    maxiter = int(cfg.get("fit_maxiter", 500))

    preset_lambda = float(cfg.get("preset_lambda", 1.0))
    preset_kappa = float(cfg.get("preset_kappa", 0.1))

    now = now_jst()
    now_iso = dt_to_iso(now)
    now_sec = int(now.timestamp())

    # worlds for fitting: ALL (enable true/false)
    world_to_dc_all = {w["world"]: w["dc"] for w in worlds_all}
    dc_to_worlds_fit: Dict[str, List[str]] = {}
    for w in worlds_all:
        dc_to_worlds_fit.setdefault(w["dc"], []).append(w["world"])

    # worlds for output / trading: enabled only
    worlds_enabled = [w for w in worlds_all if w["enable"]]
    enabled_worlds = [w["world"] for w in worlds_enabled]
    world_to_dc_enabled = {w["world"]: w["dc"] for w in worlds_enabled}
    world_to_region_enabled = {w["world"]: w["region"] for w in worlds_enabled}
    enabled_dcs = set(dc_to_worlds_fit.keys())

    uni = core.UniversalisClient()

    # 1) AB_PARAMS: 24h経過分だけ再推定（推定には全worldを使う）
    ab_new: Dict[Tuple[int, str], ABRow] = dict(ab_old)

    for item_id, _name in items:
        for dc, wlist_all in dc_to_worlds_fit.items():
            if len(wlist_all) < min_worlds_in_dc:
                continue

            key = (item_id, dc)
            need_fit = True
            if key in ab_new:
                if not bool(ab_new[key].success):
                    need_fit = True
                else:
                    dt = parse_dt_maybe(ab_new[key].fitted_at)
                    if dt is not None and (now - dt) < timedelta(hours=24):
                        need_fit = False
            if not need_fit:
                continue

            fits = core.fit_alpha_beta_per_item_dc_scipy(
                uni,
                item_id=item_id,
                worlds=wlist_all,
                world_to_dc={w: dc for w in wlist_all},
                window_days=fit_days,
                history_entries=history_entries_fit,
                now_sec=now_sec,
                min_worlds_in_dc=min_worlds_in_dc,
                maxiter=maxiter,
            )
            fit = fits.get(dc)
            if fit is None:
                continue

            ab_new[key] = ABRow(
                item_id=item_id,
                dc=dc,
                alpha=float(fit.alpha),
                beta=float(fit.beta),
                fitted_at=now_iso,
                n_worlds=int(len(wlist_all)),
                loglik=float(getattr(fit, "loglik", float("nan"))),
                success=bool(getattr(fit, "success", True)),
            )

    # item_id -> (dc -> (alpha,beta))
    dc_params_by_item: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for (iid, dc), abr in ab_new.items():
        dc_params_by_item.setdefault(iid, {})[dc] = (abr.alpha, abr.beta)

    # 2) ITEM_PRICES
    prices_headers = get_headers(ws_prices)
    prices_idx = header_index(prices_headers)

    required_prices = {"item_id", "price_JP", "price_NA", "price_EU", "updated_at"}
    missing_prices = [c for c in required_prices if c not in prices_idx]
    if missing_prices:
        raise RuntimeError(f"ITEM_PRICES missing headers: {missing_prices}")

    n_cols_prices = len(prices_headers)
    price_rows: List[List[Any]] = []

    for item_id, _name in items:
        row = [""] * n_cols_prices
        row[prices_idx["item_id"]] = item_id
        row[prices_idx["updated_at"]] = now_iso
        for sheet_reg, uni_reg in SHEET_REGION_TO_UNI.items():
            ci = prices_idx.get(f"price_{sheet_reg}")
            if ci is None:
                continue
            try:
                p, _dbg = core.estimate_region_floor_price_mad(
                    uni,
                    region=uni_reg,
                    item_id=item_id,
                    listing_limit=listing_limit,
                    scan_n=scan_n,
                    near_ratio=near_ratio,
                    mad_k=mad_k,
                    min_keep=min_keep,
                    weight_by_quantity=False,
                )
                row[ci] = float(p)
            except Exception:
                row[ci] = ""
        price_rows.append(row)

    # 3) WORLD_ITEM（出力は enabled world のみ / world_enabled列は上書きしない）
    wi_headers = get_headers(ws_world_item)
    wi_idx = header_index(wi_headers)

    required_wi = {"item_id", "world", "sell_region", "k72", "delta_h", "mu", "var", "skew", "rot_score", "updated_at"}
    missing_wi = [c for c in required_wi if c not in wi_idx]
    if missing_wi:
        raise RuntimeError(f"WORLD_ITEM missing headers: {missing_wi}")

    # 書き込み列数：world_enabled があればその左まで（＝最後の列は触らない）
    if "world_enabled" in wi_idx:
        n_cols_wi_write = wi_idx["world_enabled"]  # 0-based index => count of cols before it
    else:
        n_cols_wi_write = len(wi_headers)

    wi_rows: List[List[Any]] = []
    for item_id, _name in items:
        dc_params = dc_params_by_item.get(item_id, {})
        if not dc_params:
            continue

        for world in enabled_worlds:
            dc = world_to_dc_enabled[world]
            region = world_to_region_enabled[world]

            params = dc_params.get(dc)
            if params is None:
                continue
            alpha, beta = params

            # まず全列分のrowを作る（最後に部分書き込みで切る）
            row_full = [""] * len(wi_headers)
            row_full[wi_idx["item_id"]] = item_id
            row_full[wi_idx["world"]] = world
            row_full[wi_idx["sell_region"]] = region
            row_full[wi_idx["updated_at"]] = now_iso

            try:
                hist = uni.get_history(world, item_id, entries=history_entries_rot)
                k72, delta_h = core.k72_and_delta_h(hist, now_sec=now_sec)

                obs_eff = core.effective_window_hours(window_hours, delta_h)
                a_post, b_post = core.posterior_gamma_params(alpha, beta, k72, obs_eff)

                m = core.rotation_metrics_from_posterior(a_post, b_post)
                rot_score = core.rotation_score_preset3(m, preset_lambda=preset_lambda, preset_kappa=preset_kappa)

                row_full[wi_idx["k72"]] = int(k72)
                row_full[wi_idx["delta_h"]] = float(delta_h) if delta_h is not None else ""
                row_full[wi_idx["mu"]] = float(m["mu"])
                row_full[wi_idx["var"]] = float(m["var"])
                row_full[wi_idx["skew"]] = float(m["skew"])
                row_full[wi_idx["rot_score"]] = float(rot_score)

            except Exception:
                pass

            wi_rows.append(row_full)

    # 4) AB_PARAMS 書き戻し
    ab_headers = get_headers(ws_ab)
    ab_idx = header_index(ab_headers)

    required_ab = {"item_id", "dc", "alpha", "beta", "fitted_at", "n_worlds", "loglik", "success"}
    missing_ab = [c for c in required_ab if c not in ab_idx]
    if missing_ab:
        raise RuntimeError(f"AB_PARAMS missing headers: {missing_ab}")

    n_cols_ab = len(ab_headers)
    ab_rows: List[List[Any]] = []
    enabled_item_ids = {iid for iid, _ in items}

    for (iid, dc), abr in sorted(ab_new.items(), key=lambda x: (x[0][0], x[0][1])):
        if iid not in enabled_item_ids:
            continue
        if dc not in enabled_dcs:
            continue
        row = [""] * n_cols_ab
        row[ab_idx["item_id"]] = abr.item_id
        row[ab_idx["dc"]] = abr.dc
        row[ab_idx["alpha"]] = float(abr.alpha)
        row[ab_idx["beta"]] = float(abr.beta)
        row[ab_idx["fitted_at"]] = abr.fitted_at
        row[ab_idx["n_worlds"]] = int(abr.n_worlds)
        row[ab_idx["loglik"]] = float(abr.loglik)
        row[ab_idx["success"]] = bool(abr.success)
        ab_rows.append(row)

    # write
    bad = [(i, j, v) for i, row in enumerate(wi_rows) for j, v in enumerate(row) if _has_bad_float(v)]
    print("bad floats in WORLD_ITEM (first 20):", bad[:20])

    bad2 = [(i, j, v) for i, row in enumerate(ab_rows) for j, v in enumerate(row) if _has_bad_float(v)]
    print("bad floats in AB_PARAMS (first 20):", bad2[:20])

    bad3 = [(i, j, v) for i, row in enumerate(price_rows) for j, v in enumerate(row) if _has_bad_float(v)]
    print("bad floats in ITEM_PRICES (first 20):", bad3[:20])
    update_table_by_headers(ws_prices, prices_headers, price_rows)
    update_table_partial(ws_world_item, n_cols_wi_write, wi_rows)  # world_enabled列は触らない
    update_table_by_headers(ws_ab, ab_headers, ab_rows)

    print("DONE:", now_iso)

def _sanitize_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return v

def _has_bad_float(x):
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))



if __name__ == "__main__":
    main()
