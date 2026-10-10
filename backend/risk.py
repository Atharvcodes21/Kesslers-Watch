"""Conjunction screening, collision probability, avoidance-maneuver sizing and re-entry heuristic.

Follows CLAUDE.md "Physics rules": km, km/s, s, UTC; coarse vectorized SGP4 pass (SatrecArray,
60 s step, chunks of 2000) refined at 1 s; isotropic sigma from TLE age (an ASSUMPTION, TLEs carry
no covariance); Pc = ncx2.cdf((R/sigma)^2, df=2, nc=(d/sigma)^2); along-track drift Δs ≈ 3·Δv·Δt.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.stats import ncx2
from sgp4.api import Satrec, SatrecArray, jday

COARSE_STEP_S = 60
CHUNK = 2000
REFINE_HALF_S = 60  # refine at 1 s over ±60 s around each coarse minimum
ALT_MARGIN_KM = 100.0  # perigee/apogee prefilter margin (mean elements vs osculating radius)
VREL_MAX_KMS = 16.0  # upper bound on LEO relative speed, sets the coarse capture distance
LINEAR_MARGIN_KM = 50.0  # screen_paths: straight-line pre-check margin before the 1 s refinement
PC_ALERT = 1e-4  # "alert" (high risk)
PC_WATCH = 1e-6  # "watch"
PC_MANEUVER_TARGET = 1e-6  # maneuvers aim to bring Pc below this
MANEUVER_LEADS_H = (0.5, 1, 2, 6, 12, 24)
REENTRY_N_REVDAY = 16.4  # ~165 km circular altitude; below this, re-entry follows within hours
REENTRY_MAX_PERIGEE_KM = 400.0
REENTRY_MAX_DAYS = 365.0


# --------------------------------------------------------------------------- catalog index


@dataclass
class CatalogIndex:
    """Pre-parsed satrecs and arrays for fast screening; rebuilt when the catalog changes."""

    entries: list[dict]
    satrecs: list[Satrec]
    norad: np.ndarray
    perigee: np.ndarray
    apogee: np.ndarray
    epoch_jd: np.ndarray
    pos: dict[int, int]


def build_index(entries: list[dict]) -> CatalogIndex:
    keep, recs = [], []
    for e in entries:
        try:
            s = Satrec.twoline2rv(e["tle1"], e["tle2"])
        except Exception:
            continue
        keep.append(e)
        recs.append(s)
    return CatalogIndex(
        entries=keep,
        satrecs=recs,
        norad=np.array([e["norad"] for e in keep], dtype=np.int64),
        perigee=np.array([e["perigee_km"] for e in keep], dtype=float),
        apogee=np.array([e["apogee_km"] for e in keep], dtype=float),
        epoch_jd=np.array([s.jdsatepoch + s.jdsatepochF for s in recs], dtype=float),
        pos={e["norad"]: i for i, e in enumerate(keep)},
    )


# --------------------------------------------------------------------------- helpers


def _jd(t0: datetime, seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Julian date split (whole, fraction) for t0 + seconds, keeping full precision."""
    jd0, fr0 = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + t0.microsecond * 1e-6)
    fr = fr0 + np.asarray(seconds, dtype=float) / 86400.0
    whole = np.floor(fr)
    return jd0 + whole, fr - whole


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sigma_km(tle_age_days: float) -> float:
    """ASSUMED isotropic 1-sigma position error: 0.5 km + 1 km per day of TLE age at TCA."""
    return 0.5 + 1.0 * abs(tle_age_days)


def collision_probability(miss_km: float, sigma: float, hbr_km: float) -> float:
    """Pc = P(|x| < R) for a 2-D isotropic Gaussian miss vector: ncx2.cdf((R/s)^2, 2, (d/s)^2)."""
    return float(ncx2.cdf((hbr_km / sigma) ** 2, 2, (miss_km / sigma) ** 2))


def _risk_level(pc: float) -> str:
    return "alert" if pc >= PC_ALERT else "watch" if pc >= PC_WATCH else "low"


def _refine(primary: Satrec, secondary: Satrec, t0: datetime, center_s: float, lo_s: float, hi_s: float, perturb=None):
    """1 s grid over [center-60, center+60] ∩ [lo, hi], then sub-second polish along relative motion.

    `perturb(seconds_from_t0) -> (dr, dv)` (km, km/s) is added to the primary's SGP4 state when given
    (used for post-burn trajectories). Returns (tca_s, miss_km, rel, vrel, v_primary) or None if SGP4 failed.
    """
    tt = np.arange(max(lo_s, center_s - REFINE_HALF_S), min(hi_s, center_s + REFINE_HALF_S) + 1, 1.0)
    jd, fr = _jd(t0, tt)
    e1, r1, v1 = primary.sgp4_array(jd, fr)
    if perturb is not None:
        dr, dv = perturb(tt)
        r1, v1 = r1 + dr, v1 + dv
    e2, r2, v2 = secondary.sgp4_array(jd, fr)
    d = np.linalg.norm(r2 - r1, axis=1)
    d[(e1 != 0) | (e2 != 0)] = np.inf
    j = int(np.argmin(d))
    if not np.isfinite(d[j]):
        return None
    rel = r2[j] - r1[j]
    vrel = v2[j] - v1[j]
    # Closest approach of the locally linear relative motion, kept within ±1 s of the grid point.
    tau = float(np.clip(-np.dot(rel, vrel) / max(np.dot(vrel, vrel), 1e-12), -1.0, 1.0))
    rel_c = rel + vrel * tau
    return float(tt[j] + tau), float(np.linalg.norm(rel_c)), rel_c, vrel, v1[j]


# --------------------------------------------------------------------------- screening


def screen(
    index: CatalogIndex,
    primary_entry: dict,
    horizon_h: float = 24.0,
    screen_km: float = 10.0,
    hbr_m: float = 20.0,
    t0: datetime | None = None,
) -> tuple[list[dict], dict]:
    """Screen one asset against the whole catalog. Returns (conjunctions, stats)."""
    t_start = time.perf_counter()
    t0 = (t0 or datetime.now(timezone.utc)).replace(microsecond=0)
    horizon_s = horizon_h * 3600.0
    hbr_km = hbr_m / 1000.0
    pnorad = primary_entry["norad"]
    primary = Satrec.twoline2rv(primary_entry["tle1"], primary_entry["tle2"])
    p_epoch_jd = primary.jdsatepoch + primary.jdsatepochF

    times = np.arange(0.0, horizon_s + 1e-9, COARSE_STEP_S)
    jd, fr = _jd(t0, times)
    ep, rp, _ = primary.sgp4_array(jd, fr)
    if np.all(ep != 0):
        raise ValueError("SGP4 cannot propagate the protected asset over this window.")

    # Altitude-band prefilter: skip objects whose perigee/apogee shell cannot reach the asset's.
    p_lo = primary_entry["perigee_km"] - screen_km - ALT_MARGIN_KM
    p_hi = primary_entry["apogee_km"] + screen_km + ALT_MARGIN_KM
    mask = (index.apogee >= p_lo) & (index.perigee <= p_hi) & (index.norad != pnorad)
    candidates = np.nonzero(mask)[0]

    # Any pass closer than this at a 60 s sample could dip below screen_km in between.
    capture_km = screen_km + VREL_MAX_KMS * COARSE_STEP_S / 2
    minima: list[tuple[int, float]] = []
    pair_epochs = 0
    co_located: list[int] = []
    for start in range(0, len(candidates), CHUNK):
        idx = candidates[start : start + CHUNK]
        arr = SatrecArray([index.satrecs[i] for i in idx])
        e, r, _ = arr.sgp4(jd, fr)  # (n, nt), (n, nt, 3)
        d = np.linalg.norm(r - rp[None, :, :], axis=2)
        d[(e != 0) | (ep[None, :] != 0)] = np.inf
        pair_epochs += d.size
        # Objects that never leave the screening radius are attached to / flying with the asset
        # (e.g. ISS modules catalogued separately with the same elements): not conjunctions.
        dmax = np.where(np.isfinite(d), d, -np.inf).max(axis=1)
        attached = np.isfinite(dmax) & (dmax >= 0) & (dmax < screen_km)
        co_located += [int(index.norad[i]) for i in idx[attached]]
        d[attached] = np.inf
        interior = (d[:, 1:-1] <= d[:, :-2]) & (d[:, 1:-1] <= d[:, 2:]) & (d[:, 1:-1] < capture_km)
        rows, cols = np.nonzero(interior)
        minima += [(int(idx[a]), float(times[b + 1])) for a, b in zip(rows, cols)]
        for edge, nb in ((0, 1), (-1, -2)):
            hit = np.nonzero((d[:, edge] < capture_km) & (d[:, edge] <= d[:, nb]))[0]
            minima += [(int(idx[a]), float(times[edge])) for a in hit]
    t_coarse = time.perf_counter() - t_start

    events: dict[tuple[int, int], dict] = {}
    for i, center in minima:
        ref = _refine(primary, index.satrecs[i], t0, center, 0.0, horizon_s)
        if ref is None:
            continue
        tca_s, miss, rel, vrel, _ = ref
        if miss > screen_km:
            continue
        key = (i, int(round(tca_s / 300)))  # one event per object per 5-minute window
        if key in events and events[key]["miss_km"] <= miss:
            continue
        tca = t0 + timedelta(seconds=tca_s)
        jw, jf = _jd(t0, np.array([tca_s]))
        tca_jd = float(jw[0] + jf[0])
        s1 = sigma_km(tca_jd - p_epoch_jd)
        s2 = sigma_km(tca_jd - index.epoch_jd[i])
        sig = math.hypot(s1, s2)
        pc = collision_probability(miss, sig, hbr_km)
        sec = index.entries[i]
        events[key] = {
            "norad": sec["norad"],
            "name": sec["name"],
            "object_type": sec["object_type"],
            "tca_utc": _iso(tca),
            "miss_km": round(miss, 3),
            "vrel_kms": round(float(np.linalg.norm(vrel)), 3),
            "sigma_km": round(sig, 3),
            "pc": pc,
            "alert": pc >= PC_ALERT,
            "level": _risk_level(pc),
            "primary_norad": pnorad,
        }

    results = sorted(events.values(), key=lambda c: (-c["pc"], c["miss_km"]))
    elapsed = time.perf_counter() - t_start
    pairs = len(index.entries) - (1 if pnorad in index.pos else 0)
    stats = {
        "primary_norad": pnorad,
        "window_start_utc": _iso(t0),
        "horizon_h": horizon_h,
        "screen_km": screen_km,
        "hbr_m": hbr_m,
        "objects_in_catalog": pairs,
        "pairs_after_altitude_filter": int(len(candidates)),
        "coarse_pair_epochs": int(pair_epochs),
        "refined_candidates": len(minima),
        "co_located_excluded": len(co_located),
        "co_located_norads": co_located[:50],
        "conjunctions": len(results),
        "alerts": sum(1 for c in results if c["alert"]),
        "seconds": round(elapsed, 2),
        "coarse_seconds": round(t_coarse, 2),
        # Catalog pairs (asset vs every other object) resolved per second, prefilter included.
        "pairs_per_sec": round(pairs / elapsed) if elapsed > 0 else None,
        # Raw SGP4 pair-epoch throughput of the coarse pass.
        "pair_epochs_per_sec": round(pair_epochs / t_coarse) if t_coarse > 0 else None,
    }
    return results, stats


def screen_paths(
    index: CatalogIndex,
    primary_entry: dict,
    t0: datetime,
    horizon_s: float,
    perturbations: list,
    screen_km: float = 10.0,
    hbr_m: float = 20.0,
    skip: frozenset[int] = frozenset(),
) -> tuple[list[list[dict]], list[int], dict]:
    """Screen several versions of one asset's trajectory against the catalog in ONE SGP4 pass.

    perturbations[0] must be None (the asset as published); perturbations[k] for k >= 1 is a callable
    f(seconds_from_t0) -> (dr, dv) added to the asset's SGP4 state (e.g. a burn's effect, zero before the burn).
    The catalog is propagated once per chunk and compared with every path, so screening ~10 candidate
    burns costs little more than one screen. Same coarse/refine method and Pc/σ assumptions as `screen`.
    Objects co-located with the PUBLISHED path for the whole window (docked modules, attached vehicles)
    are excluded from every path: they move with the asset physically, but their TLEs would not.
    Returns (events per path, co-located NORAD IDs, stats).
    """
    assert perturbations and perturbations[0] is None
    t_start = time.perf_counter()
    hbr_km = hbr_m / 1000.0
    pnorad = primary_entry["norad"]
    primary = Satrec.twoline2rv(primary_entry["tle1"], primary_entry["tle2"])
    p_epoch_jd = primary.jdsatepoch + primary.jdsatepochF

    times = np.arange(0.0, horizon_s + 1e-9, COARSE_STEP_S)
    jd, fr = _jd(t0, times)
    ep, rp, vp = primary.sgp4_array(jd, fr)
    if np.all(ep != 0):
        raise ValueError("SGP4 cannot propagate the asset over this window.")
    paths, vpaths = [rp], [vp]
    for f in perturbations[1:]:
        dr, dv = f(times)
        paths.append(rp + dr)
        vpaths.append(vp + dv)

    # Altitude-band prefilter from the actual radii of all paths (burns change the shell).
    alts = np.concatenate([np.linalg.norm(p[ep == 0], axis=1) for p in paths]) - 6378.137
    p_lo, p_hi = alts.min() - screen_km - ALT_MARGIN_KM, alts.max() + screen_km + ALT_MARGIN_KM
    mask = (index.apogee >= p_lo) & (index.perigee <= p_hi) & (index.norad != pnorad)
    for n in skip:
        if n in index.pos:
            mask[index.pos[n]] = False
    candidates = np.nonzero(mask)[0]

    capture_km = screen_km + VREL_MAX_KMS * COARSE_STEP_S / 2
    minima: list[list[tuple[int, float]]] = [[] for _ in paths]
    co_located: list[int] = []
    linear_keep_km = screen_km + LINEAR_MARGIN_KM
    for start in range(0, len(candidates), CHUNK):
        idx = candidates[start : start + CHUNK]
        e, r, v = SatrecArray([index.satrecs[i] for i in idx]).sgp4(jd, fr)
        bad = (e != 0) | (ep[None, :] != 0)
        attached = None
        for k, pk in enumerate(paths):
            d = np.linalg.norm(r - pk[None, :, :], axis=2)
            d[bad] = np.inf
            if k == 0:
                dmax = np.where(np.isfinite(d), d, -np.inf).max(axis=1)
                attached = np.isfinite(dmax) & (dmax >= 0) & (dmax < screen_km)
                co_located += [int(index.norad[i]) for i in idx[attached]]
            d[attached] = np.inf
            interior = (d[:, 1:-1] <= d[:, :-2]) & (d[:, 1:-1] <= d[:, 2:]) & (d[:, 1:-1] < capture_km)
            rows, cols = np.nonzero(interior)
            cols = cols + 1
            for edge, nb in ((0, 1), (len(times) - 1, len(times) - 2)):
                hit = np.nonzero((d[:, edge] < capture_km) & (d[:, edge] <= d[:, nb]))[0]
                rows = np.concatenate([rows, hit])
                cols = np.concatenate([cols, np.full(len(hit), edge)])
            if len(rows) == 0:
                continue
            # Cheap straight-line closest approach within ±1 coarse step of the sample; only minima that
            # could come within screen_km (+ margin for curvature) get the full 1 s refinement.
            rel = r[rows, cols] - pk[cols]
            vrel = v[rows, cols] - vpaths[k][cols]
            tau = np.clip(-np.einsum("ij,ij->i", rel, vrel) / np.maximum(np.einsum("ij,ij->i", vrel, vrel), 1e-12), -COARSE_STEP_S, COARSE_STEP_S)
            dlin = np.linalg.norm(rel + vrel * tau[:, None], axis=1)
            keep = dlin < linear_keep_km
            minima[k] += [(int(idx[a]), float(times[b])) for a, b in zip(rows[keep], cols[keep])]
    t_coarse = time.perf_counter() - t_start

    out: list[list[dict]] = []
    for k, mins in enumerate(minima):
        f = perturbations[k]
        events: dict[tuple[int, int], dict] = {}
        for i, center in mins:
            ref = _refine(primary, index.satrecs[i], t0, center, 0.0, horizon_s, perturb=f)
            if ref is None:
                continue
            tca_s, miss, _, vrel, _ = ref
            if miss > screen_km:
                continue
            key = (i, int(round(tca_s / 300)))
            if key in events and events[key]["miss_km"] <= miss:
                continue
            jw, jf = _jd(t0, np.array([tca_s]))
            tca_jd = float(jw[0] + jf[0])
            sig = math.hypot(sigma_km(tca_jd - p_epoch_jd), sigma_km(tca_jd - index.epoch_jd[i]))
            sec = index.entries[i]
            events[key] = {
                "norad": sec["norad"],
                "name": sec["name"],
                "object_type": sec["object_type"],
                "tca_utc": _iso(t0 + timedelta(seconds=tca_s)),
                "miss_km": round(miss, 3),
                "vrel_kms": round(float(np.linalg.norm(vrel)), 3),
                "sigma_km": round(sig, 3),
                "pc": collision_probability(miss, sig, hbr_km),
            }
        out.append(sorted(events.values(), key=lambda c: (-c["pc"], c["miss_km"])))
    stats = {
        "window_start_utc": _iso(t0),
        "horizon_h": round(horizon_s / 3600.0, 2),
        "paths": len(paths),
        "objects_in_catalog": len(index.entries) - (1 if pnorad in index.pos else 0),
        "pairs_after_altitude_filter": int(len(candidates)),
        "co_located_excluded": len(co_located),
        "seconds": round(time.perf_counter() - t_start, 2),
        "coarse_seconds": round(t_coarse, 2),
    }
    return out, co_located, stats


# --------------------------------------------------------------------------- maneuver


def maneuver(primary_entry: dict, secondary_entry: dict, tca_utc: datetime, hbr_m: float = 20.0, now: datetime | None = None) -> list[dict]:
    """Along-track burn sizes per lead time that bring Pc below PC_MANEUVER_TARGET.

    Model (CLAUDE.md): a tangential burn Δv applied Δt before TCA shifts the asset along-track by
    Δs ≈ 3·Δv·Δt (prograde burns make it arrive later: Δs = −3·Δv·Δt). The new miss distance is the
    shifted relative position projected on the encounter plane (⊥ relative velocity). Estimates only.
    """
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    primary = Satrec.twoline2rv(primary_entry["tle1"], primary_entry["tle2"])
    secondary = Satrec.twoline2rv(secondary_entry["tle1"], secondary_entry["tle2"])
    ref = _refine(primary, secondary, tca_utc, 0.0, -REFINE_HALF_S, REFINE_HALF_S)
    if ref is None:
        raise ValueError("SGP4 failed near the given TCA.")
    tca_off, miss0, rel, vrel, vp = ref
    tca = tca_utc + timedelta(seconds=tca_off)
    tca_jd = sum(x[0] for x in _jd(tca, np.array([0.0])))
    sig = math.hypot(
        sigma_km(tca_jd - (primary.jdsatepoch + primary.jdsatepochF)),
        sigma_km(tca_jd - (secondary.jdsatepoch + secondary.jdsatepochF)),
    )
    hbr_km = hbr_m / 1000.0
    pc0 = collision_probability(miss0, sig, hbr_km)

    # Miss distance needed for Pc < target (Pc falls monotonically with distance): bisection.
    lo, hi = 0.0, 60.0 * sig + hbr_km
    for _ in range(80):
        mid = (lo + hi) / 2
        if collision_probability(mid, sig, hbr_km) > PC_MANEUVER_TARGET:
            lo = mid
        else:
            hi = mid
    d_star = hi * 1.02  # small margin

    vhat = vrel / np.linalg.norm(vrel)
    p = rel - np.dot(rel, vhat) * vhat  # miss vector in the encounter plane
    that = vp / np.linalg.norm(vp)
    q = that - np.dot(that, vhat) * vhat  # along-track direction seen in the encounter plane
    # Relative position after shifting the asset by Δs along-track: rel - Δs·t̂ → plane miss |p - Δs·q|.
    a, b, c = float(np.dot(q, q)), float(-2 * np.dot(p, q)), float(np.dot(p, p) - d_star**2)

    rows = []
    to_tca_h = (tca - now).total_seconds() / 3600.0
    for lead_h in MANEUVER_LEADS_H:
        if lead_h > to_tca_h:
            continue
        base = {"lead_h": lead_h, "baseline_miss_km": round(miss0, 3), "baseline_pc": pc0, "sigma_km": round(sig, 3), "target_pc": PC_MANEUVER_TARGET}
        if pc0 <= PC_MANEUVER_TARGET:
            rows.append({**base, "dv_ms": 0.0, "direction": "none", "new_miss_km": round(miss0, 3), "new_pc": pc0, "feasible": True})
            continue
        if a < 1e-8 or b * b - 4 * a * c < 0:
            rows.append({**base, "dv_ms": None, "direction": None, "new_miss_km": None, "new_pc": None, "feasible": False})
            continue
        roots = [(-b + sgn * math.sqrt(b * b - 4 * a * c)) / (2 * a) for sgn in (1, -1)]
        ds = min(roots, key=abs)  # km, along-track shift of the asset at TCA
        dt = lead_h * 3600.0
        dv_kms = -ds / (3 * dt)  # Δs = -3·Δv·Δt
        new_miss = float(np.linalg.norm(p - ds * q))
        rows.append(
            {
                **base,
                "dv_ms": round(dv_kms * 1000.0, 4),
                "direction": "prograde" if dv_kms > 0 else "retrograde",
                "new_miss_km": round(new_miss, 3),
                "new_pc": collision_probability(new_miss, sig, hbr_km),
                "feasible": True,
            }
        )
    return rows


# --------------------------------------------------------------------------- re-entry heuristic


def _ndot_half(tle1: str) -> float | None:
    """TLE line 1 cols 34-43: first derivative of mean motion / 2, rev/day^2."""
    try:
        return float(tle1[33:43].replace(" ", ""))
    except ValueError:
        return None


def reentry(entries: list[dict], now: datetime | None = None, limit: int = 25) -> list[dict]:
    """HEURISTIC (est.): days until mean motion reaches ~16.4 rev/day at the TLE's current decay rate.

    est_days = (16.4 - n) / (dn/dt) - (days since epoch), with dn/dt = 2 × TLE ndot/2 term.
    Decay accelerates as perigee drops, so this tends to OVERestimate the remaining time.
    """
    now = now or datetime.now(timezone.utc)
    out = []
    for e in entries:
        if e["perigee_km"] > REENTRY_MAX_PERIGEE_KM:
            continue
        half = _ndot_half(e["tle1"])
        try:
            n = float(e["tle2"][52:63])
        except ValueError:
            continue
        if half is None or half <= 0:
            continue
        age_d = (now - datetime.fromisoformat(e["epoch"].replace("Z", "+00:00"))).total_seconds() / 86400.0
        raw = (REENTRY_N_REVDAY - n) / (2 * half) - age_d
        est = max(0.0, raw)
        if est > REENTRY_MAX_DAYS:
            continue
        out.append(
            {
                "norad": e["norad"],
                "name": e["name"],
                "object_type": e["object_type"],
                "perigee_km": e["perigee_km"],
                "apogee_km": e["apogee_km"],
                "est_days": round(est, 1),
                "risk": "high" if est < 7 else "medium" if est < 30 else "low",
                "tle_age_days": round(age_d, 1),
                # Estimated date already passed since the TLE epoch: imminent or already re-entered.
                "overdue": raw < 0,
                "basis": "heuristic: mean-motion derivative (est.)",
            }
        )
    out.sort(key=lambda r: r["est_days"])
    return out[:limit]
