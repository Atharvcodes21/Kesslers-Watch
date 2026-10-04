# KesslerWatch: Space Debris Risk Prediction

Hackathon prototype (24h). Goal: a working, impressive demo, not production. Prefer simple + working over complete.

## What it does
1. Loads the full on-orbit catalog (Space-Track GP, official object types; CelesTrak fallback).
2. Propagates orbits with SGP4 and renders them on a 3D Earth.
3. Screens a selected asset vs. catalog for conjunctions; computes miss distance, TCA, Pc.
4. Recommends along-track delta-v to bring Pc below threshold.
5. Flags re-entry risk (low perigee + decay rate); ML model refines drag/decay.

## Stack
- frontend/: Next.js (App Router, TS), React Three Fiber + drei, Tailwind, Framer Motion, lucide-react, satellite.js (client-side SGP4 for animation)
- backend/: FastAPI, sgp4, numpy, scipy, scikit-learn (PyTorch only if time allows)

## Commands
- Backend: `cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000`
- Retrain the AI risk model: `cd backend && pip install -r requirements-train.txt && python ml.py train` (writes backend/models/; ~2 min)
- Frontend: `cd frontend && npm install && npm run dev` (http://localhost:3000)
- Frontend reads `NEXT_PUBLIC_API_URL` (default http://localhost:8000)

## Data sources
- Secrets live in `backend/.env` (gitignored): SPACETRACK_USER, SPACETRACK_PASS, N2YO_API_KEY. Never hardcode, log, or send them to the frontend; upstream errors are sanitized before reaching clients.
- Primary catalog: Space-Track GP class, all on-orbit objects:
  `https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/epoch/%3Enow-10/predicates/NORAD_CAT_ID,OBJECT_NAME,OBJECT_TYPE,EPOCH,DECAY_DATE,TLE_LINE1,TLE_LINE2/orderby/norad_cat_id%20asc/format/json`
  Login: POST `https://www.space-track.org/ajaxauth/login` (identity, password), cookie session.
  Rate rules (documented): <30 requests/min, <300/hour, GP at most once per hour; never one request per satellite. We refresh the full catalog at most every 4 h (cache `backend/cache/spacetrack_gp.json`), limit to 25/min and 250/h, and back off 15 min after a failure.
  Single object: `.../class/gp/norad_cat_id/{N}/predicates/.../format/json` (cached 4 h; misses 10 min).
- Use the official OBJECT_TYPE: PAYLOAD / ROCKET BODY / DEBRIS / UNKNOWN.
- Fallback (no credentials or Space-Track failure): CelesTrak `https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle`
  Groups: stations, active, cosmos-1408-debris, fengyun-1c-debris, iridium-33-debris, cosmos-2251-debris. Cache to `backend/cache/` for 2h; never fetch per request. Types are inferred from names (DEB / R/B) and marked "inferred".
  Single object fallback: `https://celestrak.org/NORAD/elements/gp.php?CATNR={N}&FORMAT=tle`.
- Ship a fallback snapshot `backend/cache/sample_tles.txt` so the demo works offline.
- N2YO REST v1 (`https://api.n2yo.com/rest/v1/satellite/...&apiKey=KEY`): `above` and `visualpasses` are limited to 100 transactions/hour each (we cap at 90). Cache: above 60 s, passes 30 min.
- Space weather: NOAA SWPC JSON (F10.7, Kp). Cache too.
- Space-Track CDMs: not used.
- AI risk model training data: ESA Kelvins Collision Avoidance Challenge CDMs (2015–2019), `backend/data/kelvins/kelvins_competition_data/` (train_data/train_data.csv; official test inputs test_data.csv + labels test_data_private.csv `true_risk`). Gitignored. https://kelvins.esa.int/collision-avoidance-challenge/ · paper arXiv:2008.03069.
  Kelvins task: target = `risk` (log10) of the event's last CDM; inputs only CDMs with time_to_tca ≥ 2 d; high-risk = final risk ≥ −6. Official score L = MSE_HR / F2, MSE on true-high-risk events with predictions clipped below at −6 (reproduces the published latest-risk baseline 0.694 exactly).

## API contract
- GET  /catalog → [{norad, name, object_type, type_source, group, tle1, tle2, epoch, perigee_km, apogee_km}]
  object_type ∈ PAYLOAD | ROCKET BODY | DEBRIS | UNKNOWN; type_source ∈ space-track (official) | inferred. Gzipped.
- GET  /status → {objects, by_type, catalog: {source, fetched_utc, type_source, note}, spacetrack_configured, n2yo_configured, default_observer}
- GET  /object/{norad} → catalog entry for any NORAD ID (cache → Space-Track → CelesTrak CATNR). 404 not found, 410 re-entered, 502 upstream failure.
- GET  /above?lat=15.3647&lon=75.1240&radius=70&alt_m=627 → {observer, radius_deg, count, satellites: [{norad, name, int_designator, launch_date, lat, lon, alt_km}], cached}. Default observer: Hubballi (Hubli), Karnataka. The frontend lets the user change it (browser geolocation, or city search via the keyless Open-Meteo geocoding API, GeoNames data, non-commercial use; called from the browser on submit only, results cached per session) and sends lat/lon/alt_m on every request. The choice is remembered in sessionStorage.
- GET  /passes/{norad}?lat&lon&alt_m&days=10&min_visibility=60 → {observer, norad, name, count, passes: [{start_utc, start_az_compass, max_utc, max_el, max_az_compass, end_utc, end_az_compass, mag, duration_s}], cached}
  N2YO endpoints return 503 if the key is missing, 429 at our hourly cap, 502 on upstream errors, each with a readable `detail`.
- POST /screen {norad, horizon_h=24 (≤72), screen_km=10 (≤50), hbr_m=20} → [{norad, name, object_type, tca_utc, miss_km, vrel_kms, sigma_km, pc, alert, level, primary_norad, ai_p_high, ai_log10_risk, ai_in_domain}], sorted by Pc.
  ai_* = live AI model estimate (label it "AI risk (model est.)"): P(final risk ≥ 1e-6) and estimated final log10 risk; ai_in_domain false when TCA is < 2 d or > 7 d away (outside training range).
  alert = Pc ≥ 1e-4; level ∈ alert | watch (Pc ≥ 1e-6) | low. Run stats (pairs/sec etc.) go to GET /metrics → last_screen.
  Implementation notes: a perigee/apogee prefilter (±screen_km + 100 km) skips objects whose shell cannot reach the asset; objects that stay within screen_km for the whole window (docked modules/vehicles, formation flyers) are excluded and counted as co_located_excluded, not reported as conjunctions. After the 1 s refinement, the miss distance is polished within ±1 s along the local relative motion.
- POST /maneuver {norad, secondary, tca_utc, hbr_m=20} → [{lead_h, dv_ms (+ prograde / − retrograde), direction, new_miss_km, new_pc, baseline_miss_km, baseline_pc, sigma_km, target_pc=1e-6, feasible}] for leads 0.5/1/2/6/12/24 h before TCA. Estimates.
- GET  /reentry?limit=25 → [{norad, name, object_type, perigee_km, apogee_km, est_days, risk, tle_age_days, overdue, basis}] (perigee < 400 km; heuristic, est.). UI shows overdue as "imminent (est.)".
- GET  /spaceweather → {f107, f107_time_utc, kp, kp_time_utc, ap, ap_basis, source, updated, stale} (NOAA SWPC, cached 30 min; Ap = mean of last 8 three-hourly ap)
- GET  /metrics → {objects_tracked, pairs_per_sec, median_tle_age_h, brier_score (full model, held-out events), last_screen}
- GET  /model/metrics → backend/models/metrics.json: dataset, split, thresholds, results {holdout, official_test} × {baseline, full, live} (brier, roc_auc, precision, recall, f1, f2, false_alert_rate, mse_log_risk, mse_high_risk, official_score, …), bootstrap intervals, calibration, importance, feature_mapping, verdict. 404 if untrained.

## Physics rules (do not deviate)
- Units: km, km/s, seconds, UTC. Convert to m/s only for display of delta-v.
- Screening: coarse vectorized pass (SatrecArray, 60s step, chunks of 2000) → refine candidates at 1s around coarse minimum.
- TLEs have no covariance. Assume isotropic sigma_km = 0.5 + 1.0 * |TLE age at TCA in days|; combined sigma = hypot(s1, s2). Label this as an assumption in UI.
- Pc = ncx2.cdf((R/sigma)^2, df=2, nc=(d/sigma)^2), R = hard-body radius in km.
- Maneuver: along-track drift approx. Δs ≈ 3·Δv·Δt; solve Δv for miss distance where Pc < threshold.
- Re-entry estimate is a heuristic (mean-motion derivative); label it as such.
- Never invent data. If something is estimated, the UI says "est."

## UI rules
- Full-bleed R3F canvas behind everything: Earth, atmosphere glow shader, debris as ONE InstancedMesh (never one mesh per object). Colors: debris vs. active vs. selected vs. alert.
- HUD panels: `backdrop-blur-xl bg-black/40 border border-white/10 text-white`, absolute positioned.
- Motion: subtle Y drift (≤4px, ~6s) only on idle panels; stop drift on hover/focus. Tilt on hover ≤6°. Respect prefers-reduced-motion.
- Top bar: object search (name/NORAD), catalog count, data freshness.
- Bottom dock: time scrubber with 1x / 10x / 100x / 1000x, play/pause, "now" button.
- Side deck: selected conjunction (Pc, miss km, TCA countdown, rel. velocity, sigma), maneuver options, space weather (F10.7, Kp/Ap).
- Model panel (separate): Brier score, pairs/sec, objects tracked, TLE age.
- Click an object → camera flies to it, shows orbit trail and stats.
- Monospace (JetBrains Mono / Geist Mono) for numbers, coordinates, timestamps. Sans for labels.
- Must stay at 30+ fps with the full catalog (~32k objects). SGP4 runs in a Web Worker (`public/workers/propagate.worker.js`, loads the vendored satellite.js UMD build copied by `scripts/vendor-satellite.mjs` before dev/build). Snapshots every 0.2–1 s; the shader extrapolates between them with two-body gravity.
- Earth rotates by GMST (satellite.js gstime) of the simulated time; day/night from the real sun direction at that time.

## Build order (stop and demo-check after each)
1. Backend /catalog + Earth + debris points rendering.
2. Time scrubber animating positions (satellite.js).
3. /screen + alert card.
4. /maneuver + /reentry + space weather.
5. ML drag/decay model + Brier score on historical data.
6. Polish: camera fly-to, shaders, motion.

## Conventions
- TypeScript strict, functional components, no class components.
- Keep backend in few files: main.py, orbit.py, risk.py, ml.py.
- No auth, no database. In-memory + file cache only.
- When unsure, ask before adding a dependency.
## Design reference
- design/designreference.html is the visual target. Match its look, colors, fonts, layout and feel when building React components. Port it into React; don't paste it in as-is.

## Design specifics (always follow, these override the reference and the UI rules above where they conflict)
Vibe: minimal, posh, editorial. Like a luxury observatory brand, not a gaming HUD. Lots of empty space, few elements, nothing cramped or cluttered.

Logo:
- Use frontend/public/logo.jpg as the KesslerWatch logo. No text-based logo.
- Its background is black: render it with mix-blend-mode: screen so it blends into the page. It must be a direct fixed child of <main> (not inside a glass panel or other stacking context), or the black box shows.
- Top-left, about 272px wide (20% smaller than the original hero size), crisp and balanced; smaller (~160px) on Model/About.
- Hover: a thin gold (#D9B77E) SVG orbit ring around the logo, slowly rotating (~12s per turn), plus a very subtle warm glow. Smooth ease in/out. Never put a CSS drop-shadow on the image itself (it glows the whole black rectangle); draw ring/glow on a hit area beneath the blended image.

Fonts (Google Fonts via next/font):
- Headings (every heading, including panel titles like "Screening"): Newsreader, weight 500–600. Italic allowed for subtitles.
- Body and labels: Hanken Grotesk, weight 500 (600 for emphasis). Sentence case.
- Numbers, coordinates, timestamps: Geist Mono, weight 500.
- Never use font weights below 400, and no other decorative or thin fonts.

Readability:
- Body text min 17px, labels min 14px, key numbers 28px+.
- Main text ivory #F2EBDD at full opacity. Muted text #B8B2A7 only for small secondary labels.

Palette (only these):
- Space black #05060A, ivory text #F2EBDD, muted text #B8B2A7
- Champagne gold #D9B77E (accents, active states)
- Payloads soft ice blue #A8C8E8, debris warm silver #CFC6B8, rocket bodies muted amber #C9A36A
- Alert rose #E0707A (only for high risk). No neon, no acid colors. Exception: the globe's blue atmosphere rim (and its barely visible aura).

Background: existing starfield stays. Faint procedural Milky Way band far behind, about 6% opacity, slow parallax. If using an image instead, store it locally and confirm its license allows reuse.

Globe:
- The main focus: as large as possible, centered, with only the top nav, time dock and screen edges reserved. Edge panels float over the scene when opened; the globe does not shrink for them.
- Strong blue atmosphere rim so it separates clearly from the background.
- The moving blue aura (ported from design/designreference.html) is only a very, very faint hint (~20% of the reference strength). Decorative until space-weather data drives it.
- Auto-rotates slowly; user can drag to rotate fully in any direction and zoom. Pause auto-rotate while the user interacts.
- Thin, faint orbital shell rings labeled LEO / MEO / GEO.
- Objects as small soft glowing points, colored by type; high-risk objects pulse in rose.
- Click an object: smooth camera fly-to, its orbit drawn as a thin gold line, details in side panel.
- During a conjunction: draw a thin line between the two objects at TCA.

Layout:
- Structure like the first version: logo + catalog/stats on the left, screening/alerts on the right, globe in the middle.
- Top nav: tabs center (Live, Model, About); catalog count and data freshness right.
- Nav hover: a very faint thin gold orbit ring behind the tab label (opacity ~15–20%), slowly rotating with a soft glow. Noticeable but never competing with the text.
- Side panels: compact pill tabs docked at the screen edges ("Catalog" left under the logo, "Screening" right). Clicking a pill expands its panel (max 340px, compact, no wasted space); clicking again collapses it. Collapsed by default so the globe is unobstructed. Selecting an object opens Catalog to show its details. Max 2 side panels visible.
- Panels: glass (backdrop-blur, bg-black/50, 1px border white/8%), soft radius, thin dividers.
- Model tab: Brier score, calibration chart, model metrics, training data source.
- About tab: problem, how it works (3 steps), data sources, assumptions labeled clearly.
- Motion: slow, soft, eased. One elegant intro fade on load. No bouncy or flashy effects. All motion (spins, drifts, fades, camera auto-rotate/fly) respects prefers-reduced-motion.

## Accuracy
- Never invent APIs, package names, versions, or data fields. If unsure, check docs or ask.
- After writing code, run it and confirm it works before saying it's done.
- Label every estimated or assumed value in code comments and UI ("est.").
- If something can't be done, say so instead of faking it with mock data.