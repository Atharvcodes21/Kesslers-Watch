export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Official Space-Track OBJECT_TYPE values. */
export type OfficialType = "PAYLOAD" | "ROCKET BODY" | "DEBRIS" | "UNKNOWN";

export interface CatalogEntry {
  norad: number;
  name: string;
  object_type: OfficialType;
  /** "space-track" = official type; "inferred" = guessed from name (CelesTrak fallback); "pasted" = user TLE. */
  type_source: "space-track" | "inferred" | "pasted";
  group: string;
  tle1: string;
  tle2: string;
  epoch: string;
  perigee_km: number;
  apogee_km: number;
}

export type ObjectType = "payload" | "rocket" | "debris" | "unknown";
export const OBJECT_TYPES: ObjectType[] = ["payload", "rocket", "debris", "unknown"];

const FROM_OFFICIAL: Record<OfficialType, ObjectType> = {
  PAYLOAD: "payload",
  "ROCKET BODY": "rocket",
  DEBRIS: "debris",
  UNKNOWN: "unknown",
};

export function objectType(entry: CatalogEntry): ObjectType {
  return FROM_OFFICIAL[entry.object_type] ?? "unknown";
}

export interface CatalogStatus {
  objects: number;
  by_type: Record<OfficialType, number>;
  catalog: { source: string; fetched_utc: string; type_source: "official" | "inferred"; note: string | null };
  spacetrack_configured: boolean;
  n2yo_configured: boolean;
  default_observer: { lat: number; lon: number; alt_m: number; name: string };
}

export interface AboveSat {
  norad: number;
  name: string;
  int_designator: string | null;
  launch_date: string | null;
  lat: number;
  lon: number;
  alt_km: number;
}

export interface AboveResponse {
  observer: { lat: number; lon: number; alt_m: number };
  radius_deg: number;
  count: number;
  satellites: AboveSat[];
  cached: boolean;
}

export interface VisualPass {
  start_utc: string;
  start_az_compass: string | null;
  max_utc: string;
  max_el: number;
  max_az_compass: string | null;
  end_utc: string;
  end_az_compass: string | null;
  mag: number | null;
  duration_s: number;
}

export interface PassesResponse {
  norad: number;
  name: string | null;
  count: number;
  passes: VisualPass[];
  cached: boolean;
}

/** Error carrying the server's user-facing `detail` message. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "Backend unreachable. Is the API running on port 8000?");
  }
  if (!res.ok) {
    let detail = `Request failed (HTTP ${res.status}).`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = "Invalid input.";
    } catch {
      /* keep generic message */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const fetchCatalog = (signal?: AbortSignal) => getJson<CatalogEntry[]>("/catalog", signal);
export const fetchStatus = (signal?: AbortSignal) => getJson<CatalogStatus>("/status", signal);
export const fetchObject = (norad: number) => getJson<CatalogEntry>(`/object/${norad}`);

export function fetchAbove(lat: number, lon: number, radius: number, altM?: number) {
  const q = new URLSearchParams({ lat: String(lat), lon: String(lon), radius: String(radius) });
  if (altM !== undefined) q.set("alt_m", String(Math.round(altM)));
  return getJson<AboveResponse>(`/above?${q}`);
}

export function fetchPasses(norad: number, lat: number, lon: number, altM?: number) {
  const q = new URLSearchParams({ lat: String(lat), lon: String(lon) });
  if (altM !== undefined) q.set("alt_m", String(Math.round(altM)));
  return getJson<PassesResponse>(`/passes/${norad}?${q}`);
}

// --------------------------------------------------------------------------- risk (build steps 3-4)

export type RiskLevel = "alert" | "watch" | "low";

export interface Conjunction {
  norad: number;
  name: string;
  object_type: OfficialType;
  tca_utc: string;
  miss_km: number;
  vrel_kms: number;
  /** Combined 1-sigma, ASSUMED from TLE ages (TLEs carry no covariance). */
  sigma_km: number;
  pc: number;
  alert: boolean;
  level: RiskLevel;
  primary_norad: number;
  /** AI risk (model estimate): calibrated probability that the FINAL collision risk is >= 1e-6. */
  ai_p_high?: number;
  /** The model's estimate of the final log10 collision probability. */
  ai_log10_risk?: number;
  /** False when the conjunction is outside the 2-7 day lead the model was trained on. */
  ai_in_domain?: boolean;
}

export interface ScreenStats {
  primary_norad: number;
  primary_name: string;
  window_start_utc: string;
  horizon_h: number;
  screen_km: number;
  hbr_m: number;
  objects_in_catalog: number;
  pairs_after_altitude_filter: number;
  coarse_pair_epochs: number;
  refined_candidates: number;
  co_located_excluded: number;
  conjunctions: number;
  alerts: number;
  seconds: number;
  pairs_per_sec: number | null;
  pair_epochs_per_sec: number | null;
}

export interface ManeuverOption {
  lead_h: number;
  dv_ms: number | null;
  direction: "prograde" | "retrograde" | "none" | null;
  new_miss_km: number | null;
  new_pc: number | null;
  baseline_miss_km: number;
  baseline_pc: number;
  sigma_km: number;
  target_pc: number;
  feasible: boolean;
}

export interface ReentryItem {
  norad: number;
  name: string;
  object_type: OfficialType;
  perigee_km: number;
  apogee_km: number;
  est_days: number;
  risk: "high" | "medium" | "low";
  tle_age_days: number;
  overdue: boolean;
  basis: string;
}

export interface SpaceWeather {
  f107: number;
  f107_time_utc: string;
  kp: number;
  kp_time_utc: string;
  ap: number | null;
  ap_basis: string;
  source: string;
  updated: string;
  stale: boolean;
}

export interface Metrics {
  objects_tracked: number;
  pairs_per_sec: number | null;
  median_tle_age_h: number | null;
  brier_score: number | null;
  last_screen: ScreenStats | null;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Backend unreachable. Is the API running on port 8000?");
  }
  if (!res.ok) {
    let detail = `Request failed (HTTP ${res.status}).`;
    try {
      const b = (await res.json()) as { detail?: unknown };
      if (typeof b.detail === "string") detail = b.detail;
      else if (Array.isArray(b.detail)) detail = "Invalid input.";
    } catch {
      /* keep generic message */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const screen = (req: { norad: number; horizon_h: number; screen_km: number; hbr_m: number }) =>
  postJson<Conjunction[]>("/screen", req);
export const planManeuver = (req: { norad: number; secondary: number; tca_utc: string }) =>
  postJson<ManeuverOption[]>("/maneuver", req);
export const fetchReentry = (limit = 12) => getJson<ReentryItem[]>(`/reentry?limit=${limit}`);
export const fetchSpaceWeather = () => getJson<SpaceWeather>("/spaceweather");
export const fetchMetrics = () => getJson<Metrics>("/metrics");

// --------------------------------------------------------------------------- AI risk model (build step 5)

export interface ScoreSet {
  n: number;
  n_high_risk: number;
  brier: number;
  roc_auc: number | null;
  avg_precision: number | null;
  precision: number;
  recall: number;
  f1: number;
  f2: number;
  false_alert_rate: number;
  false_alerts_per_1000: number;
  alerts: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  mse_log_risk: number;
  /** Official Kelvins MSE on high-risk events, predictions clipped at -6. */
  mse_high_risk: number | null;
  mse_high_risk_unclipped: number | null;
  /** Official Kelvins loss L = MSE_HR / F2 (lower is better); null if undefined. */
  official_score: number | null;
}

export type ModelName = "baseline" | "full" | "live";
export type EvalSet = "holdout" | "official_test";
export type Interval = { lo: number | null; hi: number | null };

export interface CalibrationBin {
  bin: string;
  n: number;
  mean_predicted: number;
  observed_rate: number;
  positives: number;
}

export interface ImportanceRow {
  feature: string;
  importance: number;
  std: number;
}

export interface ModelMetrics {
  generated_utc: string;
  seed: number;
  dataset: {
    name: string;
    url: string;
    paper: string;
    years: string;
    description: string;
    train_events: number;
    train_events_eligible: number;
    train_high_risk_events: number;
    official_test_events: number;
    official_test_high_risk_events: number;
    note: string;
  };
  task: { target: string; inputs: string; high_risk: string; eligibility: string };
  split: { train_events: number; holdout_events: number; holdout_high_risk: number; seed: number };
  models: Record<ModelName, string>;
  thresholds: { full: number; live: number };
  n_features: { full: number; live: number };
  feature_mapping: { feature: string; kelvins: string; live: string }[];
  results: Record<EvalSet, Record<ModelName, ScoreSet>>;
  intervals: Record<EvalSet, { vs_baseline: Record<"full" | "live", Record<string, Interval>> }>;
  calibration: Record<EvalSet, Record<"full" | "live", CalibrationBin[]>>;
  importance: Record<"full" | "live", ImportanceRow[]>;
  published: {
    latest_risk_baseline: { score: number; mse_hr: number; f2: number };
    winner_sesc: { score: number; mse_hr: number; f2: number };
    source: string;
  };
  verdict: Record<EvalSet, Record<"full" | "live", string[]>>;
}

export const fetchModelMetrics = () => getJson<ModelMetrics>("/model/metrics");

// --------------------------------------------------------------------------- avoidance planning (CAM)

export type BurnDirection = "+T" | "-T" | "+R" | "-R" | "+N" | "-N";

export interface AvoidCandidate {
  id: string;
  direction: BurnDirection;
  direction_label: string;
  dv_ms: number;
  orbits_before: number;
  burn_utc: string;
  lead_h: number;
  new_miss_km: number;
  new_pc: number;
  new_tca_utc: string;
  meets_target: boolean;
  along_track_shift_km: number;
  total_shift_km: number;
  rule_3dvt_km: number;
  /** "not_checked" unless this option was re-screened against the catalog. */
  constraint: "passed" | "rejected" | "not_checked";
  new_conjunctions?: { norad: number; name: string; miss_km: number; pc: number; tca_utc: string }[];
  propellant_kg: number;
}

export interface AvoidBrief {
  norad: number;
  name: string;
  object_type: OfficialType;
  type_source?: string;
}

export interface AvoidPath {
  start_utc: string;
  step_s: number;
  original_km: [number, number, number][];
  burned_km: [number, number, number][];
  burn_point_km: [number, number, number];
  tca_original_km: [number, number, number];
  tca_burned_km: [number, number, number];
  other_at_tca_km: [number, number, number];
  frame: string;
}

export interface AvoidResponse {
  possible: boolean;
  reason?: string;
  note?: string;
  primary: AvoidBrief;
  secondary: AvoidBrief;
  maneuvering?: AvoidBrief & { role: "primary" | "secondary" };
  other?: AvoidBrief;
  tca_utc?: string;
  baseline?: { miss_km: number; pc: number; sigma_km: number; vrel_kms: number };
  target_pc?: number;
  burn_times?: { orbits_before: number; burn_utc: string; lead_h: number; feasible: boolean; reason: string | null }[];
  candidates?: AvoidCandidate[];
  recommended?: AvoidCandidate | null;
  decision_deadline?: { burn_utc: string; lead_h: number; orbits_before: number; dv_ms: number; direction: BurnDirection; propellant_kg: number } | null;
  constraint?: {
    checked: number;
    passed: number;
    pairs_after_altitude_filter: number;
    co_located_excluded: number;
    window_start_utc: string;
    window_end_utc: string;
    existing_conjunctions: { norad: number; name: string; miss_km: number; pc: number }[];
  } | null;
  path?: AvoidPath | null;
  inputs?: { mass_kg: number; isp_s: number; mass_assumed: boolean; isp_assumed: boolean };
  assumptions: string[];
  timing_s?: { grid: number; constraint: number; total: number };
  cached?: boolean;
}

export const planAvoidance = (req: {
  primary: number;
  secondary: number;
  tca: string;
  mass_kg?: number;
  isp_s?: number;
  target_pc?: number;
}) => postJson<AvoidResponse>("/avoid", req);
