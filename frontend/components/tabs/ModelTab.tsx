"use client";

import { ExternalLink } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import {
  ApiError,
  fetchModelMetrics,
  type CalibrationBin,
  type EvalSet,
  type ImportanceRow,
  type Kpis,
  type ModelMetrics,
  type ModelName,
  type ScoreSet,
} from "@/lib/api";
import { PALETTE } from "@/lib/theme";
import { DASH, Glass, Row } from "../hud/Glass";

const MODELS: ModelName[] = ["baseline", "full", "live"];
const MODEL_LABEL: Record<ModelName, string> = { baseline: "Baseline", full: "Full model", live: "Live model" };
const SET_LABEL: Record<EvalSet, string> = { official_test: "Official Kelvins test", holdout: "Our held-out events" };
const CURVE_COLOR = { full: PALETTE.gold, live: PALETTE.ice } as const;

type MetricRow = { key: keyof ScoreSet; label: string; better: "lower" | "higher" | null; digits: number; pct?: boolean };
const ROWS: MetricRow[] = [
  { key: "official_score", label: "Kelvins score L = MSE_HR ÷ F2", better: "lower", digits: 3 },
  { key: "mse_high_risk", label: "MSE on high-risk events (official)", better: "lower", digits: 3 },
  { key: "f2", label: "F2", better: "higher", digits: 3 },
  { key: "brier", label: "Brier score", better: "lower", digits: 4 },
  { key: "roc_auc", label: "ROC-AUC", better: "higher", digits: 3 },
  { key: "pr_auc", label: "PR-AUC (average precision)", better: "higher", digits: 3 },
  { key: "ece", label: "Calibration error (ECE)", better: "lower", digits: 4 },
  { key: "precision", label: "Precision", better: "higher", digits: 3 },
  { key: "recall", label: "Recall", better: "higher", digits: 3 },
  { key: "f1", label: "F1", better: "higher", digits: 3 },
  { key: "false_alert_rate", label: "False alert rate", better: "lower", digits: 1, pct: true },
  { key: "false_alerts_per_1000", label: "False alerts per 1,000 events", better: "lower", digits: 1 },
  { key: "mse_log_risk", label: "MSE on log risk, all events", better: "lower", digits: 2 },
];

const FEATURE_LABEL: Record<string, string> = {
  latest_risk: "Latest risk (log10)",
  log10_pc: "Collision probability (log10)",
  miss_over_sigma: "Miss distance ÷ uncertainty",
  sigma_km: "Position uncertainty",
  relative_speed_ms: "Relative speed",
  miss_distance_m: "Miss distance",
  time_to_tca_d: "Time to closest approach",
  object_type: "Object type",
  mission_id: "ESA mission (anonymised)",
  radial_sep_m: "Radial separation",
  max_risk_estimate_first: "Max-risk estimate, first CDM",
  risk_first: "Risk at first CDM",
  risk_min: "Lowest risk so far",
  risk_max: "Highest risk so far",
};

function fmt(v: number | null | undefined, digits: number, pct = false): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return DASH;
  return pct ? `${(v * 100).toFixed(digits)}%` : v.toFixed(digits);
}

function SetSwitch({ value, onChange }: { value: EvalSet; onChange: (v: EvalSet) => void }) {
  return (
    <div role="tablist" aria-label="Evaluation set" className="glass inline-flex gap-1 rounded-full p-1" style={{ borderRadius: 999 }}>
      {(["official_test", "holdout"] as EvalSet[]).map((k) => (
        <button
          key={k}
          role="tab"
          type="button"
          aria-selected={value === k}
          onClick={() => onChange(k)}
          className={`tab rounded-full px-4 py-1.5 text-[15px] transition-colors duration-500 ${
            value === k ? "bg-white/[0.08] font-semibold text-gold" : "text-ivory hover:text-gold"
          }`}
        >
          {SET_LABEL[k]}
        </button>
      ))}
    </div>
  );
}

function Headline({ m, ds }: { m: ModelMetrics; ds: EvalSet }) {
  const r = m.results[ds];
  const pub = m.published;
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Glass className="p-7">
        <div className="eyebrow">Kelvins score L (lower is better)</div>
        <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-2">
          {MODELS.map((k) => (
            <div key={k}>
              <div className={`font-mono text-[32px] leading-none ${k === "baseline" ? "text-ivory/70" : "text-ivory"}`}>{fmt(r[k].official_score, 3)}</div>
              <div className="mt-1 text-[14px] text-muted">{MODEL_LABEL[k]}</div>
            </div>
          ))}
        </div>
        {ds === "official_test" && (
          <p className="mt-4 text-[14px] leading-snug text-muted">
            Published on the same test set: latest-risk baseline {pub.latest_risk_baseline.score}, competition winner{" "}
            {pub.winner_sesc.score} (Uriot et al., 2020). Our baseline reproduces the published value, which checks the scoring.
          </p>
        )}
      </Glass>
      <Glass className="p-7">
        <div className="eyebrow">Brier score of the high-risk probability (lower is better)</div>
        <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-2">
          {MODELS.map((k) => (
            <div key={k}>
              <div className={`font-mono text-[32px] leading-none ${k === "baseline" ? "text-ivory/70" : "text-ivory"}`}>{fmt(r[k].brier, 4)}</div>
              <div className="mt-1 text-[14px] text-muted">{MODEL_LABEL[k]}</div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-[14px] leading-snug text-muted">
          The baseline gives a hard yes/no, so its Brier score equals its error rate. {r.baseline.n.toLocaleString()} events,{" "}
          {r.baseline.n_high_risk} of them high-risk.
        </p>
      </Glass>
    </div>
  );
}

function MetricsTable({ m, ds }: { m: ModelMetrics; ds: EvalSet }) {
  const r = m.results[ds];
  const pub = ds === "official_test" ? m.published : null;
  const pubVal = (which: "latest_risk_baseline" | "winner_sesc", key: keyof ScoreSet) => {
    if (!pub) return null;
    const p = pub[which];
    return key === "official_score" ? p.score : key === "mse_high_risk" ? p.mse_hr : key === "f2" ? p.f2 : null;
  };
  return (
    <Glass className="overflow-x-auto p-7">
      <h2 className="heading mb-1">Baseline vs models</h2>
      <p className="mb-4 text-[14px] text-muted">
        {SET_LABEL[ds]}: {r.baseline.n.toLocaleString()} events, {r.baseline.n_high_risk} high-risk (final risk ≥ 1e−6).
      </p>
      <table className="w-full min-w-[560px] text-left">
        <thead>
          <tr className="text-[14px] text-muted">
            <th className="pb-2 pr-4 font-medium">Metric</th>
            {MODELS.map((k) => (
              <th key={k} className="pb-2 pr-4 text-right font-medium">
                {MODEL_LABEL[k]}
              </th>
            ))}
            {pub && <th className="pb-2 text-right font-medium">Published baseline / winner</th>}
          </tr>
        </thead>
        <tbody className="font-mono text-[15px]">
          {ROWS.map((row) => (
            <tr key={row.key} className="divider">
              <td className="py-1.5 pr-4 font-sans text-[15px]">
                {row.label}
                {row.better && <span className="ml-1 text-[14px] text-muted">({row.better} is better)</span>}
              </td>
              {MODELS.map((k) => (
                <td key={k} className="py-1.5 pr-4 text-right">
                  {fmt(r[k][row.key] as number | null, row.digits, row.pct)}
                </td>
              ))}
              {pub && (
                <td className="py-1.5 text-right text-muted">
                  {pubVal("latest_risk_baseline", row.key) !== null
                    ? `${pubVal("latest_risk_baseline", row.key)} / ${pubVal("winner_sesc", row.key)}`
                    : ""}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-4 text-[14px] leading-snug text-muted">
        Official metric: MSE on log10 risk over truly high-risk events only, with predictions below −6 counted as −6, divided by
        F2. False alert rate = false alarms ÷ all low-risk events.
      </p>
    </Glass>
  );
}

function Verdicts({ m, ds }: { m: ModelMetrics; ds: EvalSet }) {
  const alt = (k: "full" | "live") => (m.results[ds][k] as ScoreSet & { alt_f2_threshold?: Record<string, number | null> }).alt_f2_threshold;
  return (
    <Glass className="p-7">
      <h2 className="heading mb-1">Does it beat the baseline?</h2>
      <p className="mb-4 text-[14px] text-muted">95% bootstrap intervals of each difference (resampling events). If the interval includes zero, we call it a tie.</p>
      <div className="grid gap-6 md:grid-cols-2">
        {(["full", "live"] as const).map((k) => (
          <div key={k}>
            <div className="mb-1.5 font-serif text-[19px] font-semibold">{MODEL_LABEL[k]}</div>
            <ul className="space-y-1.5">
              {m.verdict[ds][k].map((line) => (
                <li key={line} className="text-[15px] leading-snug">
                  {line}
                </li>
              ))}
            </ul>
            {alt(k) && (
              <p className="mt-2 text-[14px] leading-snug text-muted">
                At the alternative F2-optimal threshold ({fmt(alt(k)!.threshold, 3)}): recall {fmt(alt(k)!.recall, 3)}, F2{" "}
                {fmt(alt(k)!.f2, 3)}, L {fmt(alt(k)!.official_score, 3)}.
              </p>
            )}
          </div>
        ))}
      </div>
      <p className="mt-5 text-[14px] leading-snug text-muted">
        Alert thresholds were chosen on the validation split: the highest probability that still catches at least 95% of
        high-risk validation events (full {fmt(m.thresholds.full, 4)}, live {fmt(m.thresholds.live, 4)}). Missing a dangerous
        event costs far more than a false alarm, so this trades many more false alerts for recall, which also lowers F2 and
        raises the Kelvins score L compared with a threshold tuned for F2.
      </p>
    </Glass>
  );
}

/** Reliability diagram on square-root axes (most predictions are small). */
function CalibrationChart({ bins }: { bins: Record<"full" | "live", CalibrationBin[]> }) {
  const S = 300;
  const P = 36;
  const inner = S - P - 12;
  const sx = (v: number) => P + Math.sqrt(Math.max(0, Math.min(1, v))) * inner;
  const sy = (v: number) => 12 + inner - Math.sqrt(Math.max(0, Math.min(1, v))) * inner;
  const ticks = [0, 0.05, 0.2, 0.5, 1];
  return (
    <figure>
      <svg viewBox={`0 0 ${S} ${S}`} className="h-auto w-full max-w-[340px]" role="img" aria-label="Calibration chart: predicted probability versus observed frequency">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={P} x2={P + inner} y1={sy(t)} y2={sy(t)} stroke="rgb(255 255 255 / 0.06)" />
            <line x1={sx(t)} x2={sx(t)} y1={12} y2={12 + inner} stroke="rgb(255 255 255 / 0.06)" />
            <text x={sx(t)} y={S - 8} fill={PALETTE.muted} fontSize="12" textAnchor="middle" fontFamily="var(--font-geist-mono)">
              {t}
            </text>
            <text x={P - 6} y={sy(t) + 4} fill={PALETTE.muted} fontSize="12" textAnchor="end" fontFamily="var(--font-geist-mono)">
              {t}
            </text>
          </g>
        ))}
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke={PALETTE.ivory} strokeOpacity={0.35} strokeDasharray="3 5" />
        {(["full", "live"] as const).map((k) => {
          const pts = bins[k];
          return (
            <g key={k}>
              <polyline
                points={pts.map((b) => `${sx(b.mean_predicted)},${sy(b.observed_rate)}`).join(" ")}
                fill="none"
                stroke={CURVE_COLOR[k]}
                strokeOpacity={0.8}
                strokeWidth={1.5}
              />
              {pts.map((b) => (
                <circle
                  key={b.bin}
                  cx={sx(b.mean_predicted)}
                  cy={sy(b.observed_rate)}
                  r={Math.max(2.5, Math.min(7, Math.sqrt(b.n) / 3))}
                  fill={CURVE_COLOR[k]}
                  fillOpacity={0.85}
                >
                  <title>{`${MODEL_LABEL[k]} · ${b.bin}: predicted ${b.mean_predicted.toFixed(3)}, observed ${b.observed_rate.toFixed(3)} (${b.positives}/${b.n})`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[14px] text-muted">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: CURVE_COLOR.full }} /> Full model
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: CURVE_COLOR.live }} /> Live model
        </span>
        <span>Dashed: perfect calibration. Square-root axes; dot size ∝ events in the bin.</span>
      </figcaption>
    </figure>
  );
}

function ImportanceBars({ rows, title }: { rows: ImportanceRow[]; title: string }) {
  const max = Math.max(...rows.map((r) => r.importance), 1e-9);
  return (
    <div>
      <div className="mb-2 font-serif text-[19px] font-semibold">{title}</div>
      <ul className="space-y-1.5">
        {rows.map((r) => (
          <li key={r.feature}>
            <div className="flex items-baseline justify-between gap-3 text-[15px]">
              <span className="min-w-0 truncate" title={r.feature}>
                {FEATURE_LABEL[r.feature] ?? r.feature}
              </span>
              <span className="shrink-0 font-mono text-[14px] text-muted">{r.importance > 0 ? r.importance.toExponential(1) : "≈ 0"}</span>
            </div>
            <div className="mt-0.5 h-1.5 rounded-full bg-white/[0.06]">
              <div className="h-1.5 rounded-full bg-gold/70" style={{ width: `${Math.max(0, (r.importance / max) * 100)}%` }} />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

const pct = (v: number | null | undefined, d = 1) => (v === null || v === undefined || !Number.isFinite(v) ? DASH : `${(v * 100).toFixed(d)}%`);

function KpiTile({ title, children, foot }: { title: string; children: ReactNode; foot?: ReactNode }) {
  return (
    <Glass className="p-6">
      <div className="eyebrow">{title}</div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-2">{children}</div>
      {foot && <p className="mt-3 text-[14px] leading-snug text-muted">{foot}</p>}
    </Glass>
  );
}

function KpiNum({ value, label, dim = false }: { value: string; label: string; dim?: boolean }) {
  return (
    <div>
      <div className={`font-mono text-[30px] leading-none ${dim ? "text-ivory/70" : "text-ivory"}`}>{value}</div>
      <div className="mt-1 text-[14px] text-muted">{label}</div>
    </div>
  );
}

/** Hackfest PS 1C KPIs: Brier, recall on high-risk events, PR-AUC, lead time, throughput. All read from metrics.json. */
function KpiSection({ m, ds }: { m: ModelMetrics; ds: EvalSet }) {
  const k = m.kpis as Kpis;
  const r = m.results[ds];
  const op = k.operating_point.validation;
  const tp = k.throughput;
  return (
    <section className="space-y-6">
      <div>
        <h2 className="heading">Key results</h2>
        <p className="mt-1 text-[14px] text-muted">
          {SET_LABEL[ds]}, never used for training or for any choice. Alert thresholds were chosen on a separate validation split (
          {m.split.validation_events?.toLocaleString()} events, {m.split.validation_high_risk} high-risk) to catch at least{" "}
          {Math.round(k.operating_point.target_recall * 100)}% of high-risk events.
        </p>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <KpiTile title="Brier score (lower is better)" foot="Baseline: final risk = latest risk, a hard yes/no, so its Brier score is its error rate.">
          <KpiNum value={fmt(r.live.brier, 4)} label="Live model" />
          <KpiNum value={fmt(r.full.brier, 4)} label="Full model" />
          <KpiNum value={fmt(r.baseline.brier, 4)} label="Baseline" dim />
        </KpiTile>
        <KpiTile
          title="Recall on high-risk events"
          foot={
            <>
              At the operating threshold (live {pct(op.live.threshold, 2)}, full {pct(op.full.threshold, 2)}); validation recall live{" "}
              {pct(op.live.recall)}, full {pct(op.full.recall)}
              {!op.live.reached || !op.full.reached ? " (95% not reached)" : ""}. False-alert rate here: live {pct(r.live.false_alert_rate)}, full{" "}
              {pct(r.full.false_alert_rate)}, baseline {pct(r.baseline.false_alert_rate)}.
            </>
          }
        >
          <KpiNum value={pct(r.live.recall)} label={`Live (${r.live.tp} of ${r.live.n_high_risk})`} />
          <KpiNum value={pct(r.full.recall)} label={`Full (${r.full.tp} of ${r.full.n_high_risk})`} />
          <KpiNum value={pct(r.baseline.recall)} label="Baseline" dim />
        </KpiTile>
        <KpiTile title="PR-AUC (higher is better)" foot={`High-risk share of these events: ${pct(r.baseline.prevalence, 2)}, the PR-AUC of a random guess.`}>
          <KpiNum value={fmt(r.live.pr_auc, 3)} label="Live model" />
          <KpiNum value={fmt(r.full.pr_auc, 3)} label="Full model" />
          <KpiNum value={fmt(r.baseline.pr_auc, 3)} label="Baseline" dim />
        </KpiTile>
        <KpiTile
          title="Screening throughput (measured)"
          foot={
            tp ? (
              <>
                One object against {tp.catalog_objects.toLocaleString()} objects over {tp.horizon_h} h on one core; mean over {tp.per_object_seconds.n}{" "}
                objects {tp.per_object_seconds.mean.toFixed(1)} s. All-vs-all is an estimate: catalog size × mean time, serial. Measured{" "}
                {tp.measured_utc.slice(0, 10)}.
              </>
            ) : (
              "Not measured yet: run `python ml.py throughput` in backend/."
            )
          }
        >
          {tp?.reference && <KpiNum value={`${tp.reference.seconds.toFixed(1)} s`} label={`${tp.reference.name}, 24 h`} />}
          {tp && <KpiNum value={`${tp.all_vs_all_estimate.hours.toLocaleString()} h`} label="All-vs-all (est.)" dim />}
        </KpiTile>
      </div>
      <LeadTimeTable k={k} ds={ds} />
      <div className="grid gap-6 md:grid-cols-2">
        <ImbalancePanel k={k} />
        <RadialPanel k={k} />
      </div>
    </section>
  );
}

function LeadTimeTable({ k, ds }: { k: Kpis; ds: EvalSet }) {
  const rows = k.lead_time.rows[ds] ?? [];
  return (
    <Glass className="overflow-x-auto p-7">
      <h2 className="heading mb-1">Lead time</h2>
      <p className="mb-4 text-[14px] text-muted">
        Each method may only use warnings issued at least this many days before closest approach. Same thresholds throughout;
        events without such an early warning drop out.
      </p>
      <table className="w-full min-w-[620px] text-left">
        <thead>
          <tr className="text-[14px] text-muted">
            <th className="pb-2 pr-4 font-medium">Inputs from</th>
            <th className="pb-2 pr-4 text-right font-medium">Events (high-risk)</th>
            <th className="pb-2 pr-4 text-right font-medium">Recall: live / full / baseline</th>
            <th className="pb-2 text-right font-medium">Brier: live / full / baseline</th>
          </tr>
        </thead>
        <tbody className="font-mono text-[15px]">
          {rows.map((row) => (
            <tr key={row.days} className="divider">
              <td className="py-1.5 pr-4 font-sans">{row.label} before TCA</td>
              <td className="py-1.5 pr-4 text-right">
                {row.n.toLocaleString()} ({row.n_high_risk})
              </td>
              <td className="py-1.5 pr-4 text-right">
                {pct(row.live.recall, 0)} / {pct(row.full.recall, 0)} / <span className="text-ivory/70">{pct(row.baseline.recall, 0)}</span>
              </td>
              <td className="py-1.5 text-right">
                {fmt(row.live.brier, 4)} / {fmt(row.full.brier, 4)} / <span className="text-ivory/70">{fmt(row.baseline.brier, 4)}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.some((x) => x.n_high_risk < 10) && (
        <p className="mt-3 text-[14px] text-muted">Rows with fewer than 10 high-risk events are too small to read much into.</p>
      )}
    </Glass>
  );
}

const FS_LABEL: Record<string, string> = { full: "Full model", live_base: "Live, without radial", live_radial: "Live, with radial" };

function ImbalancePanel({ k }: { k: Kpis }) {
  const ci = k.class_imbalance;
  return (
    <Glass className="overflow-x-auto p-7">
      <h2 className="heading mb-1">Class imbalance</h2>
      <p className="mb-4 text-[14px] text-muted">
        About 1 in 100 events ends high-risk. Before: {ci.before}. Tried: {ci.tried}. Kept the variant with the {ci.selection}.
      </p>
      <table className="w-full min-w-[420px] text-left">
        <thead>
          <tr className="text-[14px] text-muted">
            <th className="pb-2 pr-3 font-medium">Validation Brier / ECE</th>
            <th className="pb-2 pr-3 text-right font-medium">Unweighted</th>
            <th className="pb-2 pr-3 text-right font-medium">Balanced + recalibrated</th>
            <th className="pb-2 text-right font-medium">Balanced, raw</th>
          </tr>
        </thead>
        <tbody className="font-mono text-[15px]">
          {Object.entries(ci.by_feature_set).map(([fs, v]) => (
            <tr key={fs} className="divider align-top">
              <td className="py-1.5 pr-3 font-sans text-[15px]">{FS_LABEL[fs] ?? fs}</td>
              {(["none", "balanced"] as const).map((key) => (
                <td key={key} className={`py-1.5 pr-3 text-right ${v.chosen === key ? "text-gold" : ""}`}>
                  {fmt(v[key].validation_brier, 4)}
                  <div className="text-[14px] text-muted">{fmt(v[key].validation_ece, 4)}</div>
                </td>
              ))}
              <td className="py-1.5 text-right text-ivory/70">
                {fmt(v.balanced_uncalibrated.validation_brier, 4)}
                <div className="text-[14px] text-muted">{fmt(v.balanced_uncalibrated.validation_ece, 4)}</div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-[14px] text-muted">Gold: kept. Raw weighted probabilities are inflated, which is why they are recalibrated.</p>
    </Glass>
  );
}

const RADIAL_ROWS = [
  ["validation", "brier", "Validation Brier", 4],
  ["holdout", "brier", "Held-out Brier", 4],
  ["holdout", "pr_auc", "Held-out PR-AUC", 3],
  ["holdout", "recall", "Held-out recall", 3],
  ["official_test", "brier", "Official-test Brier", 4],
  ["official_test", "pr_auc", "Official-test PR-AUC", 3],
  ["official_test", "recall", "Official-test recall", 3],
] as const;

function RadialPanel({ k }: { k: Kpis }) {
  const rf = k.radial_feature;
  return (
    <Glass className="overflow-x-auto p-7">
      <h2 className="heading mb-1">Radial separation as an AI input</h2>
      <p className="mb-4 text-[14px] text-muted">
        Live model retrained with and without the radial separation at closest approach; {rf.rule}. Result: {rf.used ? "kept" : "not kept"}.
      </p>
      <table className="w-full min-w-[380px] text-left">
        <thead>
          <tr className="text-[14px] text-muted">
            <th className="pb-2 pr-3 font-medium">Metric</th>
            <th className="pb-2 pr-3 text-right font-medium">Without</th>
            <th className="pb-2 text-right font-medium">With radial</th>
          </tr>
        </thead>
        <tbody className="font-mono text-[15px]">
          {RADIAL_ROWS.map(([set, key, label, digits]) => (
            <tr key={label} className="divider">
              <td className="py-1.5 pr-3 font-sans text-[15px]">{label}</td>
              {(["live_base", "live_radial"] as const).map((fs) => (
                <td key={fs} className={`py-1.5 pr-3 text-right ${(fs === "live_radial") === rf.used ? "text-gold" : ""}`}>
                  {fmt(rf.by_set[fs]?.[set]?.[key] as number | undefined, digits)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </Glass>
  );
}

export function ModelTab({ total, medianAge, pairsPerSec }: { total: number; medianAge: string | null; pairsPerSec: number | null }) {
  const [m, setM] = useState<ModelMetrics | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ds, setDs] = useState<EvalSet>("official_test");

  useEffect(() => {
    fetchModelMetrics()
      .then(setM)
      .catch((e) => setErr(e instanceof ApiError ? e.message : "Could not load the model report."));
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-serif text-[44px] font-semibold leading-tight">Model</h1>
        <p className="font-serif text-[20px] italic text-ivory">
          Learning, from real conjunction warnings, which close approaches end up dangerous.
        </p>
      </header>

      <Glass className="p-7">
        <h2 className="heading mb-3">What the AI does</h2>
        <div className="max-w-[68ch] space-y-3 text-[17px] text-ivory">
          <p>
            When two objects are predicted to pass close, operators receive a series of warnings (CDMs) over the following
            days, each with an updated collision probability. A manoeuvre has to be decided about two days before the
            closest approach, but the probability keeps changing until the end.
          </p>
          <p>
            The model looks at the warnings available two or more days out and estimates two things: the chance that the
            final collision probability ends up at or above 1 in a million, and what that final value will be. It learned
            this from about 9,000 real ESA conjunction events.
          </p>
          <p>
            The simple rule it has to beat is &ldquo;assume the latest number will not change&rdquo;. That rule is hard to beat, and
            the results below say plainly where the model does and does not do better. On this site the live model scores
            every screened approach as &ldquo;AI risk&rdquo;, a model estimate shown next to the physics Pc.
          </p>
        </div>
      </Glass>

      {err && (
        <Glass className="p-7">
          <p className="text-[17px] text-rose">{err}</p>
        </Glass>
      )}
      {!m && !err && <p className="text-[17px] text-muted">Loading the model report…</p>}

      {m && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <SetSwitch value={ds} onChange={setDs} />
            <span className="text-[14px] text-muted">Trained {m.generated_utc.slice(0, 10)} · seed {m.seed}</span>
          </div>
          {m.kpis && <KpiSection m={m} ds={ds} />}
          <Headline m={m} ds={ds} />
          <MetricsTable m={m} ds={ds} />
          <Verdicts m={m} ds={ds} />

          <div className="grid gap-6 md:grid-cols-2">
            <Glass className="p-7">
              <h2 className="heading mb-1">Calibration</h2>
              <p className="mb-4 text-[14px] text-muted">
                When the model says 20%, how often does the event really end high-risk? {SET_LABEL[ds]}.
              </p>
              <CalibrationChart bins={m.calibration[ds]} />
              {ds === "official_test" && (
                <p className="mt-3 text-[14px] leading-snug text-muted">
                  Points above the diagonal mean the model under-predicts here. The official test set was hand-picked with
                  about 7× more high-risk events than the training data, so probabilities calibrated on training run low on it.
                </p>
              )}
            </Glass>
            <Glass className="p-7">
              <h2 className="heading mb-1">What the model relies on</h2>
              <p className="mb-4 text-[14px] text-muted">
                Rise in Brier score when one input is shuffled (held-out events). Larger means the model leans on it more.
              </p>
              <div className="space-y-6">
                <ImportanceBars rows={m.importance.full.slice(0, 8)} title="Full model (top 8)" />
                <ImportanceBars rows={m.importance.live} title="Live model" />
              </div>
              <p className="mt-3 text-[14px] leading-snug text-muted">
                With only {m.dataset.train_high_risk_events} high-risk training events, most of the signal comes from the latest
                risk value itself; other inputs add little.
              </p>
            </Glass>
          </div>

          <Glass className="p-7">
            <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
              <h2 className="heading">Training data</h2>
              <a
                href={m.dataset.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-[15px] text-gold underline-offset-4 hover:underline"
              >
                ESA Kelvins challenge <ExternalLink className="h-3.5 w-3.5" strokeWidth={1.75} />
              </a>
            </div>
            <p className="max-w-[68ch] text-[17px] text-ivory">
              {m.dataset.name}, {m.dataset.years}. {m.dataset.description}
            </p>
            <div className="mt-4 grid gap-x-8 md:grid-cols-2">
              <div>
                <Row label="Training events (all)" value={m.dataset.train_events.toLocaleString()} />
                <Row label="Usable training events" value={m.dataset.train_events_eligible.toLocaleString()} />
                <Row label="…of which high-risk" value={m.dataset.train_high_risk_events} />
                <Row label="Held out for testing (by event)" value={`${m.split.holdout_events.toLocaleString()} (${m.split.holdout_high_risk} high-risk)`} />
              </div>
              <div>
                <Row label="Official test events" value={m.dataset.official_test_events.toLocaleString()} />
                <Row label="…of which high-risk" value={m.dataset.official_test_high_risk_events} />
                <Row label="Inputs" value={<span className="font-sans">CDMs ≥ 2 days before TCA</span>} />
                <Row label="High-risk" value={<span className="font-sans">final risk ≥ 1e−6</span>} />
              </div>
            </div>
            <p className="mt-3 text-[14px] leading-snug text-muted">
              {m.dataset.note} Paper:{" "}
              <a href={m.dataset.paper} target="_blank" rel="noreferrer" className="text-gold underline-offset-4 hover:underline">
                Uriot et al., 2020
              </a>
              .
            </p>

            <h3 className="mt-6 font-serif text-[19px] font-semibold">How the live model&apos;s inputs map to ESA&apos;s data</h3>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full min-w-[560px] text-left text-[15px]">
                <thead className="text-[14px] text-muted">
                  <tr>
                    <th className="pb-1.5 pr-4 font-medium">Input</th>
                    <th className="pb-1.5 pr-4 font-medium">From ESA CDMs (training)</th>
                    <th className="pb-1.5 font-medium">From our screener (live)</th>
                  </tr>
                </thead>
                <tbody>
                  {m.feature_mapping.map((f) => (
                    <tr key={f.feature} className="divider align-top">
                      <td className="py-1.5 pr-4">{FEATURE_LABEL[f.feature] ?? f.feature}</td>
                      <td className="py-1.5 pr-4 text-[14px]">{f.kelvins}</td>
                      <td className="py-1.5 text-[14px]">{f.live}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[14px] leading-snug text-muted">
              Caveat: ESA&apos;s probabilities use full covariances; ours use an assumed isotropic σ. Live AI risk is therefore an
              approximation, and the results above (measured on ESA data) are a best case for it.
            </p>
          </Glass>
        </>
      )}

      <Glass className="p-7">
        <h2 className="heading mb-3">Live system</h2>
        <Row label="Objects tracked" value={total > 0 ? total.toLocaleString() : DASH} />
        <Row label="Median TLE age" value={medianAge ?? DASH} />
        <Row label="Pairs screened per second (last screen)" value={pairsPerSec !== null ? pairsPerSec.toLocaleString() : "Run a screen"} />
      </Glass>
    </div>
  );
}
