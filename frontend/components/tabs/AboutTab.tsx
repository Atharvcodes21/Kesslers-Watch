import type { ReactNode } from "react";
import { Glass } from "../hud/Glass";

function Status({ kind }: { kind: "in-use" | "soon" | "assumption" }) {
  const map = {
    "in-use": { label: "In use", cls: "border-ice/30 text-ice" },
    soon: { label: "Coming soon", cls: "border-gold/30 text-gold/90" },
    assumption: { label: "Assumption", cls: "border-silver/30 text-silver" },
  }[kind];
  return <span className={`shrink-0 rounded-full border px-2.5 py-px text-[14px] ${map.cls}`}>{map.label}</span>;
}

function Item({ title, status, children }: { title: string; status: "in-use" | "soon" | "assumption"; children: ReactNode }) {
  return (
    <li className="divider flex items-start justify-between gap-6 py-4 first:border-t-0 first:pt-0">
      <div>
        <div className="text-[17px] font-semibold text-ivory">{title}</div>
        <div className="mt-1 text-[17px] text-ivory">{children}</div>
      </div>
      <Status kind={status} />
    </li>
  );
}

const STEPS = [
  {
    n: "I",
    title: "Gather",
    body: "The latest element sets for every tracked object still in orbit come from Space-Track, refreshed at most every four hours, with CelesTrak as a fallback.",
  },
  {
    n: "II",
    title: "Propagate",
    body: "Each object is flown forward with SGP4, the propagator the element sets were fitted with, in a background thread, and drawn on a globe turned to the simulated time.",
  },
  {
    n: "III",
    title: "Assess",
    body: "A chosen asset is screened against the catalog for close approaches; probability of collision and a small avoidance burn are estimated.",
  },
];

export function AboutTab() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-serif text-[44px] font-semibold leading-tight">About</h1>
        <p className="font-serif text-[20px] italic text-ivory">Why the crowded orbits matter, and how this watch is kept.</p>
      </header>

      <Glass className="p-8">
        <h2 className="heading mb-3">The problem</h2>
        <p className="max-w-[62ch] text-[17px] text-ivory">
          Low Earth orbit holds tens of thousands of tracked objects, most of them debris. At orbital speeds even a
          fragment the size of a coin can disable a spacecraft, and each collision creates more fragments. Two
          anti-satellite tests and one accidental collision, all shown here, produced some of the largest debris
          clouds on record. Operators need early, honest warning of close approaches and of objects about to fall.
        </p>
      </Glass>

      <Glass className="p-8">
        <h2 className="heading mb-5">How it works</h2>
        <ol className="grid gap-6 md:grid-cols-3">
          {STEPS.map((s) => (
            <li key={s.n}>
              <div className="font-serif text-[22px] font-semibold text-gold">{s.n}</div>
              <div className="mt-1 font-serif text-[22px] font-semibold">{s.title}</div>
              <p className="mt-1 text-[17px] text-ivory">{s.body}</p>
            </li>
          ))}
        </ol>
        <p className="mt-5 text-[14px] text-muted">Step III is in progress; screening and burn estimates arrive in the next build steps.</p>
      </Glass>

      <Glass className="p-8">
        <h2 className="heading mb-4">Data sources</h2>
        <ul>
          <Item title="Space-Track GP catalog" status="in-use">
            Primary source: the latest general perturbations element set for every on-orbit object (no decay date,
            epoch within 10 days), with the official object type. Fetched at most every 4 hours, within Space-Track&apos;s
            published rate limits. Single objects are looked up by NORAD ID on demand.
          </Item>
          <Item title="CelesTrak GP element sets" status="in-use">
            Fallback when Space-Track credentials are missing or a request fails: groups stations, active and four
            breakup clouds, cached for two hours, with a bundled offline snapshot. Also used for single-object lookups.
          </Item>
          <Item title="N2YO" status="in-use">
            Satellites above an observer and upcoming visible passes, cached on the server within N2YO&apos;s hourly
            limits. The API key never leaves the server.
          </Item>
          <Item title="ESA Kelvins conjunction data" status="in-use">
            About 13,000 real conjunction events (2015–2019) from ESA&apos;s Collision Avoidance Challenge, used to train and
            test the AI risk model. See the Model tab.
          </Item>
          <Item title="NOAA SWPC space weather" status="in-use">
            Solar flux F10.7 and geomagnetic Kp / Ap (cached 30 min), which drive upper-atmosphere density and drag.
          </Item>
          <Item title="Space-Track conjunction messages" status="soon">
            Not used yet; conjunctions will be screened from element sets.
          </Item>
        </ul>
      </Glass>

      <Glass className="p-8">
        <h2 className="heading mb-4">Assumptions</h2>
        <ul>
          <Item title="Position uncertainty" status="assumption">
            TLEs carry no covariance, so each object is given an isotropic 1σ of 0.5 km + 1 km per day of element-set
            age at closest approach; two objects combine as √(σ₁² + σ₂²).
          </Item>
          <Item title="Collision probability" status="assumption">
            Pc is the chance the miss vector falls inside the combined hard-body radius under that isotropic
            uncertainty (non-central χ² with two degrees of freedom).
          </Item>
          <Item title="Avoidance burns" status="assumption">
            Along-track drift is approximated as Δs ≈ 3·Δv·Δt; burn sizes are estimates (est.).
          </Item>
          <Item title="Re-entry dates" status="assumption">
            Lifetimes come from perigee height and the mean-motion derivative: a heuristic, labelled est., not a
            certified prediction.
          </Item>
          <Item title="Object type" status="assumption">
            Official Space-Track OBJECT_TYPE when available. With the CelesTrak fallback it is inferred from names
            (DEB = debris, R/B = rocket body, otherwise payload) and labelled &ldquo;inferred&rdquo;. Pasted TLEs are
            typed Unknown.
          </Item>
          <Item title="Smooth motion between updates" status="assumption">
            Positions are recomputed with SGP4 every 0.2–1 s; in between, each point is extrapolated with two-body
            gravity. The next SGP4 update corrects any drift.
          </Item>
        </ul>
      </Glass>
    </div>
  );
}
