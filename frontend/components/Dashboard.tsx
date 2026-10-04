"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  OBJECT_TYPES,
  fetchCatalog,
  fetchMetrics,
  fetchObject,
  fetchStatus,
  objectType,
  screen,
  type CatalogEntry,
  type CatalogStatus,
  type Conjunction,
  type ObjectType,
  type ReentryItem,
} from "@/lib/api";
import { medianTleAgeHours, positionAt, toSatrec, type Vec3 } from "@/lib/orbit";
import { useObserver } from "@/lib/observer";
import { PALETTE } from "@/lib/theme";
import { simClock, simNow, useSimNow } from "@/lib/simClock";
import { AbovePanel } from "./hud/AbovePanel";
import { FocusCard } from "./hud/FocusCard";
import { Toast, type ToastMsg } from "./hud/Toast";
import { BrandLogo, LOGO_Y, logoHeight } from "./hud/BrandLogo";
import { PILL_GAP, PILL_HEIGHT } from "./hud/EdgePanel";
import { LeftPanel } from "./hud/LeftPanel";
import { RightPanel, type ScreenParams, type ScreenState } from "./hud/RightPanel";
import { TimeDock } from "./hud/TimeDock";
import { TopNav, type CatalogStatus as LoadStatus, type Tab } from "./hud/TopNav";
import type { Insets } from "./scene/CameraRig";
import type { LayerVisibility } from "./scene/ObjectField";
import type { SceneFocus } from "./scene/Scene";
import { AboutTab } from "./tabs/AboutTab";
import { ModelTab } from "./tabs/ModelTab";

const Scene = dynamic(() => import("./scene/Scene"), { ssr: false });

const EMPTY: CatalogEntry[] = [];

// Layout (px)
const EDGE = 24;
const NAV_H = 84;
const DOCK_H = 80;
const RIGHT_TOP = 112; // below the catalog count / freshness block
const LOGO_FULL = 272;
const LOGO_COMPACT = 160;
const LOGO_MOBILE = 120;
const NARROW_QUERY = "(max-width: 859px)";

type RightPanelId = "screening" | "above" | null;

type Focus =
  | { kind: "conjunction"; c: Conjunction; at: number; tcaPoint: Vec3 | null }
  | { kind: "reentry"; item: ReentryItem; at: number };

function formatAge(hours: number | null): string | null {
  if (hours === null) return null;
  return hours < 24 ? `${hours.toFixed(0)} h` : `${(hours / 24).toFixed(1)} d`;
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const update = () => setMatches(mq.matches);
    const first = setTimeout(update, 0);
    mq.addEventListener("change", update);
    return () => {
      clearTimeout(first);
      mq.removeEventListener("change", update);
    };
  }, [query]);
  return matches;
}

export function Dashboard() {
  const [catalog, setCatalog] = useState<CatalogEntry[]>(EMPTY);
  // Objects added on demand (NORAD lookup / pasted TLE); these win over catalog entries.
  const [extras, setExtras] = useState<CatalogEntry[]>(EMPTY);
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [info, setInfo] = useState<CatalogStatus | null>(null);
  const [fetchedAt, setFetchedAt] = useState<Date | null>(null);
  const [tab, setTab] = useState<Tab>("live");
  const [layers, setLayers] = useState<LayerVisibility>({ payload: true, rocket: true, debris: true, unknown: true });
  const [selection, setSelection] = useState<{ norad: number; at: number } | null>(null);
  // Edge panels start collapsed so the globe is unobstructed. One right-hand panel at a time.
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState<RightPanelId>(null);
  const [observer, setObserver] = useObserver();
  const [fps, setFps] = useState<number | null>(null);
  const [overview, setOverview] = useState(false);
  const [screenState, setScreenState] = useState<ScreenState>({ s: "idle" });
  const now = useSimNow(1000);
  const narrow = useMediaQuery(NARROW_QUERY);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchCatalog(ctrl.signal)
      .then((data) => {
        setCatalog(data);
        setFetchedAt(new Date());
        setStatus("ok");
        return fetchStatus(ctrl.signal).then(setInfo);
      })
      .catch((err: unknown) => {
        if (ctrl.signal.aborted) return;
        console.error(err);
        setStatus((s) => (s === "ok" ? s : "error"));
      });
    return () => ctrl.abort();
  }, []);

  const objects = useMemo(() => {
    if (extras.length === 0) return catalog;
    const extraIds = new Set(extras.map((e) => e.norad));
    return [...catalog.filter((c) => !extraIds.has(c.norad)), ...extras];
  }, [catalog, extras]);

  const byNorad = useMemo(() => new Map(objects.map((c) => [c.norad, c])), [objects]);
  const selected = selection ? (byNorad.get(selection.norad) ?? null) : null;

  const stats = useMemo(() => {
    const counts = Object.fromEntries(OBJECT_TYPES.map((t) => [t, 0])) as Record<ObjectType, number>;
    for (const c of objects) counts[objectType(c)]++;
    const ageH = fetchedAt ? medianTleAgeHours(objects.map((c) => c.epoch), fetchedAt) : null;
    return { counts, medianAge: formatAge(ageH) };
  }, [objects, fetchedAt]);

  /** Focus mode: one conjunction (asset + threat) or one re-entry object; everything else fades. */
  const [focus, setFocus] = useState<Focus | null>(null);
  const [toast, setToast] = useState<ToastMsg | null>(null);

  const select = useCallback((norad: number) => {
    setFocus(null); // picking anything else leaves focus mode
    setSelection({ norad, at: simNow() });
    setTab("live");
    setLeftOpen(true); // details live in the catalog panel
  }, []);

  const clearFocus = useCallback(() => {
    setFocus((f) => {
      if (f?.kind === "conjunction") simClock.now(); // we moved the clock to TCA; return to live time
      return null;
    });
    setSelection(null);
  }, []);

  const clearSelection = useCallback(() => {
    setSelection(null);
    setFocus(null);
  }, []);

  const addExtra = useCallback(
    (entry: CatalogEntry) => {
      setExtras((xs) => [...xs.filter((x) => x.norad !== entry.norad), entry]);
      select(entry.norad);
    },
    [select],
  );

  /** Make sure an object is loaded (fetch from /object if needed) without selecting it. */
  const ensureLoaded = useCallback(
    async (norad: number): Promise<CatalogEntry | null> => {
      const have = byNorad.get(norad);
      if (have) return have;
      try {
        const e = await fetchObject(norad);
        setExtras((xs) => [...xs.filter((x) => x.norad !== e.norad), e]);
        return e;
      } catch {
        return null;
      }
    },
    [byNorad],
  );

  /** Select by NORAD ID, fetching from /object (Space-Track, CelesTrak fallback) if not loaded. */
  const lookup = useCallback(
    async (norad: number) => {
      if (byNorad.has(norad)) {
        select(norad);
        return;
      }
      addExtra(await fetchObject(norad));
    },
    [byNorad, select, addExtra],
  );

  const runScreen = useCallback(async (norad: number, p: ScreenParams) => {
    setScreenState({ s: "running", norad });
    setFocus((f) => (f?.kind === "conjunction" ? null : f));
    try {
      const results = await screen({ norad, ...p });
      const stats = await fetchMetrics()
        .then((m) => m.last_screen)
        .catch(() => null);
      setScreenState({ s: "done", results, stats });
      const high = results.filter((c) => c.alert).length;
      if (high > 0) {
        const who = stats?.primary_name ?? `NORAD ${norad}`;
        setToast({ id: Date.now(), text: `${high} approach${high === 1 ? "" : "es"} above Pc 1e−4 for ${who}. Open Screening to inspect.` });
      }
    } catch (e) {
      setScreenState({ s: "error", msg: e instanceof ApiError ? e.message : "Screening failed." });
    }
  }, []);

  /** Conjunction focus: clock to TCA (paused), fade the rest, both orbits, TCA ring, camera glides to TCA. */
  const focusConjunction = useCallback(
    async (c: Conjunction) => {
      const [a, b] = await Promise.all([ensureLoaded(c.primary_norad), ensureLoaded(c.norad)]);
      const tca = Date.parse(c.tca_utc);
      let tcaPoint: Vec3 | null = null;
      const ra = a && toSatrec(a);
      const rb = b && toSatrec(b);
      const pa = ra && positionAt(ra, new Date(tca));
      const pb = rb && positionAt(rb, new Date(tca));
      if (pa && pb) tcaPoint = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2];
      simClock.setTime(tca);
      simClock.pause();
      setSelection(null);
      setFocus({ kind: "conjunction", c, at: tca, tcaPoint });
    },
    [ensureLoaded],
  );

  /** Re-entry focus: fade the rest, object in amber with its orbit; camera follows it. */
  const focusReentry = useCallback(
    async (item: ReentryItem) => {
      const e = await ensureLoaded(item.norad);
      if (!e) {
        setToast({ id: Date.now(), text: `Could not load NORAD ${item.norad}.` });
        return;
      }
      const at = simNow();
      setSelection({ norad: item.norad, at });
      setFocus({ kind: "reentry", item, at });
    },
    [ensureLoaded],
  );

  // Rose pulse in the normal view: high-risk secondaries from the last screen.
  const alerts = useMemo(() => {
    const s = new Set<number>();
    if (screenState.s === "done") for (const c of screenState.results) if (c.alert) s.add(c.norad);
    return s as ReadonlySet<number>;
  }, [screenState]);

  const sceneFocus = useMemo((): SceneFocus | null => {
    if (!focus) return null;
    if (focus.kind === "conjunction") {
      const a = byNorad.get(focus.c.primary_norad);
      const b = byNorad.get(focus.c.norad);
      return {
        roles: { primary: focus.c.primary_norad, threat: focus.c.norad },
        orbits: [
          ...(a ? [{ entry: a, color: PALETTE.ice }] : []),
          ...(b ? [{ entry: b, color: PALETTE.rose }] : []),
        ],
        at: focus.at,
        line: a && b ? [a, b] : null,
        tcaPoint: focus.tcaPoint,
        cameraPoint: focus.tcaPoint,
      };
    }
    const e = byNorad.get(focus.item.norad);
    return {
      roles: { reentry: focus.item.norad },
      orbits: e ? [{ entry: e, color: PALETTE.amber }] : [],
      at: focus.at,
      line: null,
      tcaPoint: null,
      cameraPoint: null,
    };
  }, [focus, byNorad]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") clearFocus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clearFocus]);

  const dismissToast = useCallback(() => setToast(null), []);

  const live = tab === "live";
  const logoWidth = narrow ? LOGO_MOBILE : live ? LOGO_FULL : LOGO_COMPACT;
  const belowLogo = Math.round(LOGO_Y + logoHeight(logoWidth) + 18);
  const rightTop = narrow ? belowLogo : RIGHT_TOP;
  const rightPill2 = rightTop + PILL_HEIGHT + PILL_GAP;
  const rightPanelTop = rightPill2 + PILL_HEIGHT + PILL_GAP;

  // The globe is the focus: only the nav, the time dock and the screen edges are reserved.
  // Edge panels float over the scene when opened, so the globe keeps its full size.
  const insets: Insets = useMemo(
    () => ({ left: EDGE, right: EDGE, top: NAV_H, bottom: live ? DOCK_H : EDGE }),
    [live],
  );

  const toggleRight = (id: Exclude<RightPanelId, null>) => setRightOpen((cur) => (cur === id ? null : id));

  return (
    <main className="intro-fade relative h-dvh w-full overflow-hidden bg-space">
      <Scene
        catalog={objects}
        layers={layers}
        selected={selected}
        selectedAt={selection?.at ?? 0}
        alerts={alerts}
        onSelect={select}
        onFps={setFps}
        insets={insets}
        framing={overview ? "overview" : "leo"}
        focus={sceneFocus}
      />

      {/* Dims the globe behind the reading tabs */}
      <div
        aria-hidden
        className={`pointer-events-none fixed inset-0 bg-space/75 transition-opacity duration-1000 ease-soft motion-reduce:transition-none ${
          live ? "opacity-0" : "opacity-100"
        }`}
      />

      {/* Must remain a direct child of <main> for mix-blend-mode to reach the canvas. */}
      <BrandLogo visibleWidth={logoWidth} />

      <TopNav
        tab={tab}
        onTab={setTab}
        status={status}
        total={objects.length}
        medianAge={stats.medianAge}
        fetchedAt={fetchedAt}
      />

      {live && (
        <div key="live" className="fade-in">
          <LeftPanel
            catalog={objects}
            counts={stats.counts}
            medianAge={stats.medianAge}
            source={info?.catalog.source ?? null}
            typesInferred={info?.catalog.type_source === "inferred"}
            fps={fps}
            layers={layers}
            onToggleLayer={(t) => setLayers((l) => ({ ...l, [t]: !l[t] }))}
            selected={selected}
            now={now}
            observer={observer}
            onSelect={select}
            onLookup={lookup}
            onAddPasted={addExtra}
            onClear={clearSelection}
            open={leftOpen}
            onToggle={() => setLeftOpen((o) => !o)}
            top={belowLogo}
          />
          <RightPanel
            asset={selected}
            byNorad={byNorad}
            screenState={screenState}
            onScreen={(n, p) => void runScreen(n, p)}
            active={focus?.kind === "conjunction" ? focus.c : null}
            onFocus={(c) => void focusConjunction(c)}
            activeReentry={focus?.kind === "reentry" ? focus.item.norad : null}
            onFocusReentry={(r) => void focusReentry(r)}
            onPick={lookup}
            now={now}
            open={rightOpen === "screening"}
            onToggle={() => toggleRight("screening")}
            top={rightTop}
            panelTop={rightPanelTop}
          />
          <AbovePanel
            observer={observer}
            onObserver={setObserver}
            byNorad={byNorad}
            onPick={lookup}
            open={rightOpen === "above"}
            onToggle={() => toggleRight("above")}
            top={rightPill2}
            panelTop={rightPanelTop}
          />
          <TimeDock style={{ left: insets.left, right: insets.right }} overview={overview} onOverview={setOverview} />
          {focus && (
            <FocusCard
              conjunction={focus.kind === "conjunction" ? focus.c : null}
              primary={focus.kind === "conjunction" ? (byNorad.get(focus.c.primary_norad) ?? null) : null}
              reentry={focus.kind === "reentry" ? focus.item : null}
              onClear={clearFocus}
            />
          )}
          <Toast toast={toast} onDone={dismissToast} />
        </div>
      )}

      {!live && (
        <div
          key={tab}
          className="fade-in fixed inset-x-0 bottom-0 z-10 overflow-y-auto overscroll-contain"
          style={{ top: Math.max(NAV_H, belowLogo) }}
        >
          <div className="mx-auto max-w-4xl px-6 pb-20 pt-2">
            {tab === "model" ? <ModelTab
                total={objects.length}
                medianAge={stats.medianAge}
                pairsPerSec={screenState.s === "done" ? (screenState.stats?.pairs_per_sec ?? null) : null}
              /> : <AboutTab />}
          </div>
        </div>
      )}
    </main>
  );
}
