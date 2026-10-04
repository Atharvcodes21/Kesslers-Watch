"use client";

import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";
import type { CatalogEntry } from "@/lib/api";
import type { Vec3 } from "@/lib/orbit";
import { PALETTE } from "@/lib/theme";
import { CameraRig, HOME_DISTANCE, type Framing, type Insets } from "./CameraRig";
import { ConjunctionLine } from "./ConjunctionLine";
import { Earth } from "./Earth";
import { MilkyWay } from "./MilkyWay";
import { ObjectField, type FocusRoles, type LayerVisibility } from "./ObjectField";
import { SelectedOrbit } from "./SelectedOrbit";
import { ShellRings } from "./ShellRings";
import { Starfield } from "./Starfield";
import { TcaMarker } from "./TcaMarker";

/** Focus mode: highlight a few objects, fade the rest, draw their orbits and markers. */
export interface SceneFocus {
  roles: FocusRoles;
  orbits: { entry: CatalogEntry; color: string }[];
  /** Sim time (ms) the orbit paths start from. */
  at: number;
  /** Two objects to join with a line (conjunction). */
  line: [CatalogEntry, CatalogEntry] | null;
  /** Point of closest approach, marked with a pulsing ring. */
  tcaPoint: Vec3 | null;
  /** Where the camera glides to and holds; null = follow the selection as usual. */
  cameraPoint: Vec3 | null;
}

export interface SceneProps {
  catalog: CatalogEntry[];
  layers: LayerVisibility;
  selected: CatalogEntry | null;
  selectedAt: number;
  alerts: ReadonlySet<number>;
  onSelect: (norad: number) => void;
  onFps?: (fps: number) => void;
  insets: Insets;
  framing: Framing;
  focus: SceneFocus | null;
}

export default function Scene({
  catalog,
  layers,
  selected,
  selectedAt,
  alerts,
  onSelect,
  onFps,
  insets,
  framing,
  focus,
}: SceneProps) {
  return (
    <div className="fixed inset-0 cursor-grab touch-none active:cursor-grabbing">
      <Canvas
        camera={{ position: [0, HOME_DISTANCE * 0.28, HOME_DISTANCE * 0.96], fov: 42, near: 0.005, far: 1200 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
        aria-label="3D view of Earth and every tracked object in orbit"
      >
        <color attach="background" args={[PALETTE.space]} />
        <MilkyWay />
        <Starfield />
        <Suspense fallback={null}>
          <Earth />
        </Suspense>
        <ShellRings />
        <ObjectField
          catalog={catalog}
          layers={layers}
          selected={selected?.norad ?? null}
          alerts={alerts}
          onSelect={onSelect}
          focus={focus?.roles ?? null}
          onFps={onFps}
        />
        {focus ? (
          focus.orbits.map((o) => <SelectedOrbit key={o.entry.norad} entry={o.entry} at={focus.at} color={o.color} glow />)
        ) : (
          <SelectedOrbit entry={selected} at={selectedAt} />
        )}
        {focus?.line && <ConjunctionLine a={focus.line[0]} b={focus.line[1]} />}
        {focus?.tcaPoint && <TcaMarker point={focus.tcaPoint} />}
        <CameraRig
          target={selected}
          targetAt={selectedAt}
          insets={insets}
          framing={framing}
          focusPoint={focus?.cameraPoint ?? null}
        />
      </Canvas>
    </div>
  );
}
