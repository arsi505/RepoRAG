"use client";

import { PointerEvent, useRef } from "react";

import { RepoRagLogo } from "./reporag-logo";

export function RepositoryGraph() {
  const sceneRef = useRef<HTMLDivElement>(null);

  const moveScene = (event: PointerEvent<HTMLDivElement>) => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce), (pointer: coarse)").matches) return;
    const scene = sceneRef.current;
    if (!scene) return;
    const bounds = scene.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const horizontal = (event.clientX - bounds.left) / bounds.width - 0.5;
    const vertical = (event.clientY - bounds.top) / bounds.height - 0.5;
    scene.style.setProperty("--graph-rotate-x", `${(-vertical * 5.5).toFixed(2)}deg`);
    scene.style.setProperty("--graph-rotate-y", `${(horizontal * 7).toFixed(2)}deg`);
    scene.style.setProperty("--graph-light-x", `${((horizontal + 0.5) * 100).toFixed(1)}%`);
    scene.style.setProperty("--graph-light-y", `${((vertical + 0.5) * 100).toFixed(1)}%`);
  };

  const resetScene = () => {
    const scene = sceneRef.current;
    if (!scene) return;
    scene.style.setProperty("--graph-rotate-x", "0deg");
    scene.style.setProperty("--graph-rotate-y", "0deg");
    scene.style.setProperty("--graph-light-x", "50%");
    scene.style.setProperty("--graph-light-y", "45%");
  };

  return (
    <div
      ref={sceneRef}
      className="repository-graph"
      aria-hidden="true"
      onPointerMove={moveScene}
      onPointerLeave={resetScene}
      data-testid="repository-graph"
    >
      <div className="graph-ambient" />
      <div className="graph-world">
        <div className="graph-floor" />
        <svg className="graph-routes" viewBox="0 0 600 190" preserveAspectRatio="none">
          <defs>
            <linearGradient id="graph-route-cyan" x1="0" y1="0" x2="1" y2="0">
              <stop stopColor="#54D9F2" stopOpacity=".08" />
              <stop offset=".5" stopColor="#62B6FF" stopOpacity=".72" />
              <stop offset="1" stopColor="#927FFF" stopOpacity=".1" />
            </linearGradient>
          </defs>
          <path className="graph-route route-query" d="M108 95 C176 95 210 95 286 95" />
          <path className="graph-route route-one" d="M205 38 C235 54 251 67 292 87" />
          <path className="graph-route route-two" d="M205 151 C237 132 258 116 292 103" />
          <path className="graph-route route-three" d="M400 38 C371 55 348 67 319 87" />
          <path className="graph-route route-four" d="M400 151 C367 132 346 118 319 103" />
          <path className="graph-route route-answer" d="M318 95 C391 95 430 95 500 95" />
        </svg>

        <div className="graph-capsule graph-query"><span>Q</span>Repository query</div>
        <div className="graph-file graph-file-api"><span className="graph-file-icon">{`{ }`}</span><span>api</span></div>
        <div className="graph-file graph-file-docs"><span className="graph-file-icon">#</span><span>docs</span></div>
        <div className="graph-file graph-file-auth"><span className="graph-file-icon">◇</span><span>auth</span></div>
        <div className="graph-file graph-file-tests"><span className="graph-file-icon">✓</span><span>tests</span></div>

        <div className="graph-hub">
          <span className="graph-hub-ring" />
          <RepoRagLogo className="graph-hub-logo" />
          <span className="graph-hub-label">retrieve</span>
        </div>

        <div className="graph-capsule graph-answer"><span>A</span>Grounded answer</div>
        <span className="graph-particle particle-one" />
        <span className="graph-particle particle-two" />
        <span className="graph-particle particle-three" />
      </div>
    </div>
  );
}
