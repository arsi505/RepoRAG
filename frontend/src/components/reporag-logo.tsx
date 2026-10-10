import { useId } from "react";

interface RepoRagLogoProps {
  className?: string;
  labelled?: boolean;
}

export function RepoRagLogo({ className = "", labelled = false }: RepoRagLogoProps) {
  const rawId = useId();
  const id = rawId.replace(/:/g, "");
  const surfaceId = `reporag-surface-${id}`;
  const lineId = `reporag-line-${id}`;
  const glowId = `reporag-glow-${id}`;

  return (
    <svg
      className={className}
      viewBox="0 0 48 48"
      fill="none"
      role={labelled ? "img" : undefined}
      aria-hidden={labelled ? undefined : true}
      aria-label={labelled ? "RepoRAG" : undefined}
    >
      <defs>
        <linearGradient id={surfaceId} x1="7" y1="5" x2="42" y2="44" gradientUnits="userSpaceOnUse">
          <stop stopColor="#122B51" />
          <stop offset="0.52" stopColor="#0B1D37" />
          <stop offset="1" stopColor="#071426" />
        </linearGradient>
        <linearGradient id={lineId} x1="14" y1="11" x2="37" y2="38" gradientUnits="userSpaceOnUse">
          <stop stopColor="#8BE9FF" />
          <stop offset="0.52" stopColor="#54A5FF" />
          <stop offset="1" stopColor="#8A7DFF" />
        </linearGradient>
        <filter id={glowId} x="5" y="4" width="39" height="40" filterUnits="userSpaceOnUse">
          <feGaussianBlur stdDeviation="2.5" />
        </filter>
      </defs>
      <rect x="3.5" y="3.5" width="41" height="41" rx="12.5" fill={`url(#${surfaceId})`} />
      <rect x="3.5" y="3.5" width="41" height="41" rx="12.5" stroke="#73B9FF" strokeOpacity=".32" />
      <path
        d="M15 36V13H26.5C31.2 13 34 15.8 34 20.1C34 24.4 31.1 27.2 26.5 27.2H15M25.5 27.2L35.3 36"
        stroke="#397DFF"
        strokeOpacity=".35"
        strokeWidth="5"
        strokeLinecap="round"
        strokeLinejoin="round"
        filter={`url(#${glowId})`}
      />
      <path
        d="M15 36V13H26.5C31.2 13 34 15.8 34 20.1C34 24.4 31.1 27.2 26.5 27.2H15M25.5 27.2L35.3 36"
        stroke={`url(#${lineId})`}
        strokeWidth="2.35"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="15" cy="13" r="2.35" fill="#A8F2FF" />
      <circle cx="34" cy="20" r="2.35" fill="#62B6FF" />
      <circle cx="25.5" cy="27.2" r="2.35" fill="#6599FF" />
      <circle cx="35.3" cy="36" r="2.35" fill="#9B8CFF" />
    </svg>
  );
}
