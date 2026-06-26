import { useState, useCallback } from "react";
import { Eye, EyeOff, ChevronDown, ChevronUp } from "lucide-react";

/* ── Status entries ─────────────────────────────────────────────────────── */

interface LegendEntry {
  fill: string;
  label: string;
  description: string;
  borderStyle: string;
}

const LEGEND_ENTRIES: LegendEntry[] = [
  {
    fill: "#16a34a",
    label: "Correct",
    description: "OCR text matches ground truth",
    borderStyle: "border-2 border-solid border-green-600",
  },
  {
    fill: "#dc2626",
    label: "Wrong",
    description: "OCR text differs from ground truth",
    borderStyle: "border-2 border-dashed border-red-600",
  },
  {
    fill: "#2563eb",
    label: "Missing",
    description: "In ground truth but not in OCR output",
    borderStyle: "border-2 border-dotted border-blue-600",
  },
  {
    fill: "#ea580c",
    label: "Extra",
    description: "In OCR output but not in ground truth",
    borderStyle: "border-2 border-dashed border-orange-600",
  },
];

/* ── Props ───────────────────────────────────────────────────────────────── */

export interface OverlayLegendProps {
  /** Current overlay opacity (0–1). */
  opacity: number;
  /** Called when opacity changes. */
  onOpacityChange: (opacity: number) => void;
  /** Whether the overlay is visible. */
  visible: boolean;
  /** Called when visibility toggles. */
  onVisibilityChange: (visible: boolean) => void;
}

/* ── Component ───────────────────────────────────────────────────────────── */

export function OverlayLegend({
  opacity,
  onOpacityChange,
  visible,
  onVisibilityChange,
}: OverlayLegendProps) {
  const [collapsed, setCollapsed] = useState(false);

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onOpacityChange(Number(e.target.value) / 100);
    },
    [onOpacityChange],
  );

  return (
    <div className="absolute right-2 top-2 z-40 w-56 rounded-lg border border-surface-200 bg-white/95 shadow-md backdrop-blur-sm">
      {/* ── Header ── */}
      <div className="flex items-center justify-between border-b border-surface-100 px-3 py-2">
        <button
          type="button"
          className="flex items-center gap-1.5 text-xs font-semibold text-surface-700"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? "Expand legend" : "Collapse legend"}
        >
          {collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          Legend
        </button>

        <button
          type="button"
          className="rounded p-0.5 text-surface-400 hover:text-surface-600"
          onClick={() => onVisibilityChange(!visible)}
          aria-label={visible ? "Hide overlay" : "Show overlay"}
        >
          {visible ? <Eye size={14} /> : <EyeOff size={14} />}
        </button>
      </div>

      {/* ── Body ── */}
      {!collapsed && (
        <div className="space-y-2 px-3 py-2">
          {/* Legend items */}
          <div className="space-y-1.5">
            {LEGEND_ENTRIES.map((entry) => (
              <div
                key={entry.label}
                className="flex items-start gap-2"
              >
                <span
                  className={`mt-0.5 inline-block h-3.5 w-3.5 shrink-0 rounded-sm ${entry.borderStyle}`}
                  style={{ backgroundColor: entry.fill, opacity: 0.6 }}
                  aria-hidden="true"
                />
                <div>
                  <p className="text-xs font-medium text-surface-700">
                    {entry.label}
                  </p>
                  <p className="text-[10px] leading-tight text-surface-400">
                    {entry.description}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Opacity slider */}
          <div className="border-t border-surface-100 pt-2">
            <label
              htmlFor="overlay-opacity"
              className="text-[10px] font-medium uppercase tracking-wider text-surface-400"
            >
              Opacity: {Math.round(opacity * 100)}%
            </label>
            <input
              id="overlay-opacity"
              type="range"
              min={0}
              max={100}
              value={Math.round(opacity * 100)}
              onChange={handleSliderChange}
              className="mt-1 w-full accent-blue-600"
              aria-label="Overlay opacity"
            />
          </div>
        </div>
      )}
    </div>
  );
}
