import { useState, useCallback } from "react";
import { Layers, Eye, EyeOff, ChevronDown, ChevronUp } from "lucide-react";
import type { EngineLayerConfig } from "@/lib/types";

/* ── Props ───────────────────────────────────────────────────────────────── */

export interface EngineLayerManagerProps {
  layers: EngineLayerConfig[];
  onUpdateLayer: (id: string, partial: Partial<EngineLayerConfig>) => void;
  onShowAll: () => void;
  onHideAll: () => void;
}

/* ── Component ───────────────────────────────────────────────────────────── */

export function EngineLayerManager({
  layers,
  onUpdateLayer,
  onShowAll,
  onHideAll,
}: EngineLayerManagerProps) {
  const [collapsed, setCollapsed] = useState(false);

  const handleVisibilityToggle = useCallback(
    (id: string, currentVisible: boolean) => {
      onUpdateLayer(id, { visible: !currentVisible });
    },
    [onUpdateLayer],
  );

  const handleOpacityChange = useCallback(
    (id: string, value: number) => {
      onUpdateLayer(id, { opacity: value });
    },
    [onUpdateLayer],
  );

  const handleColorChange = useCallback(
    (id: string, color: string) => {
      onUpdateLayer(id, { color });
    },
    [onUpdateLayer],
  );

  const anyVisible = layers.some((l) => l.visible);
  const allVisible = layers.every((l) => l.visible);

  return (
    <div className="absolute bottom-2 left-2 z-40 w-64 rounded-lg border border-surface-200 bg-white/95 shadow-md backdrop-blur-sm">
      {/* ── Header ── */}
      <div className="flex items-center justify-between border-b border-surface-100 px-3 py-2">
        <button
          type="button"
          className="flex items-center gap-1.5 text-xs font-semibold text-surface-700"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? "Expand engine layers" : "Collapse engine layers"}
        >
          {collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          <Layers size={14} />
          Engine Layers
        </button>

        <div className="flex items-center gap-1">
          <button
            type="button"
            className="rounded px-1.5 py-0.5 text-[10px] font-medium text-surface-500 hover:bg-surface-100 hover:text-surface-700 disabled:opacity-30"
            onClick={onShowAll}
            disabled={allVisible}
            aria-label="Show all layers"
          >
            Show all
          </button>
          <span className="text-[10px] text-surface-300">·</span>
          <button
            type="button"
            className="rounded px-1.5 py-0.5 text-[10px] font-medium text-surface-500 hover:bg-surface-100 hover:text-surface-700 disabled:opacity-30"
            onClick={onHideAll}
            disabled={!anyVisible}
            aria-label="Hide all layers"
          >
            Hide all
          </button>
        </div>
      </div>

      {/* ── Body ── */}
      {!collapsed && (
        <div className="max-h-60 space-y-2 overflow-y-auto px-3 py-2">
          {layers.length === 0 && (
            <p className="py-2 text-center text-[11px] text-surface-400">
              No engine layers available.
            </p>
          )}

          {layers.map((layer) => (
            <div
              key={layer.id}
              className="rounded-md border border-surface-100 bg-surface-50/50 p-2"
            >
              {/* Layer header row */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-sm"
                    style={{ backgroundColor: layer.color }}
                    aria-hidden="true"
                  />
                  <span className="text-xs font-medium text-surface-700">
                    {layer.name}
                  </span>
                </div>

                <button
                  type="button"
                  className="rounded p-0.5 text-surface-400 hover:text-surface-600"
                  onClick={() =>
                    handleVisibilityToggle(layer.id, layer.visible)
                  }
                  aria-label={
                    layer.visible
                      ? `Hide ${layer.name} layer`
                      : `Show ${layer.name} layer`
                  }
                >
                  {layer.visible ? <Eye size={13} /> : <EyeOff size={13} />}
                </button>
              </div>

              {/* Controls row */}
              {layer.visible && (
                <div className="mt-1.5 flex items-center gap-2">
                  {/* Opacity slider */}
                  <div className="flex flex-1 items-center gap-1">
                    <label
                      className="text-[9px] uppercase tracking-wider text-surface-400"
                      htmlFor={`opacity-${layer.id}`}
                    >
                      Op
                    </label>
                    <input
                      id={`opacity-${layer.id}`}
                      type="range"
                      min={0.05}
                      max={1}
                      step={0.05}
                      value={layer.opacity}
                      onChange={(e) =>
                        handleOpacityChange(layer.id, Number(e.target.value))
                      }
                      className="h-1 w-full accent-current"
                      style={{
                        accentColor: layer.color,
                      }}
                      aria-label={`${layer.name} opacity`}
                    />
                    <span className="min-w-[2rem] text-right text-[10px] tabular-nums text-surface-400">
                      {Math.round(layer.opacity * 100)}%
                    </span>
                  </div>

                  {/* Color picker */}
                  <input
                    type="color"
                    value={layer.color}
                    onChange={(e) =>
                      handleColorChange(layer.id, e.target.value)
                    }
                    className="h-5 w-5 cursor-pointer rounded border border-surface-200 p-0"
                    aria-label={`${layer.name} color`}
                    title={`Color: ${layer.color}`}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
