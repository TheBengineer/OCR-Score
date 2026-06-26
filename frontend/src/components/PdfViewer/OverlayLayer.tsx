import type { ReactNode } from "react";

export interface OverlayLayerProps {
  /** Original PDF page width in points (unscaled, at 72 DPI). */
  pageWidth: number;
  /** Original PDF page height in points (unscaled, at 72 DPI). */
  pageHeight: number;
  /** Current render scale factor (CSS pixels per PDF point). */
  scale: number;
  /** Current page rotation in degrees (0, 90, 180, 270). */
  rotation: number;
  /** SVG overlay elements rendered inside the layer. */
  children?: ReactNode;
}

/**
 * Absolutely-positioned overlay that sits on top of the PDF canvas.
 * Dimensions are swapped for 90°/270° rotation to match react-pdf's
 * rotated canvas. The overlay itself does not apply a CSS transform;
 * child elements receive the coordinate-mapping props and are responsible
 * for rotation-aware positioning (handled by OCR overlay components in
 * later tasks).
 */
export function OverlayLayer({
  pageWidth,
  pageHeight,
  scale,
  rotation,
  children,
}: OverlayLayerProps) {
  const isSideways = rotation % 180 === 90;
  const width = (isSideways ? pageHeight : pageWidth) * scale;
  const height = (isSideways ? pageWidth : pageHeight) * scale;

  if (width <= 0 || height <= 0) {
    return null;
  }

  return (
    <div
      className="pointer-events-none absolute left-0 top-0 overflow-hidden"
      style={{ width, height }}
    >
      {children}
    </div>
  );
}
