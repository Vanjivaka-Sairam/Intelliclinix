
"use client";

import { useEffect, useRef, forwardRef, useImperativeHandle, useState } from "react";
import { CvatLikeViewer } from "@/lib/canvas/viewer";
import { NucleiData, Nucleus } from "@/lib/canvas/types";
import { Loader2, Maximize2, Minimize2 } from "lucide-react";
import { apiFetch } from "@/lib/api";

type CanvasViewerProps = {
    sourceUrl?: string; // Can be undefined initially
    maskUrl?: string;
    nucleiUrl?: string; // New: URL to nuclei_data.json
    overlayOpacity?: number;
    smoothImage?: boolean;
    className?: string;
};

export type CanvasViewerRef = {
    fit: () => void;
    rotate: (angle: number) => void;
    zoomIn: () => void;
    zoomOut: () => void;
};

type TooltipState = {
    visible: boolean;
    x: number;
    y: number;
    data: Nucleus | null;
};

const CanvasViewer = forwardRef<CanvasViewerRef, CanvasViewerProps>(
    ({ sourceUrl, maskUrl, nucleiUrl, overlayOpacity = 0.6, smoothImage = false, className }, ref) => {
        const containerRef = useRef<HTMLDivElement>(null);
        const viewerRef = useRef<CvatLikeViewer | null>(null);
        const [tooltip, setTooltip] = useState<TooltipState>({ visible: false, x: 0, y: 0, data: null });

        const [isFullscreen, setIsFullscreen] = useState(false);

        // Initialize viewer on mount
        useEffect(() => {
            if (!containerRef.current) return;

            const viewer = new CvatLikeViewer();
            viewerRef.current = viewer;
            containerRef.current.appendChild(viewer.html());

            // Initial Config
            viewer.configure({ backgroundColor: 'transparent', smoothImage });
            viewer.fitCanvas();

            // Setup Hover Handler
            viewer.onHover((nucleus, x, y) => {
                if (nucleus) {
                    setTooltip({ visible: true, x, y, data: nucleus });
                } else {
                    setTooltip(prev => ({ ...prev, visible: false }));
                }
            });

            // Listen for fullscreen changes
            const onFsChange = () => setIsFullscreen(!!document.fullscreenElement);
            document.addEventListener('fullscreenchange', onFsChange);

            return () => {
                document.removeEventListener('fullscreenchange', onFsChange);
                viewer.destroy();
                viewerRef.current = null;
            };
        }, []);

        // Update Configuration
        useEffect(() => {
            viewerRef.current?.configure({ smoothImage });
        }, [smoothImage]);

        // Handle Source Image
        useEffect(() => {
            if (sourceUrl && viewerRef.current) {
                viewerRef.current.setup({
                    url: sourceUrl,
                    filename: "image", // filename not strictly needed for display
                    width: 0, // setup will auto-detect
                    height: 0
                });
            }
        }, [sourceUrl]);

        // Handle Overlay
        useEffect(() => {
            if (viewerRef.current) {
                if (maskUrl) {
                    viewerRef.current.setOverlay({ url: maskUrl, opacity: overlayOpacity });
                } else {
                    viewerRef.current.setOverlay(null);
                }
            }
        }, [maskUrl, overlayOpacity]);

        // Handle Nuclei Data (New)
        useEffect(() => {
            console.log("CanvasViewer: nucleiUrl changed:", nucleiUrl);
            if (!nucleiUrl || !viewerRef.current) {
                viewerRef.current?.setNucleiData(null);
                return;
            }

            const loadNuclei = async () => {
                try {
                    console.log("CanvasViewer: Fetching nuclei from", nucleiUrl);
                    // Use native fetch because nucleiUrl is a Blob URL (blob:...)
                    // apiFetch would prepend API_BASE calling it invalid.
                    const response = await fetch(nucleiUrl);
                    const res = await response.json();

                    console.log("CanvasViewer: Fetched nuclei data:", res);
                    if (res && res.nuclei) {
                        viewerRef.current?.setNucleiData(res);
                    } else {
                        console.warn("CanvasViewer: No nuclei in response");
                    }
                } catch (e) {
                    console.error("Failed to load nuclei data", e);
                }
            };
            loadNuclei();

        }, [nucleiUrl]);

        // Expose controls via ref
        useImperativeHandle(ref, () => ({
            fit: () => viewerRef.current?.fit(),
            rotate: (angle) => viewerRef.current?.rotate(angle),
            zoomIn: () => {
                const current = viewerRef.current?.scale || 1;
                viewerRef.current?.setZoom(current * 1.2);
            },
            zoomOut: () => {
                const current = viewerRef.current?.scale || 1;
                viewerRef.current?.setZoom(current / 1.2);
            }
        }));

        const toggleFullscreen = () => {
            if (!document.fullscreenElement) {
                containerRef.current?.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        };

        return (
            <div
                ref={containerRef}
                className={`relative w-full h-full min-h-[400px] bg-cvat-bg-tertiary rounded-lg overflow-hidden border border-cvat-border group ${className}`}
            >
                {!sourceUrl && (
                    <div className="absolute inset-0 flex items-center justify-center text-cvat-text-secondary pointer-events-none">
                        <Loader2 className="h-6 w-6 animate-spin mr-2" />
                        Loading Canvas...
                    </div>
                )}

                {/* Tooltip Layer */}
                {tooltip.visible && tooltip.data && (
                    <div
                        className="fixed z-50 pointer-events-none bg-black/80 backdrop-blur text-white p-3 rounded shadow-lg border border-white/20 text-xs"
                        style={{
                            left: tooltip.x + 15,
                            top: tooltip.y + 15,
                            maxWidth: '200px'
                        }}
                    >
                        <div className="font-bold mb-1 border-b border-white/20 pb-1">
                            Nucleus #{tooltip.data.id}
                        </div>
                        <div className="space-y-1">
                            {Object.entries(tooltip.data.stats).map(([k, v]) => (
                                <div key={k} className="flex justify-between gap-4">
                                    <span className="opacity-70">{k}:</span>
                                    <span className="font-mono">{v}</span>
                                </div>
                            ))}
                            {/* Ratios can be calculated here if not in stats */}
                            <div className="mt-2 pt-1 border-t border-white/10 flex justify-between gap-4 text-emerald-400">
                                <span>Total Signals:</span>
                                <span>{Object.values(tooltip.data.stats).reduce((a, b) => a + b, 0)}</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Controls overlay */}
                <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                        onClick={toggleFullscreen}
                        className="p-2 bg-black/50 text-white rounded-md hover:bg-black/75 backdrop-blur-sm"
                        title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
                    >
                        {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
                    </button>
                </div>
            </div>
        );
    }
);

CanvasViewer.displayName = "CanvasViewer";

export default CanvasViewer;
