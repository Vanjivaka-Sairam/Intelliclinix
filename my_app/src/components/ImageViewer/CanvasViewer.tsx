
"use client";

import { useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import { CvatLikeViewer } from "@/lib/canvas/viewer";
import { Loader2, Maximize2 } from "lucide-react";

type CanvasViewerProps = {
    sourceUrl?: string; // Can be undefined initially
    maskUrl?: string;
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

const CanvasViewer = forwardRef<CanvasViewerRef, CanvasViewerProps>(
    ({ sourceUrl, maskUrl, overlayOpacity = 0.6, smoothImage = false, className }, ref) => {
        const containerRef = useRef<HTMLDivElement>(null);
        const viewerRef = useRef<CvatLikeViewer | null>(null);

        // Initialize viewer on mount
        useEffect(() => {
            if (!containerRef.current) return;

            const viewer = new CvatLikeViewer();
            viewerRef.current = viewer;
            containerRef.current.appendChild(viewer.html());

            // Initial Config
            viewer.configure({ backgroundColor: 'transparent', smoothImage });
            viewer.fitCanvas();

            return () => {
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

                {/* Controls overlay */}
                <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                        onClick={toggleFullscreen}
                        className="p-2 bg-black/50 text-white rounded-md hover:bg-black/75 backdrop-blur-sm"
                        title="Toggle Fullscreen"
                    >
                        <Maximize2 className="w-5 h-5" />
                    </button>
                </div>
            </div>
        );
    }
);

CanvasViewer.displayName = "CanvasViewer";

export default CanvasViewer;
