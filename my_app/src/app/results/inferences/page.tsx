"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "react-hot-toast";
import {
    Loader2, Download, Send,
    ZoomIn, ZoomOut, RotateCw,
    Maximize2, ChevronLeft, ChevronRight, ChevronDown, ChevronUp,
    Layers, Layout, Table
} from "lucide-react";
import { apiFetch } from "@/lib/api";
import { useAuthGuard } from "@/hooks/use-auth-guard";
import CanvasViewer, { CanvasViewerRef } from "@/components/ImageViewer/CanvasViewer";
import { CsvViewer } from "@/components/CsvViewer";
import { ImageGallery, ImageItem } from "@/components/ImageGallery";
import Papa from "papaparse";

// Types matching backend payload
type InferenceResult = {
    source_filename: string;
    source_image_gridfs_id?: string;
    // Legacy fields for backward compatibility
    class_mask_id?: string;
    instance_mask_id?: string;
    // New generic artifacts list
    artifacts?: {
        kind: string;
        gridfs_id: string;
        filename: string;
    }[];
};

type InferenceResponse = {
    _id: string;
    dataset_id: string;
    status: string;
    results: InferenceResult[];
};

// Frontend state for an active image
type ActiveImageState = {
    sourceUrl?: string;
    layers: {
        classMask?: string;
        instanceMask?: string;
        stitchedOverlay?: string;
    };
    comparisonUrl?: string;
    nucleiUrl?: string;
};

export default function InferenceDetailPage() {
    const searchParams = useSearchParams();
    const inferenceId = searchParams.get("id");
    const { isLoading } = useAuthGuard();

    // Data State
    const [inference, setInference] = useState<InferenceResponse | null>(null);
    const [isPolling, setIsPolling] = useState(true);
    const [globalCsvData, setGlobalCsvData] = useState<any[]>([]);

    // UI State
    const [activeImageIndex, setActiveImageIndex] = useState<number>(0);
    const [activeImageState, setActiveImageState] = useState<ActiveImageState>({ layers: {} });
    const [isLoadingImage, setIsLoadingImage] = useState(false);

    // Viewer Configuration
    const [overlayOpacity, setOverlayOpacity] = useState(0.6);
    const [activeLayer, setActiveLayer] = useState<"none" | "classMask" | "instanceMask" | "stitchedOverlay">("stitchedOverlay");
    const [smoothImage, setSmoothImage] = useState(false);
    const [showCsvPanel, setShowCsvPanel] = useState(true);
    const [isCsvExpanded, setIsCsvExpanded] = useState(false);
    const [isGalleryCollapsed, setIsGalleryCollapsed] = useState(false);
    const [showComparison, setShowComparison] = useState(false);

    // Selection for CVAT
    const [selectedFilenames, setSelectedFilenames] = useState<string[]>([]);
    const [isSending, setIsSending] = useState(false);
    const [cvatLink, setCvatLink] = useState<string | null>(null);

    const pollRef = useRef<NodeJS.Timeout | null>(null);
    const viewerRef = useRef<CanvasViewerRef>(null);

    // 1. Fetch Inference Data
    useEffect(() => {
        if (isLoading || !inferenceId) return;

        const fetchInference = async () => {
            try {
                const response = await apiFetch(`/api/inferences/${inferenceId}`);
                if (!response.ok) throw new Error("Failed to load inference");

                const data: InferenceResponse = await response.json();
                setInference(data);

                if (data.status === "completed") {
                    setIsPolling(false);
                    // Initial load of global CSV if present
                    const csvResult = data.results.find(r =>
                        r.artifacts?.some(a => a.kind === "csv_global")
                    );
                    if (csvResult) {
                        const art = csvResult.artifacts!.find(a => a.kind === "csv_global");
                        fetchGlobalCsv(art!.gridfs_id);
                    } else {
                        setShowCsvPanel(false); // Hide panel if no CSV likely
                    }
                } else {
                    setIsPolling(true);
                }
            } catch (error) {
                console.error(error);
                toast.error("Unable to load inference");
            }
        };

        fetchInference();
        if (isPolling) pollRef.current = setInterval(fetchInference, 5000);

        return () => {
            if (pollRef.current) clearInterval(pollRef.current);
        };
    }, [inferenceId, isLoading, isPolling]);


    // 2. Helper to fetch file URL
    const fetchFileUrl = async (gridfsId: string): Promise<string | undefined> => {
        try {
            const response = await apiFetch(`/api/files/${gridfsId}`);
            if (!response.ok) throw new Error("Failed");
            const blob = await response.blob();
            return URL.createObjectURL(blob);
        } catch (e) {
            console.error(`Failed to fetch file ${gridfsId}`, e);
            return undefined;
        }
    };

    // 3. Fetch Global CSV Content
    const fetchGlobalCsv = async (gridfsId: string) => {
        try {
            const response = await apiFetch(`/api/files/${gridfsId}`);
            const text = await response.text();
            Papa.parse(text, {
                header: true,
                skipEmptyLines: true,
                complete: (res) => setGlobalCsvData(res.data)
            });
        } catch (e) {
            console.error("Failed to load global CSV", e);
        }
    };

    // 4. Computed: Navigable Images (exclude global_detections dummy entry)
    const navigableImages = useMemo(() => {
        if (!inference?.results) return [];
        return inference.results.filter(r =>
            r.source_filename !== "global_detections.csv" &&
            !!r.source_image_gridfs_id  // must have a base image to display
        );
    }, [inference]);

    // 5. Load Active Image Assets when index changes
    useEffect(() => {
        if (!navigableImages.length) return;

        // Safety check
        const idx = Math.max(0, Math.min(activeImageIndex, navigableImages.length - 1));
        const result = navigableImages[idx];

        if (!result) return; // Should not happen

        let isMounted = true;
        setIsLoadingImage(true);

        const loadAssets = async () => {
            // Find artifact IDs
            const stitchedArt = result.artifacts?.find(a => a.kind === "overlay_stitched");
            const compareArt = result.artifacts?.find(a => a.kind === "figure_compare");
            const classMaskArt = result.artifacts?.find(a => a.kind === "class_mask");
            const instMaskArt = result.artifacts?.find(a => a.kind === "instance_mask");
            const nucleiArt = result.artifacts?.find(a => a.kind === "nuclei_json");

            const stitchedId = stitchedArt?.gridfs_id;
            const compareId = compareArt?.gridfs_id;
            const classId = classMaskArt?.gridfs_id || result.class_mask_id;
            const instId = instMaskArt?.gridfs_id || result.instance_mask_id;
            const nucleiId = nucleiArt?.gridfs_id;

            // source_image_gridfs_id is now always the raw base image
            // (DAPI for FISH, raw scan for D-DISH / CellPose)
            const sourceId = result.source_image_gridfs_id;

            const [src, stitched, compare, cls, inst, nuclei] = await Promise.all([
                sourceId ? fetchFileUrl(sourceId) : undefined,
                stitchedId ? fetchFileUrl(stitchedId) : undefined,
                compareId ? fetchFileUrl(compareId) : undefined,
                classId ? fetchFileUrl(classId) : undefined,
                instId ? fetchFileUrl(instId) : undefined,
                nucleiId ? fetchFileUrl(nucleiId) : undefined,
            ]);

            if (isMounted) {
                setActiveImageState({
                    sourceUrl: src,
                    layers: {
                        stitchedOverlay: stitched,
                        classMask: cls,
                        instanceMask: inst
                    },
                    comparisonUrl: compare,
                    nucleiUrl: nuclei
                });

                // Default active layer
                if (stitched) setActiveLayer("stitchedOverlay");
                else if (inst) setActiveLayer("instanceMask");
                else if (cls) setActiveLayer("classMask");
                else setActiveLayer("none");

                setIsLoadingImage(false);
            }
        };

        loadAssets();

        return () => {
            isMounted = false;
        };
    }, [activeImageIndex, navigableImages]);

    // 6. Gallery Thumbnails (Lazy Load)
    // Simple approach: load all for now if N < 50, otherwise we'd need a window.
    // Given the requirement for "neat and clean", showing spinners is bad.
    const [galleryUrls, setGalleryUrls] = useState<Record<string, string>>({});

    useEffect(() => {
        if (!navigableImages.length) return;

        const windowSize = 5;
        const start = Math.max(0, activeImageIndex - windowSize);
        const end = Math.min(navigableImages.length, activeImageIndex + windowSize + 1);

        const toLoad = navigableImages.slice(start, end)
            .filter(r => {
                const thumbId = r.source_image_gridfs_id;
                return thumbId && !galleryUrls[r.source_filename];
            });

        if (toLoad.length === 0) return;

        let isMounted = true;

        const loadBatch = async () => {
            const newUrls: Record<string, string> = {};
            await Promise.all(toLoad.map(async (r) => {
                const thumbId = r.source_image_gridfs_id;
                if (thumbId) {
                    const url = await fetchFileUrl(thumbId);
                    if (url) newUrls[r.source_filename] = url;
                }
            }));

            if (isMounted) {
                setGalleryUrls(prev => ({ ...prev, ...newUrls }));
            }
        };

        loadBatch();

        return () => { isMounted = false; };
    }, [activeImageIndex, navigableImages, galleryUrls]);


    // 7. Filter CSV for current image
    const activeCsvData = useMemo(() => {
        if (!navigableImages[activeImageIndex]) return [];

        const filename = navigableImages[activeImageIndex].source_filename;
        const getBasename = (path: string) => path.split(/[/\\]/).pop() || path;
        const targetBasename = getBasename(filename);
        // Stem without extension, for FISH CSV matching (image column = stem)
        const targetStem = targetBasename.replace(/\.[^/.]+$/, "").replace(/_DAPI$/i, "");

        return globalCsvData.filter(row => {
            const rowImg = row.image || row.Image;
            if (!rowImg) return false;
            const rowBasename = getBasename(String(rowImg));
            const rowStem = rowBasename.replace(/\.[^/.]+$/, "");
            return rowImg === filename ||
                rowBasename === targetBasename ||
                rowStem === targetStem;        // FISH: image col is stem (no extension)
        });
    }, [globalCsvData, activeImageIndex, navigableImages]);


    // --- Handlers ---

    const handleNext = () => setActiveImageIndex(i => Math.min(i + 1, navigableImages.length - 1));
    const handlePrev = () => setActiveImageIndex(i => Math.max(i - 1, 0));

    const toggleSelection = () => {
        const filename = navigableImages[activeImageIndex]?.source_filename;
        if (!filename) return;
        setSelectedFilenames(curr =>
            curr.includes(filename) ? curr.filter(f => f !== filename) : [...curr, filename]
        );
    };

    const sendToCvat = async () => {
        if (!inference || selectedFilenames.length === 0) {
            toast.error("Select at least one image.");
            return;
        }
        setIsSending(true);
        try {
            const response = await apiFetch(`/api/cvat/push-inference/${inference._id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filenames: selectedFilenames }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to push");
            setCvatLink(data.task_url || data.url);
            toast.success("Sent to CVAT");
        } catch (e: any) {
            toast.error(e.message);
        } finally {
            setIsSending(false);
        }
    };

    const downloadResults = async () => {
        if (!inference) return;
        try {
            const response = await apiFetch(`/api/inferences/${inference._id}/download`);
            if (!response.ok) throw new Error("Download failed");
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `inference_${inference._id}.zip`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (e) {
            toast.error("Download failed");
        }
    };

    if (isLoading || !inference) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-cvat-bg-primary text-cvat-text-secondary">
                <Loader2 className="h-8 w-8 animate-spin" />
            </div>
        );
    }

    // Determine overlay URL based on activeLayer
    const currentOverlayUrl = activeLayer === "none" ? undefined : activeImageState.layers[activeLayer];

    return (
        <div className="flex flex-col min-h-screen bg-cvat-bg-primary overflow-hidden">
            {/* Top Bar / Sub-header */}
            <div className="h-12 border-b border-cvat-border bg-cvat-bg-secondary flex items-center justify-between px-4 shrink-0">
                <div className="flex items-center gap-4">
                    <div>
                        <h1 className="text-sm font-semibold text-cvat-text-primary">
                            {inference.status === "completed" ? "Inference Review" : "Processing Inference..."}
                        </h1>
                        <div className="text-xs text-cvat-text-secondary flex gap-2">
                            <span>ID: {inference._id.slice(-6)}</span>
                            <span>•</span>
                            <span>{navigableImages.length} Images</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    {/* Toolbar */}
                    <button
                        onClick={() => setShowComparison(!showComparison)}
                        className={`p-2 rounded text-xs flex items-center gap-2 border transition-colors ${showComparison
                            ? "bg-cvat-primary/10 text-cvat-primary border-cvat-primary font-medium"
                            : "border-cvat-border text-cvat-text-secondary hover:border-cvat-text-primary hover:text-cvat-text-primary"
                            }`}
                        disabled={!activeImageState.comparisonUrl}
                        title="Identify class distribution changes"
                    >
                        <Layout size={16} /> Comparing
                    </button>

                    <div className="h-6 w-px bg-cvat-border mx-2" />

                    <button
                        onClick={downloadResults}
                        className="p-2 rounded text-xs flex items-center justify-center border border-cvat-border text-cvat-text-secondary hover:text-cvat-text-primary hover:bg-slate-50 transition-colors"
                        title="Download ZIP"
                    >
                        <Download size={16} />
                    </button>



                    <button
                        onClick={sendToCvat}
                        disabled={selectedFilenames.length === 0 || isSending}
                        className={`p-2 rounded text-xs flex items-center gap-2 border transition-all ${isSending || selectedFilenames.length === 0
                            ? "bg-cvat-border text-cvat-text-secondary border-transparent cursor-not-allowed opacity-70"
                            : "bg-cvat-primary text-white border-transparent hover:opacity-90 shadow-sm"
                            }`}
                        title="Start Annotation Task"
                    >
                        {isSending ? (
                            <Loader2 size={16} className="animate-spin" />
                        ) : (
                            <Send size={16} />
                        )}
                        <span>To CVAT ({selectedFilenames.length})</span>
                    </button>



                </div>
            </div>

            {/* Main Workspace */}
            <div className="flex-1 flex overflow-hidden">

                {/* Center Canvas */}
                <div className="flex-1 relative bg-black flex flex-col min-w-0">
                    {/* Canvas Toolbar / Header */}
                    <div className="absolute top-4 left-4 z-20 flex gap-2">
                        <div className="bg-black/60 backdrop-blur text-white px-3 py-1.5 rounded-full text-xs flex items-center gap-2">
                            <span className="opacity-70">File:</span>
                            <span className="font-mono">{navigableImages[activeImageIndex]?.source_filename}</span>
                        </div>
                        <label className="bg-black/60 backdrop-blur text-white px-3 py-1.5 rounded-full text-xs flex items-center gap-2 cursor-pointer hover:bg-black/80 transition-colors">
                            <input
                                type="checkbox"
                                checked={!!navigableImages[activeImageIndex] && selectedFilenames.includes(navigableImages[activeImageIndex].source_filename)}
                                onChange={toggleSelection}
                                className="rounded-sm bg-transparent border-white/50 focus:ring-0 checked:bg-cvat-primary"
                            />
                            Needs Correction
                        </label>
                    </div>

                    {showComparison && activeImageState.comparisonUrl ? (
                        // Comparison View (Interactive)
                        <div className="flex-1 relative w-full h-full">
                            <CanvasViewer
                                sourceUrl={activeImageState.comparisonUrl}
                                className="w-full h-full border-none rounded-none"
                                smoothImage={true} // Comparisons often utilize text/graphs, better to smooth or not? Actually usually text needs no smoothing in canvas if low res. Let's stick to default or smooth.
                            />
                            <button
                                onClick={() => setShowComparison(false)}
                                className="absolute top-4 right-4 bg-black/50 text-white p-2 rounded-full hover:bg-black/80 z-50"
                            >
                                ✕
                            </button>
                        </div>
                    ) : (
                        // Interactive Canvas
                        <CanvasViewer
                            ref={viewerRef}
                            sourceUrl={activeImageState.sourceUrl}
                            maskUrl={currentOverlayUrl}
                            nucleiUrl={activeImageState.nucleiUrl}
                            overlayOpacity={overlayOpacity}
                            smoothImage={smoothImage}
                            className="flex-1 w-full h-full border-none rounded-none"
                        />
                    )}

                    {/* Bottom Floating Controls for Viewer */}
                    {!showComparison && (
                        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-cvat-bg-secondary border border-cvat-border shadow-xl rounded-full px-4 py-2 flex items-center gap-4 z-20">
                            {/* Layer Toggles */}
                            <div className="flex items-center bg-cvat-bg-tertiary rounded-full p-1">
                                {activeImageState.layers.stitchedOverlay && (
                                    <button
                                        onClick={() => setActiveLayer(activeLayer === 'stitchedOverlay' ? 'none' : 'stitchedOverlay')}
                                        className={`px-3 py-1 rounded-full text-xs transition-colors ${activeLayer === 'stitchedOverlay' ? 'bg-indigo-600 text-white shadow-sm' : 'text-cvat-text-secondary hover:text-cvat-text-primary'}`}
                                    >
                                        Overlay
                                    </button>
                                )}
                                {activeImageState.layers.instanceMask && (
                                    <button
                                        onClick={() => setActiveLayer(activeLayer === 'instanceMask' ? 'none' : 'instanceMask')}
                                        className={`px-3 py-1 rounded-full text-xs transition-colors ${activeLayer === 'instanceMask' ? 'bg-emerald-600 text-white shadow-sm' : 'text-cvat-text-secondary hover:text-cvat-text-primary'}`}
                                    >
                                        Instance
                                    </button>
                                )}
                                {activeImageState.layers.classMask && (
                                    <button
                                        onClick={() => setActiveLayer(activeLayer === 'classMask' ? 'none' : 'classMask')}
                                        className={`px-3 py-1 rounded-full text-xs transition-colors ${activeLayer === 'classMask' ? 'bg-purple-600 text-white shadow-sm' : 'text-cvat-text-secondary hover:text-cvat-text-primary'}`}
                                    >
                                        Class
                                    </button>
                                )}
                            </div>

                            <div className="w-px h-4 bg-cvat-border" />

                            {/* Opacity Slider */}
                            <div className="flex items-center gap-2 w-32">
                                <Layers size={14} className="text-cvat-text-secondary" />
                                <input
                                    type="range" min={0} max={1} step={0.1}
                                    value={overlayOpacity}
                                    onChange={e => setOverlayOpacity(parseFloat(e.target.value))}
                                    className="flex-1 h-1 bg-cvat-bg-tertiary rounded-lg appearance-none cursor-pointer"
                                />
                            </div>

                            <div className="w-px h-4 bg-cvat-border" />

                            <button onClick={() => viewerRef.current?.fit()} className="text-cvat-text-secondary hover:text-cvat-text-primary text-xs">
                                Fit
                            </button>
                        </div>
                    )}
                </div>

                {/* Right Side Panel (CSV) */}
                {showCsvPanel && (globalCsvData.length > 0) && (
                    <div className={`${isCsvExpanded ? 'w-2/3' : 'w-80'} min-w-[20rem] bg-cvat-bg-secondary border-l border-cvat-border flex flex-col shrink-0 transition-all duration-300 ease-in-out`}>
                        <CsvViewer
                            data={activeCsvData}
                            className="flex-1"
                            onClose={() => setShowCsvPanel(false)}
                            isExpanded={isCsvExpanded}
                            onToggleExpand={() => setIsCsvExpanded(!isCsvExpanded)}
                            onRowClick={(row) => {
                                // Potential future: zoom to object if row has bbox
                                console.log("Selected", row);
                            }}
                        />
                    </div>
                )}
            </div>

            {/* Bottom Gallery Strip */}
            <div className="bg-cvat-bg-secondary border-t border-cvat-border shrink-0 flex flex-col transition-all duration-300">
                <div
                    className="flex justify-between items-center px-2 py-1 bg-cvat-bg-tertiary text-[10px] text-cvat-text-secondary cursor-pointer hover:bg-cvat-border/50 transition-colors"
                    onClick={() => setIsGalleryCollapsed(!isGalleryCollapsed)}
                >
                    <div className="flex items-center gap-2">
                        {isGalleryCollapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        <span>Gallery ({activeImageIndex + 1} / {navigableImages.length})</span>
                    </div>
                    <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
                        {globalCsvData.length > 0 && (
                            <button
                                onClick={() => setShowCsvPanel(!showCsvPanel)}
                                className={`hover:text-cvat-text-primary ${showCsvPanel ? 'text-cvat-primary' : ''}`}
                            >
                                <Table size={12} className="inline mr-1" /> Data Panel
                            </button>
                        )}
                    </div>
                </div>

                {!isGalleryCollapsed && (
                    <div className="flex items-center animate-in slide-in-from-bottom-2 duration-200">
                        <button
                            onClick={handlePrev} disabled={activeImageIndex === 0}
                            className="p-2 hover:bg-cvat-bg-tertiary disabled:opacity-30 disabled:hover:bg-transparent"
                        >
                            <ChevronLeft size={20} className="text-cvat-text-primary" />
                        </button>

                        <div className="flex-1 overflow-hidden">
                            <ImageGallery
                                images={navigableImages.map(r => ({
                                    id: r.source_filename,
                                    filename: r.source_filename,
                                    url: galleryUrls[r.source_filename]
                                }))}
                                activeId={navigableImages[activeImageIndex]?.source_filename}
                                onSelect={(img) => {
                                    const idx = navigableImages.findIndex(r => r.source_filename === img.id);
                                    if (idx !== -1) setActiveImageIndex(idx);
                                }}
                                className="border-none"
                            />
                        </div>

                        <button
                            onClick={handleNext} disabled={activeImageIndex === navigableImages.length - 1}
                            className="p-2 hover:bg-cvat-bg-tertiary disabled:opacity-30 disabled:hover:bg-transparent"
                        >
                            <ChevronRight size={20} className="text-cvat-text-primary" />
                        </button>
                    </div>
                )}
            </div>

        </div>
    );
}
