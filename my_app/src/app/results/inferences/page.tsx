"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "react-hot-toast";
import {
    Loader2, Download, Send,
    ZoomIn, ZoomOut, RotateCw,
    Maximize2
} from "lucide-react";
import DashboardNav from "@/components/DashboardNav";
import { apiFetch } from "@/lib/api";
import { useAuthGuard } from "@/hooks/use-auth-guard";
import CanvasViewer, { CanvasViewerRef } from "@/components/ImageViewer/CanvasViewer";

type InferenceResult = {
    source_filename: string;
    source_image_gridfs_id?: string;
    class_mask_id?: string;
    instance_mask_id?: string;
};

type InferenceResponse = {
    _id: string;
    dataset_id: string;
    status: string;
    results: InferenceResult[];
};

type ImageEntry = {
    source?: string;
    classMask?: string;
    instanceMask?: string;
};

export default function InferenceDetailPage() {
    const searchParams = useSearchParams();
    const inferenceId = searchParams.get("id");
    const { isLoading } = useAuthGuard();
    const [inference, setInference] = useState<InferenceResponse | null>(null);
    const [isPolling, setIsPolling] = useState(true);
    const [selectedFilenames, setSelectedFilenames] = useState<string[]>([]);
    const [imageMap, setImageMap] = useState<Record<string, ImageEntry>>({});
    const [overlayOpacity, setOverlayOpacity] = useState(0.6);
    const [activeOverlay, setActiveOverlay] = useState<"classMask" | "instanceMask">("instanceMask");
    const [isSending, setIsSending] = useState(false);
    const [cvatLink, setCvatLink] = useState<string | null>(null);
    const [smoothImage, setSmoothImage] = useState(false);

    // We can track refs for each viewer if we want to control them externally, 
    // but for now, individual controls inside the viewer or simple scroll zoom is efficient.
    const pollRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        setSelectedFilenames([]);
        setCvatLink(null);
    }, [inferenceId]);

    useEffect(() => {
        if (isLoading || !inferenceId) return;

        const fetchInference = async () => {
            try {
                const response = await apiFetch(`/api/inferences/${inferenceId}`);
                if (!response.ok) {
                    throw new Error("Failed to load inference");
                }
                const data: InferenceResponse = await response.json();
                setInference(data);
                if (data.status === "completed") {
                    setIsPolling(false);
                    hydrateImages(data.results);
                } else {
                    setIsPolling(true);
                }
            } catch (error) {
                console.error(error);
                toast.error("Unable to load inference");
            }
        };

        fetchInference();
        if (isPolling) {
            pollRef.current = setInterval(fetchInference, 5000);
        }

        return () => {
            if (pollRef.current) {
                clearInterval(pollRef.current);
            }
        };
    }, [inferenceId, isLoading, isPolling]);

    const hydrateImages = async (results: InferenceResult[]) => {
        const newEntries: Record<string, ImageEntry> = {};
        for (const result of results) {
            newEntries[result.source_filename] = {
                source: await fetchImageUrl(result.source_image_gridfs_id),
                classMask: await fetchImageUrl(result.class_mask_id),
                instanceMask: await fetchImageUrl(result.instance_mask_id),
            };
        }
        setImageMap(newEntries);
    };

    const fetchImageUrl = async (gridfsId?: string) => {
        if (!gridfsId) return undefined;
        try {
            const response = await apiFetch(`/api/files/${gridfsId}`);
            if (!response.ok) {
                throw new Error("Failed to fetch image");
            }
            const blob = await response.blob();
            return URL.createObjectURL(blob);
        } catch (error) {
            console.error(error);
            return undefined;
        }
    };

    useEffect(() => {
        return () => {
            Object.values(imageMap).forEach((entry) => {
                entry.source && URL.revokeObjectURL(entry.source);
                entry.classMask && URL.revokeObjectURL(entry.classMask);
                entry.instanceMask && URL.revokeObjectURL(entry.instanceMask);
            });
        };
    }, [imageMap]);

    const toggleSelection = (filename: string) => {
        setSelectedFilenames((current) =>
            current.includes(filename)
                ? current.filter((item) => item !== filename)
                : [...current, filename]
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
            if (!response.ok) {
                throw new Error(data.error || "Failed to push to CVAT");
            }
            setCvatLink(data.task_url || data.url || null);
            toast.success("Images sent to CVAT");
        } catch (error) {
            console.error(error);
            toast.error(error instanceof Error ? error.message : "Failed to push to CVAT");
        } finally {
            setIsSending(false);
        }
    };

    const downloadResults = async () => {
        if (!inference) return;
        try {
            const response = await apiFetch(`/api/inferences/${inference._id}/download`);
            if (!response.ok) {
                throw new Error("Download failed");
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `inference_${inference._id}.zip`;
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error(error);
            toast.error("Unable to download results");
        }
    };

    const isReady = useMemo(
        () => Boolean(inference && inference.status === "completed"),
        [inference]
    );

    if (isLoading || !inference) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-cvat-bg-primary">
                <Loader2 className="h-6 w-6 animate-spin text-cvat-primary" />
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-cvat-bg-primary">
            <DashboardNav />
            <div className="max-w-7xl mx-auto px-4 py-10 space-y-6">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                        <h1 className="text-2xl font-semibold text-cvat-text-primary">
                            Inference #{inference._id}
                        </h1>
                        <p className="text-sm text-cvat-text-secondary">
                            Dataset: {inference.dataset_id}
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        <span
                            className={`text-xs font-semibold px-3 py-1 rounded-full ${inference.status === "completed"
                                ? "bg-emerald-100 text-emerald-700"
                                : inference.status === "failed"
                                    ? "bg-rose-100 text-rose-700"
                                    : "bg-amber-100 text-amber-700"
                                }`}
                        >
                            {inference.status.toUpperCase()}
                        </span>
                        <button
                            onClick={downloadResults}
                            className="flex items-center gap-2 rounded-md border border-cvat-border px-4 py-2 text-sm text-cvat-text-secondary hover:text-cvat-text-primary"
                            disabled={!isReady}
                        >
                            <Download className="h-4 w-4" />
                            Download ZIP
                        </button>
                        <button
                            onClick={sendToCvat}
                            className="flex items-center gap-2 rounded-md bg-cvat-primary px-4 py-2 text-sm text-white disabled:opacity-50"
                            disabled={!isReady || isSending || selectedFilenames.length === 0}
                        >
                            {isSending ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                                <Send className="h-4 w-4" />
                            )}
                            Send to CVAT
                        </button>
                    </div>
                </div>

                {cvatLink && (
                    <div className="cvat-card p-4 text-sm">
                        <span className="text-cvat-text-secondary">CVAT Task: </span>
                        <a href={cvatLink} target="_blank" className="text-cvat-primary hover:underline" rel="noreferrer">
                            {cvatLink}
                        </a>
                    </div>
                )}

                {!isReady ? (
                    <div className="cvat-card p-6 text-center text-cvat-text-secondary">
                        {inference.status === "failed"
                            ? "This inference failed. Please review the backend logs."
                            : "Inference is still running. This page will refresh automatically."}
                    </div>
                ) : (
                    <div className="grid gap-6 lg:grid-cols-2">
                        {inference.results.map((result) => {
                            const entry = imageMap[result.source_filename];
                            return (
                                <div key={result.source_filename} className="cvat-card overflow-hidden">
                                    <div className="flex items-start justify-between p-4 bg-cvat-bg-secondary border-b border-cvat-border">
                                        <div>
                                            <p className="font-medium text-cvat-text-primary">{result.source_filename}</p>
                                            <p className="text-xs text-cvat-text-secondary">
                                                {selectedFilenames.includes(result.source_filename)
                                                    ? "Marked for CVAT"
                                                    : "Review and mark for correction"}
                                            </p>
                                        </div>
                                        <label className="inline-flex items-center gap-2 text-sm text-cvat-text-primary">
                                            <input
                                                type="checkbox"
                                                className="rounded border-cvat-border text-cvat-primary focus:ring-cvat-primary"
                                                checked={selectedFilenames.includes(result.source_filename)}
                                                onChange={() => toggleSelection(result.source_filename)}
                                            />
                                            Needs correction
                                        </label>
                                    </div>
                                    <div className="relative h-[400px] w-full bg-black">
                                        {entry?.source && (
                                            <CanvasViewer
                                                sourceUrl={entry.source}
                                                maskUrl={entry?.[activeOverlay]}
                                                overlayOpacity={overlayOpacity}
                                                smoothImage={smoothImage}
                                                className="w-full h-full"
                                            />
                                        )}
                                        {!entry?.source && (
                                            <div className="flex items-center justify-center h-full text-cvat-text-secondary">
                                                Loading source image…
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}

                {isReady && (
                    <div className="cvat-card p-4 flex flex-col gap-4 md:flex-row md:items-center md:justify-between sticky bottom-4 z-10 shadow-lg border-cvat-primary/20">
                        <div className="flex items-center gap-4 text-sm text-cvat-text-secondary">
                            <span>Overlay:</span>
                            <button
                                className={`px-3 py-1 rounded-full text-xs font-semibold ${activeOverlay === "instanceMask"
                                    ? "bg-cvat-primary text-white"
                                    : "bg-cvat-bg-tertiary text-cvat-text-secondary"
                                    }`}
                                onClick={() => setActiveOverlay("instanceMask")}
                            >
                                Instance
                            </button>
                            <button
                                className={`px-3 py-1 rounded-full text-xs font-semibold ${activeOverlay === "classMask"
                                    ? "bg-cvat-primary text-white"
                                    : "bg-cvat-bg-tertiary text-cvat-text-secondary"
                                    }`}
                                onClick={() => setActiveOverlay("classMask")}
                            >
                                Class
                            </button>
                        </div>
                        <div className="flex items-center gap-3 text-sm text-cvat-text-secondary">
                            <span>Opacity</span>
                            <input
                                type="range"
                                min={0}
                                max={1}
                                step={0.05}
                                value={overlayOpacity}
                                onChange={(event) => setOverlayOpacity(Number(event.target.value))}
                                className="w-32"
                            />
                            <span className="w-8 text-right text-cvat-text-primary">
                                {Math.round(overlayOpacity * 100)}%
                            </span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                            <label className="flex items-center gap-2 cursor-pointer text-cvat-text-primary">
                                <input
                                    type="checkbox"
                                    checked={smoothImage}
                                    onChange={(e) => setSmoothImage(e.target.checked)}
                                    className="rounded border-cvat-border"
                                />
                                Smooth Image
                            </label>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
