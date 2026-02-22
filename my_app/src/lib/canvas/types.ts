
export enum CanvasMode {
    IDLE = 'idle',
    DRAG_CANVAS = 'drag_canvas',
    ZOOM_CANVAS = 'zoom_canvas',
}

export interface CanvasConfiguration {
    smoothImage?: boolean;
    displayGrid?: boolean;
    gridSize?: number; // in pixels
    gridColor?: string;
    backgroundColor?: string;
}

export interface FrameData {
    url: string; // URL to the image blob
    filename: string;
    width: number;
    height: number;
}

export interface OverlayData {
    url: string;
    opacity?: number;
}

export interface Nucleus {
    id: number;
    polygon?: number[][] | null; // [[x,y], [x,y], ...] contour (D-DISH) or bbox corners (FISH)
    bbox?: { x1: number; y1: number; x2: number; y2: number }; // raw bbox (FISH only)
    stats: Record<string, number>; // generic: {Her2, Chr17, ...} or {Green, Red, Aqua, Fusion}
}

export interface NucleiData {
    image_id: string;
    nuclei: Nucleus[];
}

export type NucleusHoverHandler = (nucleus: Nucleus | null, x: number, y: number) => void;

// The public API for our Viewer
export interface IViewer {
    html(): HTMLDivElement;
    setup(frameData: FrameData): Promise<void>;
    fitCanvas(): void;
    fit(): void;
    rotate(angle: number): void;
    setZoom(scale: number): void;
    setOverlay(overlay: OverlayData | null): Promise<void>;
    setNucleiData(data: NucleiData | null): void; // New
    onHover(handler: NucleusHoverHandler): void; // New
    configure(config: CanvasConfiguration): void;
    destroy(): void;

    // Getters for integration
    get scale(): number;
}
