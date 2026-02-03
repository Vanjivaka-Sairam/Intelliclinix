
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

// The public API for our Viewer
export interface IViewer {
    html(): HTMLDivElement;
    setup(frameData: FrameData): Promise<void>;
    fitCanvas(): void;
    fit(): void;
    rotate(angle: number): void;
    setZoom(scale: number): void;
    setOverlay(overlay: OverlayData | null): Promise<void>;
    configure(config: CanvasConfiguration): void;
    destroy(): void;

    // Getters for integration
    get scale(): number;
}
