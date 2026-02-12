/**
 * viewer.ts
 * Main entry point, equivalent to cvat-canvas/src/typescript/canvas.ts
 */

import { IViewer, FrameData, CanvasConfiguration, OverlayData } from './types';
import { ViewerModel } from './viewerModel';

export class CvatLikeViewer implements IViewer {
    public model: ViewerModel;
    private container!: HTMLDivElement;
    private canvas!: HTMLCanvasElement;
    private ctx!: CanvasRenderingContext2D;

    // Layers
    private imageBitmap: ImageBitmap | null = null;
    private overlayBitmap: ImageBitmap | null = null;
    private overlayOpacity: number = 0.6;

    private config: CanvasConfiguration = {
        smoothImage: false, // Default to false for pixel art/medical imaging
        displayGrid: false,
        gridSize: 50,
        gridColor: 'white',
        backgroundColor: '#111827' // Tailwind gray-900 like
    };

    private isDragging: boolean = false;
    private lastMouse: { x: number; y: number } = { x: 0, y: 0 };
    private resizeObserver!: ResizeObserver;

    private isDestroyed: boolean = false;

    constructor() {
        this.model = new ViewerModel();

        // Initialize DOM structure
        this.container = document.createElement('div');
        this.container.style.width = '100%';
        this.container.style.height = '100%';
        this.container.style.overflow = 'hidden';
        this.container.style.position = 'relative';
        this.container.style.backgroundColor = this.config.backgroundColor!;
        // Prevent touch scrolling on mobile
        this.container.style.touchAction = 'none';

        this.canvas = document.createElement('canvas');
        this.canvas.style.display = 'block';

        this.container.appendChild(this.canvas);

        const context = this.canvas.getContext('2d', { alpha: false });
        if (!context) throw new Error("Canvas 2D context not supported");
        this.ctx = context;

        // Bind Resize Observer
        this.resizeObserver = new ResizeObserver(() => this.fitCanvas());
        this.resizeObserver.observe(this.container);

        this.attachEventListeners();
    }

    public html(): HTMLDivElement {
        return this.container;
    }

    public get scale(): number {
        return this.model.scale;
    }

    public configure(config: CanvasConfiguration): void {
        this.config = { ...this.config, ...config };
        if (this.config.backgroundColor) {
            this.container.style.backgroundColor = this.config.backgroundColor;
        }
        this.requestRender();
    }

    public async setup(frameData: FrameData): Promise<void> {
        // Clean up old resources
        if (this.imageBitmap) {
            this.imageBitmap.close();
            this.imageBitmap = null;
        }

        try {
            const response = await fetch(frameData.url);
            const blob = await response.blob();
            if (this.isDestroyed) return; // Prevent setting if destroyed
            this.imageBitmap = await createImageBitmap(blob);
            this.model.setImageSize(this.imageBitmap.width, this.imageBitmap.height);
            this.model.fit();
            this.requestRender();
        } catch (e) {
            console.error("Failed to load image", e);
        }
    }

    public async setOverlay(data: OverlayData | null): Promise<void> {
        if (!data) {
            if (this.overlayBitmap) {
                this.overlayBitmap.close();
                this.overlayBitmap = null;
            }
            this.requestRender();
            return;
        }

        this.overlayOpacity = data.opacity ?? 0.6;

        try {
            const response = await fetch(data.url);
            const blob = await response.blob();

            if (this.isDestroyed) return;

            // Close previous if exists
            if (this.overlayBitmap) {
                this.overlayBitmap.close();
                this.overlayBitmap = null;
            }

            this.overlayBitmap = await createImageBitmap(blob);
            this.requestRender();
        } catch (e) {
            console.error("Failed to load overlay", e);
        }
    }

    /**
     * Resizes the internal canvas resolution to match the DOM display size.
     */
    public fitCanvas(): void {
        const { clientWidth, clientHeight } = this.container;
        if (clientWidth === 0 || clientHeight === 0) return;

        this.canvas.width = clientWidth;
        this.canvas.height = clientHeight;
        this.model.setCanvasSize(clientWidth, clientHeight);

        this.requestRender();
    }

    public fit(): void {
        this.model.fit();
        this.requestRender();
    }

    public rotate(angle: number): void {
        this.model.angle = angle;
        this.requestRender();
    }

    public setZoom(scale: number): void {
        this.model.scale = scale;
        this.requestRender();
    }

    public destroy(): void {
        this.isDestroyed = true;
        this.resizeObserver.disconnect();
        if (this.imageBitmap) {
            this.imageBitmap.close();
            this.imageBitmap = null;
        }
        if (this.overlayBitmap) {
            this.overlayBitmap.close();
            this.overlayBitmap = null;
        }
        // Remove container content
        if (this.container) {
            this.container.innerHTML = '';
            this.container.remove();
        }
    }

    // =================================================================
    // Event Handling
    // =================================================================

    private attachEventListeners() {
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const ZOOM_SPEED = 0.001;
            const delta = -e.deltaY * ZOOM_SPEED;
            const factor = 1 + delta;

            this.model.zoom(factor, e.offsetX, e.offsetY);
            this.requestRender();
        }, { passive: false });

        this.canvas.addEventListener('mousedown', (e) => {
            if (e.button !== 0 && e.button !== 1) return; // Left or Middle
            this.isDragging = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.container.style.cursor = 'grabbing';
            e.preventDefault(); // Check if this stops text selection
        });

        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            const dx = e.clientX - this.lastMouse.x;
            const dy = e.clientY - this.lastMouse.y;
            this.model.pan(dx, dy);
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.requestRender();
        });

        window.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.container.style.cursor = 'default';
            }
        });

        // Basic Touch support for Panning
        this.canvas.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                this.isDragging = true;
                this.lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
            }
        }, { passive: false });

        this.canvas.addEventListener('touchmove', (e) => {
            if (this.isDragging && e.touches.length === 1) {
                e.preventDefault(); // Stop scrolling
                const dx = e.touches[0].clientX - this.lastMouse.x;
                const dy = e.touches[0].clientY - this.lastMouse.y;
                this.model.pan(dx, dy);
                this.lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
                this.requestRender();
            }
        }, { passive: false });

        this.canvas.addEventListener('touchend', () => {
            this.isDragging = false;
        });
    }

    // =================================================================
    // Rendering
    // =================================================================

    private requestRender() {
        requestAnimationFrame(this.render.bind(this));
    }

    private render() {
        if (!this.ctx) return;

        // 0. Safety: Reset transform to identity to ensure we are clearing the full screen
        // regardless of previous errors or state leaks.
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);

        // 1. Clear
        // Explicitly clear before filling to prevent artifacts
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.fillStyle = this.config.backgroundColor || '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        if (!this.imageBitmap) return;

        // 2. Setup Smoothing
        this.ctx.imageSmoothingEnabled = !!this.config.smoothImage;

        this.ctx.save();
        try {
            // 3. Transform
            this.ctx.translate(this.model.x, this.model.y);
            this.ctx.scale(this.model.scale, this.model.scale);

            if (this.model.angle !== 0) {
                this.ctx.translate(this.model.imageWidth / 2, this.model.imageHeight / 2);
                this.ctx.rotate((this.model.angle * Math.PI) / 180);
                this.ctx.translate(-this.model.imageWidth / 2, -this.model.imageHeight / 2);
            }

            // 4. Draw Image
            this.ctx.drawImage(this.imageBitmap, 0, 0);

            // 5. Draw Overlay
            if (this.overlayBitmap) {
                this.ctx.globalAlpha = this.overlayOpacity;
                // Assuming overlay matches image dimensions exactly
                this.ctx.drawImage(this.overlayBitmap, 0, 0, this.model.imageWidth, this.model.imageHeight);
                this.ctx.globalAlpha = 1.0;
            }

            // 6. Draw Grid
            if (this.config.displayGrid && this.model.scale > 0.5) {
                this.drawGrid();
            }
        } finally {
            this.ctx.restore();
        }
    }

    private drawGrid() {
        const step = this.config.gridSize || 100;
        this.ctx.beginPath();
        this.ctx.strokeStyle = this.config.gridColor || 'rgba(255, 255, 255, 0.2)';
        this.ctx.lineWidth = 1 / this.model.scale;

        for (let x = 0; x <= this.model.imageWidth; x += step) {
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.model.imageHeight);
        }
        for (let y = 0; y <= this.model.imageHeight; y += step) {
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.model.imageWidth, y);
        }
        this.ctx.stroke();
    }
}
