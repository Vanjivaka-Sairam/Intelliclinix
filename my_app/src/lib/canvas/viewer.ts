/**
 * viewer.ts
 * Main entry point, equivalent to cvat-canvas/src/typescript/canvas.ts
 */

import RBush from 'rbush';
import pointInPolygon from 'point-in-polygon';
import { IViewer, FrameData, CanvasConfiguration, OverlayData, NucleiData, Nucleus, NucleusHoverHandler } from './types';
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

    // Vector Layer & Interaction
    private nucleiData: NucleiData | null = null;
    private rbush: RBush<any> | null = null;
    private hoveredNucleus: Nucleus | null = null;
    private hoverHandler: NucleusHoverHandler | null = null;

    private config: CanvasConfiguration = {
        smoothImage: false,
        displayGrid: false,
        gridSize: 50,
        gridColor: 'white',
        backgroundColor: '#111827'
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
        if (this.imageBitmap) {
            this.imageBitmap.close();
            this.imageBitmap = null;
        }

        try {
            const response = await fetch(frameData.url);
            const blob = await response.blob();
            if (this.isDestroyed) return;
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

    public setNucleiData(data: NucleiData | null): void {
        console.log("Viewer: setNucleiData called", data ? `with ${data.nuclei.length} nuclei` : "null");
        this.nucleiData = data;
        this.hoveredNucleus = null;

        if (data && data.nuclei.length > 0) {
            this.rbush = new RBush();
            // Bulk load RBush
            const items = data.nuclei.map(n => {
                // Determine bounding box
                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                for (const p of n.polygon) {
                    if (p[0] < minX) minX = p[0];
                    if (p[0] > maxX) maxX = p[0];
                    if (p[1] < minY) minY = p[1];
                    if (p[1] > maxY) maxY = p[1];
                }
                return {
                    minX, minY, maxX, maxY,
                    item: n
                };
            });
            this.rbush.load(items);
            console.log("Viewer: RBush loaded with", items.length, "items");
        } else {
            this.rbush = null;
        }
        this.requestRender();
    }

    public onHover(handler: NucleusHoverHandler): void {
        this.hoverHandler = handler;
    }

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

    private attachEventListeners() {
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const ZOOM_SPEED = 0.001;
            const delta = -e.deltaY * ZOOM_SPEED;
            const factor = 1 + delta;

            this.model.zoom(factor, e.offsetX, e.offsetY);
            this.requestRender();
        }, { passive: false });

        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            // Right click logic if needed
        });

        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;
            const { x, y } = this.model.canvasToImage(canvasX, canvasY);
            console.log(`Click at Canvas(${canvasX.toFixed(1)}, ${canvasY.toFixed(1)}) -> Image(${x.toFixed(1)}, ${y.toFixed(1)})`);
        });

        this.canvas.addEventListener('mousedown', (e) => {
            if (e.button !== 0 && e.button !== 1) return;
            this.isDragging = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.container.style.cursor = 'grabbing';
            e.preventDefault();
        });

        window.addEventListener('mousemove', (e) => {
            if (this.isDestroyed) return;

            // 1. Pan Logic
            if (this.isDragging) {
                const dx = e.clientX - this.lastMouse.x;
                const dy = e.clientY - this.lastMouse.y;
                this.model.pan(dx, dy);
                this.lastMouse = { x: e.clientX, y: e.clientY };
                this.requestRender();
                return;
            }

            // 2. Hover Logic (only if not dragging)
            if (this.nucleiData && this.rbush) {
                const rect = this.canvas.getBoundingClientRect();
                const canvasX = e.clientX - rect.left;
                const canvasY = e.clientY - rect.top;

                const { x: imgX, y: imgY } = this.model.canvasToImage(canvasX, canvasY);
                console.log("MouseMove:", canvasX, canvasY, "-> Image:", imgX, imgY);

                // Query RBush
                const candidates = this.rbush.search({
                    minX: imgX, minY: imgY, maxX: imgX, maxY: imgY
                });

                if (candidates.length > 0) {
                    console.log("Candidates found:", candidates.length);
                }

                let found: Nucleus | null = null;
                for (const candidate of candidates) {
                    const poly = candidate.item.polygon; // [[x,y],...]
                    if (pointInPolygon([imgX, imgY], poly)) {
                        found = candidate.item;
                        console.log("Hit found:", found.id);
                        break;
                    }
                }

                if (found !== this.hoveredNucleus) {
                    this.hoveredNucleus = found;
                    this.requestRender();

                    if (this.hoverHandler) {
                        this.hoverHandler(found, e.clientX, e.clientY);
                    }
                } else if (found && this.hoverHandler) {
                    // Even if same nucleus, update position for tooltip logic
                    this.hoverHandler(found, e.clientX, e.clientY);
                } else if (!found && this.hoveredNucleus === null && this.hoverHandler) {
                    // Ensure tooltip hides if we move out of bounds
                    // Optimization: Only fire if we previously had a nucleus or need to clear
                    // But we rely on this to hide the tooltip, so we should fire if it was visible?
                    // actually page component handles visibility based on null.
                    // To avoid spamming nulls:
                    // We can't easily know if the parent thinks it's visible without state.
                    // But since we are 'hoveredNucleus === null', we might have just exited.
                    // Let's just fire it.
                    this.hoverHandler(null, e.clientX, e.clientY);
                }
            }
        });

        window.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.container.style.cursor = 'default';
            }
        });

        this.canvas.addEventListener('touchmove', (e) => {
            if (e.touches.length === 1) e.preventDefault();
        }, { passive: false });
    }

    private requestRender() {
        requestAnimationFrame(this.render.bind(this));
    }

    private render() {
        if (!this.ctx) return;

        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = this.config.backgroundColor || '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        if (!this.imageBitmap) return;

        this.ctx.imageSmoothingEnabled = !!this.config.smoothImage;

        this.ctx.save();
        try {
            this.ctx.translate(this.model.x, this.model.y);
            this.ctx.scale(this.model.scale, this.model.scale);

            if (this.model.angle !== 0) {
                this.ctx.translate(this.model.imageWidth / 2, this.model.imageHeight / 2);
                this.ctx.rotate((this.model.angle * Math.PI) / 180);
                this.ctx.translate(-this.model.imageWidth / 2, -this.model.imageHeight / 2);
            }

            // Draw Image
            this.ctx.drawImage(this.imageBitmap, 0, 0);

            // 5. Draw Overlay
            if (this.overlayBitmap) {
                this.ctx.globalAlpha = this.overlayOpacity;
                this.ctx.drawImage(this.overlayBitmap, 0, 0, this.model.imageWidth, this.model.imageHeight);
                this.ctx.globalAlpha = 1.0;
            }

            // DEBUG: Draw ALL Nuclei to verify alignment
            if (this.nucleiData && this.nucleiData.nuclei) {
                this.ctx.lineWidth = 1 / this.model.scale;
                this.ctx.strokeStyle = 'cyan';
                this.ctx.globalAlpha = 0.3;

                for (const nucleus of this.nucleiData.nuclei) {
                    const poly = nucleus.polygon;
                    if (poly.length < 3) continue;

                    this.ctx.beginPath();
                    this.ctx.moveTo(poly[0][0], poly[0][1]);
                    for (let i = 1; i < poly.length; i++) {
                        this.ctx.lineTo(poly[i][0], poly[i][1]);
                    }
                    this.ctx.closePath();
                    this.ctx.stroke();
                }
                this.ctx.globalAlpha = 1.0;
            }

            // Draw Vector Layer (Interactive)
            // We can draw *just* the hovered nucleus for performance, or all if we want outlines
            // For now, let's draw the hovered nucleus on top to highlight it.
            if (this.hoveredNucleus) {
                const poly = this.hoveredNucleus.polygon;
                if (poly.length > 0) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(poly[0][0], poly[0][1]);
                    for (let i = 1; i < poly.length; i++) {
                        this.ctx.lineTo(poly[i][0], poly[i][1]);
                    }
                    this.ctx.closePath();

                    this.ctx.strokeStyle = '#00FF00'; // Bright green
                    this.ctx.lineWidth = 2 / this.model.scale; // Constant width regardless of zoom
                    this.ctx.stroke();

                    this.ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
                    this.ctx.fill();
                }
            }

            // Draw Grid
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
