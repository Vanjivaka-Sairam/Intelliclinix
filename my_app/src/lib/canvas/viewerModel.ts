
/**
 * viewerModel.ts
 * Manages state: scale, rotation, coordinates.
 */

export class ViewerModel {
    public scale: number = 1;
    public angle: number = 0; // 0, 90, 180, 270
    public x: number = 0;
    public y: number = 0;

    // Image dimensions
    public imageWidth: number = 0;
    public imageHeight: number = 0;

    // Canvas/Container dimensions
    public canvasWidth: number = 0;
    public canvasHeight: number = 0;

    constructor() { }

    public setImageSize(w: number, h: number) {
        this.imageWidth = w;
        this.imageHeight = h;
    }

    public setCanvasSize(w: number, h: number) {
        this.canvasWidth = w;
        this.canvasHeight = h;
    }

    /**
     * Calculates the "Fit to Screen" transform.
     * Mirrors the logic found in CVAT's fitCanvas method.
     */
    public fit() {
        if (this.canvasWidth === 0 || this.canvasHeight === 0 || this.imageWidth === 0 || this.imageHeight === 0) return;

        // Account for rotation swapping width/height
        const isVertical = this.angle % 180 !== 0;
        const effImgW = isVertical ? this.imageHeight : this.imageWidth;
        const effImgH = isVertical ? this.imageWidth : this.imageHeight;

        const scaleX = this.canvasWidth / effImgW;
        const scaleY = this.canvasHeight / effImgH;

        // Use the smaller scale to ensure the whole image is visible
        this.scale = Math.min(scaleX, scaleY) * 0.95; // 0.95 for a small margin

        // Center the image
        this.x = (this.canvasWidth - this.imageWidth * this.scale) / 2;
        this.y = (this.canvasHeight - this.imageHeight * this.scale) / 2;
    }

    /**
     * Zoom-to-point logic.
     * @param zoomFactor Multiplier (e.g., 1.1 or 0.9)
     * @param clientX Mouse X relative to canvas
     * @param clientY Mouse Y relative to canvas
     */
    public zoom(zoomFactor: number, clientX: number, clientY: number) {
        const newScale = this.scale * zoomFactor;

        // Prevent infinite zoom boundaries
        if (newScale < 0.01 || newScale > 100) return;

        // P_new = Mouse - (Mouse - P_old) * (ScaleNew / ScaleOld)
        this.x = clientX - (clientX - this.x) * (newScale / this.scale);
        this.y = clientY - (clientY - this.y) * (newScale / this.scale);

        this.scale = newScale;
    }

    public pan(dx: number, dy: number) {
        this.x += dx;
        this.y += dy;
    }

    /**
     * Converts screen/canvas coordinates to image coordinates.
     * Useful for hit detection.
     */
    public canvasToImage(canvasX: number, canvasY: number): { x: number, y: number } {
        // canvasX = imgX * scale + this.x
        // imgX = (canvasX - this.x) / scale
        return {
            x: (canvasX - this.x) / this.scale,
            y: (canvasY - this.y) / this.scale
        };
    }
}
