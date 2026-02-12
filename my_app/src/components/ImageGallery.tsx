import React, { useRef, useEffect } from 'react';
import { Loader2 } from 'lucide-react';

export interface ImageItem {
    id: string; // usually source_filename or unique ID
    url?: string;
    filename: string;
    thumbnailUrl?: string; // Optional: If we implement thumbnails later
}

interface ImageGalleryProps {
    images: ImageItem[];
    activeId: string;
    onSelect: (image: ImageItem) => void;
    className?: string;
}

export const ImageGallery: React.FC<ImageGalleryProps> = ({ images, activeId, onSelect, className }) => {
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    // Scroll active item into view when it changes
    useEffect(() => {
        if (scrollContainerRef.current) {
            const activeEl = scrollContainerRef.current.querySelector('[data-active="true"]');
            if (activeEl) {
                activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
            }
        }
    }, [activeId]);

    return (
        <div
            ref={scrollContainerRef}
            className={`flex gap-2 overflow-x-auto p-1 bg-cvat-bg-secondary border-t border-cvat-border custom-scrollbar h-20 items-center ${className}`}
        >
            {images.map((img) => {
                const isActive = img.id === activeId;
                return (
                    <div
                        key={img.id}
                        data-active={isActive}
                        onClick={() => onSelect(img)}
                        className={`
                            relative flex-shrink-0 cursor-pointer rounded overflow-hidden border-2 transition-all
                            w-20 h-14 bg-black
                            ${isActive
                                ? 'border-cvat-primary ring-2 ring-cvat-primary/20'
                                : 'border-transparent hover:border-white/20'}
                        `}
                    >
                        {img.url ? (
                            <img
                                src={img.url}
                                alt={img.filename}
                                className="w-full h-full object-cover"
                                loading="lazy"
                            />
                        ) : (
                            <div className="w-full h-full flex items-center justify-center bg-cvat-bg-tertiary">
                                <Loader2 className="w-4 h-4 text-cvat-text-secondary animate-spin" />
                            </div>
                        )}

                        <div className="absolute inset-x-0 bottom-0 bg-black/60 p-1">
                            <p className="text-[10px] text-white truncate text-center">
                                {img.filename}
                            </p>
                        </div>
                    </div>
                );
            })}
            {images.length === 0 && (
                <div className="w-full text-center text-xs text-cvat-text-secondary py-4">
                    No images available.
                </div>
            )}
        </div>
    );
};
