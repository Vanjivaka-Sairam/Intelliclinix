import React, { useEffect, useState, useMemo } from 'react';
import Papa from 'papaparse';
import { Maximize2, Minimize2, X } from 'lucide-react';

interface CsvViewerProps {
    fileUrl?: string; // URL to the CSV file
    dataString?: string; // Or raw CSV string
    data?: any[]; // Or already parsed data array
    onRowClick?: (row: any) => void;
    className?: string;
    onClose?: () => void;
    isExpanded?: boolean;
    onToggleExpand?: () => void;
}

export const CsvViewer: React.FC<CsvViewerProps> = ({
    fileUrl,
    dataString,
    data: propData,
    onRowClick,
    className,
    onClose,
    isExpanded = false,
    onToggleExpand
}) => {
    // Internal state for fetched/parsed content
    const [fetchedData, setFetchedData] = useState<any[]>([]);
    const [fetchHeaders, setFetchHeaders] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Effect: Handle fetching from URL or parsing raw string
    useEffect(() => {
        // If propData is provided, we don't need to fetch/parse (unless we want to support mixed mode, which is rare)
        if (propData) return;

        const parseContent = (content: string) => {
            Papa.parse(content, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        console.error("CSV Parse Errors", results.errors);
                        setError("Failed to parse CSV data.");
                    }
                    if (results.meta.fields) {
                        setFetchHeaders(results.meta.fields);
                    } else if (results.data.length > 0) {
                        setFetchHeaders(Object.keys(results.data[0] as object));
                    }
                    setFetchedData(results.data);
                    setLoading(false);
                },
                error: (err: Error) => {
                    setError(err.message);
                    setLoading(false);
                }
            });
        };

        if (fileUrl) {
            setLoading(true);
            setError(null);
            fetch(fileUrl)
                .then(res => {
                    if (!res.ok) throw new Error("Failed to fetch CSV file");
                    return res.text();
                })
                .then(parseContent)
                .catch(err => {
                    setError(err.message);
                    setLoading(false);
                });
        } else if (dataString) {
            setLoading(true);
            parseContent(dataString);
        } else {
            // Reset if nothing
            setFetchedData([]);
            setFetchHeaders([]);
        }
    }, [fileUrl, dataString, propData]);

    // Derived state: Use propData if available, otherwise fetchedData
    const displayData = propData || fetchedData;

    // Calculate headers from displayData if not available from fetch (e.g. if propData was passed without headers info)
    const displayHeaders = useMemo(() => {
        if (propData && propData.length > 0) {
            return Object.keys(propData[0]);
        }
        return fetchHeaders;
    }, [propData, fetchHeaders]);

    if (loading) {
        return <div className="text-center p-4 text-cvat-text-secondary">Loading CSV...</div>;
    }

    if (error) {
        return <div className="text-center p-4 text-rose-500">Error: {error}</div>;
    }

    if (!displayData || displayData.length === 0) {
        return (
            <div className={`flex flex-col h-full bg-cvat-bg-secondary border-l border-cvat-border ${className}`}>
                <div className="flex justify-between items-center p-2 bg-cvat-bg-tertiary border-b border-cvat-border">
                    <span className="text-xs font-semibold text-cvat-text-primary uppercase tracking-wider">
                        Metadata
                    </span>
                    {onClose && (
                        <button onClick={onClose} className="text-cvat-text-secondary hover:text-cvat-text-primary">
                            ✕
                        </button>
                    )}
                </div>
                <div className="flex-1 flex items-center justify-center text-cvat-text-secondary p-4">
                    No data available for this selection.
                </div>
            </div>
        );
    }

    return (
        <div className={`flex flex-col h-full bg-cvat-bg-secondary ${className}`}>
            <div className="flex justify-between items-center p-2 bg-cvat-bg-tertiary border-b border-cvat-border">
                <span className="text-xs font-semibold text-cvat-text-primary uppercase tracking-wider">
                    Metadata ({displayData.length} rows)
                </span>
                <div className="flex items-center gap-1">
                    {onToggleExpand && (
                        <button onClick={onToggleExpand} className="p-1 text-cvat-text-secondary hover:text-cvat-text-primary rounded hover:bg-white/5 transition-colors" title={isExpanded ? "Collapse" : "Expand"}>
                            {isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                        </button>
                    )}
                    {onClose && (
                        <button onClick={onClose} className="p-1 text-cvat-text-secondary hover:text-cvat-text-primary rounded hover:bg-white/5 transition-colors">
                            <X size={14} />
                        </button>
                    )}
                </div>
            </div>

            <div className="flex-1 overflow-auto custom-scrollbar">
                <table className="w-full text-left text-xs text-cvat-text-secondary">
                    <thead className="text-xs text-cvat-text-primary uppercase bg-cvat-bg-tertiary sticky top-0 z-10">
                        <tr>
                            {displayHeaders.map((h) => (
                                <th key={h} scope="col" className="px-3 py-2 border-b border-cvat-border whitespace-nowrap">
                                    {h}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-cvat-border">
                        {displayData.map((row, i) => (
                            <tr
                                key={i}
                                onClick={() => onRowClick?.(row)}
                                className="hover:bg-cvat-bg-tertiary/50 cursor-pointer transition-colors"
                            >
                                {displayHeaders.map((h) => (
                                    <td key={`${i}-${h}`} className="px-3 py-2 whitespace-nowrap overflow-hidden text-ellipsis max-w-[150px]" title={String(row[h])}>
                                        {/* Basic safety for rendering objects */}
                                        {typeof row[h] === 'object' ? JSON.stringify(row[h]) : row[h]}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
