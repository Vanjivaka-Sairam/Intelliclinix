"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Loader2, Download } from "lucide-react";
import DashboardNav from "@/components/DashboardNav";
import { apiFetch } from "@/lib/api";
import { toast } from "react-hot-toast";
import { useAuthGuard } from "@/hooks/use-auth-guard";

type InferenceRecord = {
  _id: string;
  dataset_id: string;
  status: string;
  created_at: string;
};

export default function ResultsPage() {
  const { isLoading } = useAuthGuard();
  const [records, setRecords] = useState<InferenceRecord[]>([]);
  const [isDownloading, setIsDownloading] = useState<string | null>(null);

  useEffect(() => {
    if (isLoading) return;

    async function loadResults() {
      try {
        const response = await apiFetch("/api/inferences/");
        if (!response.ok) {
          throw new Error("Failed to load results");
        }
        const data: InferenceRecord[] = await response.json();
        setRecords(data);
      } catch (error) {
        console.error(error);
        toast.error("Unable to load results");
      }
    }

    loadResults();
  }, [isLoading]);

  const downloadZip = async (id: string) => {
    setIsDownloading(id);
    try {
      const response = await apiFetch(`/api/inferences/${id}/download`);
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "Download failed");
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `inference_${id}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "Download failed");
    } finally {
      setIsDownloading(null);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-cvat-bg-primary">
        <Loader2 className="h-6 w-6 animate-spin text-cvat-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-cvat-bg-primary">
      <DashboardNav />
      <div className="max-w-6xl mx-auto px-4 py-10">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-semibold text-cvat-text-primary">Inference Results</h1>
          <p className="text-sm text-cvat-text-secondary">
            Monitor status, download outputs, or open detailed views.
          </p>
        </div>

        {records.length === 0 ? (
          <div className="cvat-card p-6 text-center text-cvat-text-secondary">
            No inference jobs available yet.
          </div>
        ) : (
          <div className="cvat-card overflow-hidden">
            <table className="min-w-full divide-y divide-cvat-border">
              <thead className="bg-cvat-bg-tertiary">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-cvat-text-secondary uppercase tracking-wider">
                    Job
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-cvat-text-secondary uppercase tracking-wider">
                    Dataset
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-cvat-text-secondary uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-cvat-text-secondary uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-cvat-border bg-white">
                {records.map((record) => (
                  <tr key={record._id}>
                    <td className="px-6 py-4 text-sm font-medium text-cvat-text-primary">
                      <Link
                        href={`/results/${record._id}`}
                        className="text-cvat-primary hover:underline"
                      >
                        {record._id}
                      </Link>
                    </td>
                    <td className="px-6 py-4 text-sm text-cvat-text-secondary">{record.dataset_id}</td>
                    <td className="px-6 py-4">
                      <span
                        className={`text-xs font-semibold px-3 py-1 rounded-full ${
                          record.status === "completed"
                            ? "bg-emerald-100 text-emerald-700"
                            : record.status === "failed"
                            ? "bg-rose-100 text-rose-700"
                            : "bg-amber-100 text-amber-700"
                        }`}
                      >
                        {record.status.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right text-sm">
                      <button
                        onClick={() => downloadZip(record._id)}
                        className="inline-flex items-center gap-2 rounded-md border border-cvat-border px-3 py-2 text-cvat-text-secondary hover:border-cvat-primary hover:text-cvat-primary disabled:opacity-50"
                        disabled={isDownloading === record._id}
                      >
                        {isDownloading === record._id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Download className="h-4 w-4" />
                        )}
                        Download
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

