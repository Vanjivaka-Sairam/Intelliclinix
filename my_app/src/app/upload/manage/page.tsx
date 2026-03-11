"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Loader2, Trash2, Database, AlertCircle, ArrowLeft } from "lucide-react";
import { apiFetch } from "@/lib/api";
import { useAuthGuard } from "@/hooks/use-auth-guard";
import { toast } from "react-hot-toast";

type DatasetSummary = {
  _id: string;
  name: string;
  created_at: string;
  files?: Array<{ filename: string }>;
};

export default function ManageDatasetsPage() {
  const { isLoading } = useAuthGuard();
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [isDeleting, setIsDeleting] = useState<string | null>(null);

  const [datasetToDelete, setDatasetToDelete] = useState<DatasetSummary | null>(null);

  const loadDatasets = async () => {
    try {
      const res = await apiFetch("/api/datasets/");
      if (!res.ok) throw new Error("Failed to fetch datasets");
      const data: DatasetSummary[] = await res.json();
      setDatasets(data);
    } catch (e) {
      toast.error("Error loading datasets");
      setDatasets([]);
    }
  };

  useEffect(() => {
    if (isLoading) return;
    loadDatasets();
  }, [isLoading]);

  const confirmDelete = async () => {
    if (!datasetToDelete) return;

    setIsDeleting(datasetToDelete._id);
    try {
      const res = await apiFetch(`/api/datasets/${datasetToDelete._id}`, {
        method: "DELETE",
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to delete dataset");
      
      toast.success("Dataset deleted successfully");
      setDatasets((prev) => prev?.filter((d) => d._id !== datasetToDelete._id) || []);
    } catch (e: any) {
      toast.error(e.message || "Failed to delete dataset");
    } finally {
      setIsDeleting(null);
      setDatasetToDelete(null);
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
      <div className="max-w-5xl mx-auto px-4 py-10">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Database className="h-8 w-8 text-cvat-primary" />
            <div>
              <h1 className="text-2xl font-semibold text-cvat-text-primary">Manage Datasets</h1>
              <p className="text-sm text-cvat-text-secondary">
                View and remove datasets from your workspace.
              </p>
            </div>
          </div>
          <Link
            href="/upload"
            className="flex items-center text-sm font-medium text-cvat-text-secondary hover:text-cvat-primary transition-colors"
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to Upload
          </Link>
        </div>

        <div className="cvat-card overflow-hidden">
          {datasets === null ? (
            <div className="p-10 flex justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-cvat-primary" />
            </div>
          ) : datasets.length === 0 ? (
            <div className="p-12 text-center text-cvat-text-secondary flex flex-col items-center">
              <AlertCircle className="h-10 w-10 mb-3 opacity-20" />
              <p>You have no datasets in your workspace.</p>
              <Link href="/upload" className="mt-4 cvat-button-primary px-4 py-2 inline-block">
                Upload your first dataset
              </Link>
            </div>
          ) : (
            <div className="divide-y divide-cvat-border">
              {datasets.map((dataset) => (
                <div key={dataset._id} className="p-6 flex items-center justify-between hover:bg-cvat-bg-tertiary transition-colors">
                  <div>
                    <h3 className="text-lg font-medium text-cvat-text-primary mb-1">
                      {dataset.name}
                    </h3>
                    <div className="flex gap-4 text-xs text-cvat-text-secondary">
                      <span>ID: {dataset._id}</span>
                      <span>•</span>
                      <span>{dataset.files?.length || 0} file(s)</span>
                      <span>•</span>
                      <span>
                        Created: {dataset.created_at ? new Date(dataset.created_at).toLocaleDateString() : 'Unknown'}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => setDatasetToDelete(dataset)}
                    disabled={isDeleting === dataset._id}
                    className="p-2 text-cvat-text-secondary hover:text-white hover:bg-cvat-error rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed group relative"
                    title="Delete Dataset"
                  >
                    {isDeleting === dataset._id ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Trash2 className="h-5 w-5" />
                    )}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Warning Modal */}
      {datasetToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
          <div className="bg-cvat-bg-secondary border border-cvat-border rounded-xl shadow-2xl max-w-md w-full overflow-hidden animate-in fade-in zoom-in-95 duration-200">
            <div className="p-6">
              <div className="flex items-center gap-3 text-cvat-error mb-4">
                <AlertCircle className="h-8 w-8" />
                <h2 className="text-xl font-bold">Delete Dataset?</h2>
              </div>
              <p className="text-cvat-text-primary mb-2">
                You are about to delete <strong>{datasetToDelete.name}</strong>.
              </p>
              <div className="bg-cvat-error/10 border border-cvat-error/20 p-4 rounded-lg text-sm text-cvat-error mb-6">
                <strong>Warning:</strong> Deleting this dataset will also permanently delete <strong>all associated inferences and results</strong>. This action cannot be undone.
              </div>
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setDatasetToDelete(null)}
                  disabled={!!isDeleting}
                  className="px-4 py-2 border border-cvat-border rounded-lg text-cvat-text-secondary hover:text-cvat-text-primary transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmDelete}
                  disabled={!!isDeleting}
                  className="px-4 py-2 bg-cvat-error text-white rounded-lg hover:bg-red-600 transition-colors flex items-center gap-2 disabled:opacity-50"
                >
                  {isDeleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                  Yes, Delete Everything
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
