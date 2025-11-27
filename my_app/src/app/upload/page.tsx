"use client";

import { useState, useEffect } from "react";
import { Loader2, UploadCloud } from "lucide-react";
import DashboardNav from "@/components/DashboardNav";
import { apiFetch } from "@/lib/api";
import { useAuthGuard } from "@/hooks/use-auth-guard";
import { toast } from "react-hot-toast";

export default function UploadPage() {
  const { isLoading } = useAuthGuard();
  const [datasetName, setDatasetName] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [lastDatasetId, setLastDatasetId] = useState<string | null>(null);
  const [runAfterUpload, setRunAfterUpload] = useState(false);
  const [modelsList, setModelsList] = useState<any[]>([]);
  const [modelId, setModelId] = useState<string>("cellpose_default");
  const [diameter, setDiameter] = useState<string>("");
  const [channels, setChannels] = useState<string>("0,0");

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return;
    setSelectedFiles(Array.from(event.target.files));
  };

  const resetForm = () => {
    setDatasetName("");
    setSelectedFiles([]);
    setLastDatasetId(null);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!datasetName.trim()) {
      toast.error("Please provide a dataset name.");
      return;
    }

    if (selectedFiles.length === 0) {
      toast.error("Please select at least one image.");
      return;
    }

    setIsSubmitting(true);

    try {
      const formData = new FormData();
      formData.append("name", datasetName.trim());
        if (runAfterUpload) {
          formData.append("run_inference", "true");
          formData.append("model_id", modelId);
          formData.append("params", JSON.stringify({ diameter: diameter ? Number(diameter) : null, channels: channels.split(",").map((c) => Number(c.trim() || 0)) }));
        }
      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      const response = await apiFetch("/api/datasets/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Upload failed");
      }

      toast.success("Dataset uploaded successfully");
      setLastDatasetId(data.dataset_id);
      setSelectedFiles([]);
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Load available models for the run-inference option
  useEffect(() => {
    if (isLoading) return; // wait for auth guard to resolve before calling protected endpoints
    async function loadModels() {
      try {
        const resp = await apiFetch('/api/models/');
        if (!resp.ok) return;
        const data = await resp.json();
        setModelsList(data);
        if (data.length > 0) setModelId(data[0]._id);
      } catch (e) {
        console.warn('Failed to load models for upload page');
      }
    }
    loadModels();
  }, [isLoading]);

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
      <div className="max-w-3xl mx-auto px-4 py-10">
        <div className="cvat-card p-6">
          <div className="flex items-center gap-3 mb-6">
            <UploadCloud className="h-8 w-8 text-cvat-primary" />
            <div>
              <h1 className="text-2xl font-semibold text-cvat-text-primary">Upload Dataset</h1>
              <p className="text-sm text-cvat-text-secondary">
                Create a dataset of medical images for later inference.
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="dataset-name" className="block text-sm font-medium text-cvat-text-primary">
                Dataset Name
              </label>
              <input
                id="dataset-name"
                type="text"
                value={datasetName}
                onChange={(event) => setDatasetName(event.target.value)}
                className="mt-2 w-full rounded-lg border border-cvat-border bg-cvat-bg-secondary px-4 py-3 text-cvat-text-primary focus:border-cvat-primary focus:ring-cvat-primary"
                placeholder="e.g. Brain Tumor Batch A"
                disabled={isSubmitting}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-cvat-text-primary">Image Files</label>
              <div className="mt-2 border-2 border-dashed border-cvat-border rounded-lg p-6 text-center bg-cvat-bg-secondary">
                <input
                  id="files"
                  type="file"
                  accept=".png,.jpg,.jpeg,.tif,.tiff"
                  multiple
                  onChange={handleFileChange}
                  className="hidden"
                  disabled={isSubmitting}
                />
                <label htmlFor="files" className="cursor-pointer text-cvat-primary font-medium">
                  Click to browse
                </label>
                <p className="mt-2 text-sm text-cvat-text-secondary">
                  or drag and drop multiple files here
                </p>
                {selectedFiles.length > 0 && (
                  <p className="mt-3 text-sm text-cvat-text-primary">
                    {selectedFiles.length} file(s) selected
                  </p>
                )}
              </div>
            </div>

            <div className="flex gap-3">
              <label className="flex items-center gap-2 mr-4">
                <input type="checkbox" checked={runAfterUpload} onChange={(e) => setRunAfterUpload(e.target.checked)} />
                <span className="text-sm text-cvat-text-secondary">Run inference after upload</span>
              </label>
              {runAfterUpload && (
                <div className="flex items-center gap-3">
                  <select value={modelId} onChange={(e) => setModelId(e.target.value)} className="rounded border px-2 py-1">
                    {modelsList.map((m) => (
                      <option key={m._id} value={m._id}>{m.name}</option>
                    ))}
                  </select>
                  <input value={diameter} onChange={(e) => setDiameter(e.target.value)} placeholder="diameter" className="rounded border px-2 py-1 w-24" />
                  <input value={channels} onChange={(e) => setChannels(e.target.value)} placeholder="channels e.g. 0,0" className="rounded border px-2 py-1 w-32" />
                </div>
              )}
              <button
                type="submit"
                className="cvat-button-primary flex-1 py-3 rounded-lg font-medium"
                disabled={isSubmitting}
              >
                {isSubmitting ? (
                  <span className="flex items-center justify-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Uploading...
                  </span>
                ) : (
                  "Create Dataset"
                )}
              </button>
              <button
                type="button"
                onClick={resetForm}
                className="flex-1 border border-cvat-border rounded-lg text-cvat-text-secondary hover:text-cvat-text-primary hover:border-cvat-primary"
                disabled={isSubmitting}
              >
                Reset
              </button>
            </div>
          </form>

          {lastDatasetId && (
            <div className="mt-6 rounded-lg border border-cvat-border bg-cvat-bg-secondary p-4 text-sm text-cvat-text-secondary">
              Dataset ID: <span className="text-cvat-text-primary">{lastDatasetId}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

