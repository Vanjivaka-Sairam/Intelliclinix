"use client";

import { useEffect, useState } from "react";
import { Loader2, UploadCloud } from "lucide-react";
import { apiFetch } from "@/lib/api";
import { useAuthGuard } from "@/hooks/use-auth-guard";
import { toast } from "react-hot-toast";

type Model = {
  _id: string;
  name: string;
  description?: string;
  input_type?: string;
};

export default function UploadPage() {
  const { isLoading } = useAuthGuard();
  const [datasetName, setDatasetName] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [lastDatasetId, setLastDatasetId] = useState<string | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState("");

  // Derived: does the selected model require exactly 5 images?
  const selectedModelDef = models.find((m) => m._id === selectedModel);
  const requiresExact5 = selectedModelDef?.input_type === "grouped_5";

  useEffect(() => {
    if (isLoading) return;
    apiFetch("/api/models/")
      .then((res) => res.json())
      .then((data: Model[]) => setModels(data))
      .catch(() => toast.error("Failed to load models"));
  }, [isLoading]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return;
    setSelectedFiles(Array.from(event.target.files));
  };

  const resetForm = () => {
    setDatasetName("");
    setSelectedFiles([]);
    setLastDatasetId(null);
    setSelectedModel("");
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

    if (requiresExact5 && selectedFiles.length !== 5) {
      toast.error("This model requires exactly 5 images (DAPI, FITC, ORANGE, AQUA, SKY).");
      return;
    }

    setIsSubmitting(true);

    try {
      const formData = new FormData();
      formData.append("name", datasetName.trim());
      if (selectedModel) formData.append("model_id", selectedModel);
      selectedFiles.forEach((file) => formData.append("files", file));

      const response = await apiFetch("/api/datasets/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Upload failed");

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
            {/* Dataset Name */}
            <div>
              <label htmlFor="dataset-name" className="block text-sm font-medium text-cvat-text-primary">
                Dataset Name
              </label>
              <input
                id="dataset-name"
                type="text"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="mt-2 w-full rounded-lg border border-cvat-border bg-cvat-bg-secondary px-4 py-3 text-cvat-text-primary focus:border-cvat-primary focus:ring-cvat-primary"
                placeholder="e.g. AL_16_22_ABL1"
                disabled={isSubmitting}
                required
              />
            </div>

            {/* Model Dropdown */}
            <div>
              <label htmlFor="model-select" className="block text-sm font-medium text-cvat-text-primary">
                Model <span className="text-cvat-text-secondary font-normal">(optional)</span>
              </label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => { setSelectedModel(e.target.value); setSelectedFiles([]); }}
                className="mt-2 w-full rounded-lg border border-cvat-border bg-cvat-bg-secondary px-4 py-3 text-cvat-text-primary focus:border-cvat-primary focus:ring-cvat-primary"
                disabled={isSubmitting}
              >
                <option value="">— No model selected —</option>
                {models.map((model) => (
                  <option key={model._id} value={model._id}>
                    {model.name}
                  </option>
                ))}
              </select>
              {selectedModelDef?.description && (
                <p className="mt-1 text-xs text-cvat-text-secondary">{selectedModelDef.description}</p>
              )}
            </div>

            {/* File Picker */}
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

                {requiresExact5 ? (
                  <p className="mt-2 text-sm text-cvat-text-secondary">
                    Select <strong>exactly 5</strong> channel images: DAPI, FITC, ORANGE, AQUA, SKY
                  </p>
                ) : (
                  <p className="mt-2 text-sm text-cvat-text-secondary">
                    or drag and drop multiple files here
                  </p>
                )}

                {selectedFiles.length > 0 && (
                  <p className={`mt-3 text-sm font-medium ${requiresExact5 && selectedFiles.length !== 5 ? "text-red-400" : "text-cvat-text-primary"}`}>
                    {selectedFiles.length} file(s) selected
                    {requiresExact5 && selectedFiles.length !== 5 && " — need exactly 5"}
                  </p>
                )}
              </div>
            </div>

            <div className="flex gap-3">
              <button
                type="submit"
                className="cvat-button-primary flex-1 py-3 rounded-lg font-medium"
                disabled={isSubmitting || (requiresExact5 && selectedFiles.length !== 5)}
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
