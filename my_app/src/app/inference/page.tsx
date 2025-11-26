"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "react-hot-toast";
import { Loader2 } from "lucide-react";
import DashboardNav from "@/components/DashboardNav";
import { apiFetch } from "@/lib/api";
import { useAuthGuard } from "@/hooks/use-auth-guard";

type Dataset = {
  _id: string;
  name: string;
};

type Model = {
  _id: string;
  name: string;
  runner_name?: string;
  description?: string;
};

export default function InferencePage() {
  const router = useRouter();
  const { isLoading } = useAuthGuard();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (isLoading) return;

    async function loadOptions() {
      try {
        const [datasetRes, modelRes] = await Promise.all([
          apiFetch("/api/datasets/"),
          apiFetch("/api/models/"),
        ]);

        if (datasetRes.ok) {
          const datasetData: Dataset[] = await datasetRes.json();
          setDatasets(datasetData);
        }

        if (modelRes.ok) {
          const modelData: Model[] = await modelRes.json();
          setModels(modelData);
        }
      } catch (error) {
        console.error(error);
        toast.error("Failed to load datasets or models");
      }
    }

    loadOptions();
  }, [isLoading]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!selectedDataset || !selectedModel) {
      toast.error("Select both dataset and model.");
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await apiFetch("/api/inferences/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDataset,
          model_id: selectedModel,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Unable to start inference");
      }

      toast.success("Inference started");
      router.push(`/results/${data.inference_id}`);
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "Inference failed");
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
      <DashboardNav />
      <div className="max-w-3xl mx-auto px-4 py-10">
        <div className="cvat-card p-6">
          <h1 className="text-2xl font-semibold text-cvat-text-primary mb-6">
            Run Inference
          </h1>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-cvat-text-primary">
                Select Dataset
              </label>
              <select
                value={selectedDataset}
                onChange={(event) => setSelectedDataset(event.target.value)}
                className="mt-2 w-full rounded-lg border border-cvat-border bg-cvat-bg-secondary px-4 py-3 text-cvat-text-primary focus:border-cvat-primary focus:ring-cvat-primary"
                disabled={isSubmitting || datasets.length === 0}
              >
                <option value="">Choose a dataset</option>
                {datasets.map((dataset) => (
                  <option key={dataset._id} value={dataset._id}>
                    {dataset.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-cvat-text-primary">
                Select Model
              </label>
              <select
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                className="mt-2 w-full rounded-lg border border-cvat-border bg-cvat-bg-secondary px-4 py-3 text-cvat-text-primary focus:border-cvat-primary focus:ring-cvat-primary"
                disabled={isSubmitting || models.length === 0}
              >
                <option value="">Choose a model</option>
                {models.map((model) => (
                  <option key={model._id} value={model._id}>
                    {model.name} {model.runner_name ? `(${model.runner_name})` : ""}
                  </option>
                ))}
              </select>
              {selectedModel && (
                <p className="mt-2 text-sm text-cvat-text-secondary">
                  {models.find((model) => model._id === selectedModel)?.description ||
                    "This model does not include a description."}
                </p>
              )}
            </div>

            <button
              type="submit"
              className="cvat-button-primary w-full py-3 rounded-lg font-medium"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Starting...
                </span>
              ) : (
                "Start Analysis"
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

