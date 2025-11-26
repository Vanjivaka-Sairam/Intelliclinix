'use client';

import Link from "next/link";
import { Upload, Brain, ListChecks, ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";
import DashboardNav from "@/components/DashboardNav";
import { apiFetch } from "@/lib/api";
import { useAuthGuard } from "@/hooks/use-auth-guard";

type DatasetSummary = {
  _id: string;
  name: string;
  files?: Array<{ filename: string }>;
};

type InferenceSummary = {
  _id: string;
  status: string;
  dataset_id: string;
  created_at: string;
};

export default function Home() {
  const { isLoading } = useAuthGuard();
  const [datasetCount, setDatasetCount] = useState(0);
  const [inferenceCount, setInferenceCount] = useState(0);
  const [recentInferences, setRecentInferences] = useState<InferenceSummary[]>([]);

  useEffect(() => {
    if (isLoading) return;

    async function loadSummary() {
      try {
        const [datasetsRes, inferencesRes] = await Promise.all([
          apiFetch("/api/datasets/"),
          apiFetch("/api/inferences/"),
        ]);

        if (datasetsRes.ok) {
          const datasets: DatasetSummary[] = await datasetsRes.json();
          setDatasetCount(datasets.length);
        }

        if (inferencesRes.ok) {
          const inferences: InferenceSummary[] = await inferencesRes.json();
          setInferenceCount(inferences.length);
          setRecentInferences(inferences.slice(0, 5));
        }
      } catch (error) {
        console.error("Failed to load dashboard summary", error);
      }
    }

    loadSummary();
  }, [isLoading]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-cvat-bg-primary">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cvat-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-cvat-bg-primary">
      <DashboardNav />
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-cvat-text-primary">Welcome back</h1>
          <p className="text-cvat-text-secondary">
            Manage datasets, launch inference jobs, and review results in one place.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          {[
            {
              href: "/upload",
              title: "Upload Data",
              description: "Create a dataset with medical images.",
              Icon: Upload,
            },
            {
              href: "/inference",
              title: "Run Inference",
              description: "Pick a dataset and model to start analysis.",
              Icon: Brain,
            },
            {
              href: "/results",
              title: "View Results",
              description: "Review completed jobs and download outputs.",
              Icon: ListChecks,
            },
          ].map(({ href, title, description, Icon }) => (
            <Link
              key={href}
              href={href}
              className="cvat-card p-6 flex flex-col justify-between hover:shadow-lg transition-shadow"
            >
              <div>
                <div className="flex items-center justify-between">
                  <Icon className="h-8 w-8 text-cvat-primary" />
                  <ArrowRight className="h-5 w-5 text-cvat-text-tertiary" />
                </div>
                <h2 className="mt-4 text-xl font-semibold text-cvat-text-primary">{title}</h2>
                <p className="mt-2 text-sm text-cvat-text-secondary">{description}</p>
              </div>
            </Link>
          ))}
        </div>

        <div className="mt-10 grid gap-6 md:grid-cols-3">
          <div className="cvat-card p-6">
            <p className="text-sm text-cvat-text-secondary">Datasets</p>
            <p className="text-4xl font-bold text-cvat-text-primary mt-2">{datasetCount}</p>
            <p className="text-sm text-cvat-text-tertiary mt-1">
              Total datasets stored in your workspace
            </p>
          </div>
          <div className="cvat-card p-6">
            <p className="text-sm text-cvat-text-secondary">Inference Jobs</p>
            <p className="text-4xl font-bold text-cvat-text-primary mt-2">{inferenceCount}</p>
            <p className="text-sm text-cvat-text-tertiary mt-1">
              Jobs submitted across all datasets
            </p>
          </div>
          <div className="cvat-card p-6">
            <p className="text-sm text-cvat-text-secondary">CVAT Integration</p>
            <p className="text-lg font-medium text-cvat-text-primary mt-2">
              Export any result to CVAT for corrections.
            </p>
            <p className="text-sm text-cvat-text-tertiary mt-1">
              Use the “Send to CVAT” button in each inference.
            </p>
          </div>
        </div>

        <div className="mt-10">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-cvat-text-primary">Recent Inference Jobs</h2>
            <Link href="/results" className="text-sm text-cvat-primary hover:underline">
              View all
            </Link>
          </div>
          {recentInferences.length === 0 ? (
            <div className="cvat-card p-6 text-cvat-text-secondary">
              No inference jobs yet. Start by uploading a dataset.
            </div>
          ) : (
            <div className="cvat-card divide-y divide-cvat-border">
              {recentInferences.map((inference) => (
                <Link
                  key={inference._id}
                  href={`/results/${inference._id}`}
                  className="flex items-center justify-between px-6 py-4 hover:bg-cvat-bg-tertiary transition-colors"
                >
                  <div>
                    <p className="text-cvat-text-primary font-medium">Job #{inference._id}</p>
                    <p className="text-sm text-cvat-text-secondary">
                      Dataset: {inference.dataset_id}
                    </p>
                  </div>
                  <span
                    className={`text-xs font-semibold px-3 py-1 rounded-full ${
                      inference.status === "completed"
                        ? "bg-emerald-100 text-emerald-700"
                        : inference.status === "failed"
                        ? "bg-rose-100 text-rose-700"
                        : "bg-amber-100 text-amber-700"
                    }`}
                  >
                    {inference.status.toUpperCase()}
                  </span>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}