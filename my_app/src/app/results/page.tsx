"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Loader2,
  Download,
  Trash2,
  Archive,
  Filter,
  CheckSquare
} from "lucide-react";
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
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  // New State for Models and Datasets Lookup
  const [modelsList, setModelsList] = useState<any[]>([]);
  const [datasetMap, setDatasetMap] = useState<Record<string, string>>({});

  const [modelFilter, setModelFilter] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState<string | null>(null);
  const [isFetching, setIsFetching] = useState(true);

  // 1. Load Results
  useEffect(() => {
    if (isLoading) return;

    async function loadResults() {
      setIsFetching(true);
      try {
        const query = modelFilter ? `?model_id=${encodeURIComponent(modelFilter)}` : "";
        const response = await apiFetch(`/api/inferences/${query}`);
        if (!response.ok) throw new Error("Failed to load results");
        const data: InferenceRecord[] = await response.json();
        setRecords(data);
      } catch (error) {
        console.error(error);
        toast.error("Unable to load results");
      } finally {
        setIsFetching(false);
      }
    }

    loadResults();
  }, [isLoading, modelFilter]);
  // 2. Load Models (for Filter) AND Datasets (for Names)
  useEffect(() => {
    if (isLoading) return;

    async function loadMetaData() {
      // Fetch Models
      try {
        const resp = await apiFetch('/api/models/');
        if (resp.ok) setModelsList(await resp.json());
      } catch (e) { console.warn('Failed to load models'); }

      // Fetch Datasets & Create Lookup Map
      try {
        const resp = await apiFetch('/api/datasets/'); // Assuming this endpoint exists
        if (resp.ok) {
          const data = await resp.json();
          // Transform array [{_id: '123', name: 'My Data'}] into object {'123': 'My Data'}
          const map: Record<string, string> = {};
          data.forEach((ds: any) => {
            map[ds._id] = ds.name;
          });
          setDatasetMap(map);
        }
      } catch (e) { console.warn('Failed to load datasets'); }
    }

    loadMetaData();
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

  const handleBulkArchive = async () => {
    try {
      const response = await apiFetch(`/api/inferences/archive`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: selectedIds, archived: true }),
      });
      if (!response.ok) throw new Error("Archive failed");
      toast.success(`Archived ${selectedIds.length} inferences`);

      const q = modelFilter ? `?model_id=${encodeURIComponent(modelFilter)}` : "";
      const data = await apiFetch(`/api/inferences/${q}`);
      setRecords(await data.json());
      setSelectedIds([]);
    } catch (e) {
      console.error(e);
      toast.error("Failed to archive selected inferences");
    }
  };

  const handleBulkDelete = async () => {
    if (!confirm("Are you sure you want to delete these records?")) return;
    try {
      for (const id of selectedIds) {
        await apiFetch(`/api/inferences/${id}`, { method: "DELETE" });
      }
      toast.success("Deleted selected inferences");
      setRecords((r) => r.filter((rec) => !selectedIds.includes(rec._id)));
      setSelectedIds([]);
    } catch (e) {
      console.error(e);
      toast.error("Failed to delete selected inferences");
    }
  };

  const handleDeleteSingle = async (id: string) => {
    if (!confirm("Permanently delete this inference? This action cannot be undone.")) return;

    try {
      const resp = await apiFetch(`/api/inferences/${id}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error('Delete failed');
      toast.success('Inference deleted');
      setRecords((r) => r.filter((rec) => rec._id !== id));
    } catch (e) {
      console.error(e);
      toast.error('Failed to delete inference');
    }
  };
  if (isLoading || isFetching) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-50">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <DashboardNav />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Inference Results</h1>
            <p className="mt-1 text-slate-500">Manage your model outputs and download annotations.</p>
          </div>
          <div className="mt-4 md:mt-0">
            <Link
              href="/results/archive"
              className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors"
            >
              <Archive className="w-4 h-4 mr-2" />
              View Archive
            </Link>
          </div>
        </div>

        {records.length === 0 ? (
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-12 text-center">
            <div className="mx-auto h-12 w-12 text-slate-400 bg-slate-100 rounded-full flex items-center justify-center mb-4">
              <Filter className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-medium text-slate-900">No results found</h3>
            <p className="mt-1 text-slate-500">No inference jobs match your current criteria.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            <div className="lg:col-span-3">
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-slate-200">
                    <thead className="bg-slate-50">
                      <tr>
                        <th scope="col" className="px-6 py-4 text-left">
                          <input
                            type="checkbox"
                            className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                            checked={selectedIds.length > 0 && selectedIds.length === records.length}
                            onChange={(e) => {
                              if (e.target.checked) setSelectedIds(records.map((r) => r._id));
                              else setSelectedIds([]);
                            }}
                          />
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Job ID</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Dataset Name</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Status</th>
                        <th scope="col" className="px-6 py-3 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-slate-200">
                      {records.map((record) => (
                        <tr key={record._id} className={`hover:bg-slate-50 transition-colors ${selectedIds.includes(record._id) ? 'bg-blue-50/50' : ''}`}>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <input
                              type="checkbox"
                              className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                              checked={selectedIds.includes(record._id)}
                              onChange={() =>
                                setSelectedIds((current) =>
                                  current.includes(record._id)
                                    ? current.filter((x) => x !== record._id)
                                    : [...current, record._id]
                                )
                              }
                            />
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <Link
                              href={`/results/inferences?id=${record._id}`}
                              className="text-sm font-medium text-blue-600 hover:text-blue-800 hover:underline font-mono"
                            >
                              {record._id.substring(0, 8)}...
                            </Link>
                          </td>

                          {/* UPDATED: Dataset Name with ID fallback */}
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-900 font-medium">
                            {datasetMap[record.dataset_id] || (
                              <span className="text-slate-400 font-mono text-xs">
                                {record.dataset_id.substring(0, 12)}...
                              </span>
                            )}
                          </td>

                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                              ${record.status === "completed" ? "bg-emerald-100 text-emerald-800" :
                                record.status === "failed" ? "bg-rose-100 text-rose-800" :
                                  "bg-amber-100 text-amber-800"}`}>
                              {record.status.toUpperCase()}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                            <div className="flex items-center justify-end gap-2">
                              <button
                                onClick={() => downloadZip(record._id)}
                                disabled={isDownloading === record._id}
                                className="text-slate-400 hover:text-blue-600 transition-colors p-1"
                                title="Download"
                              >
                                {isDownloading === record._id ? (
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                  <Download className="h-4 w-4" />
                                )}
                              </button>
                              <button
                                onClick={() => handleDeleteSingle(record._id)}
                                className="text-slate-400 hover:text-rose-600 transition-colors p-1"
                                title="Delete"
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="lg:col-span-1">
              <div className="sticky top-6 space-y-6">

                <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                  <div className="flex items-center gap-2 mb-4 text-slate-900 font-semibold">
                    <Filter className="w-4 h-4 text-slate-500" />
                    <h2>Filter Results</h2>
                  </div>
                  <label className="block text-xs font-medium text-slate-500 mb-1.5 uppercase">By Model</label>
                  <select
                    value={modelFilter ?? ''}
                    onChange={(e) => setModelFilter(e.target.value || null)}
                    className="w-full rounded-lg border-slate-200 text-sm focus:border-blue-500 focus:ring-blue-500"
                  >
                    <option value="">Show All Models</option>
                    {modelsList.map((m) => (
                      <option key={m._id} value={m._id}>{m.name}</option>
                    ))}
                  </select>
                </div>

                <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2 text-slate-900 font-semibold">
                      <CheckSquare className="w-4 h-4 text-slate-500" />
                      <h2>Bulk Actions</h2>
                    </div>
                    {selectedIds.length > 0 && (
                      <span className="bg-blue-100 text-blue-700 text-xs font-bold px-2 py-0.5 rounded-full">
                        {selectedIds.length}
                      </span>
                    )}
                  </div>

                  <div className="space-y-3">
                    <button
                      disabled={selectedIds.length === 0}
                      onClick={handleBulkArchive}
                      className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      <Archive className="w-4 h-4" />
                      Archive Selected
                    </button>

                    <button
                      disabled={selectedIds.length === 0}
                      onClick={handleBulkDelete}
                      className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-white bg-rose-600 rounded-lg hover:bg-rose-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-all"
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete Selected
                    </button>
                  </div>
                </div>

              </div>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}