"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Loader2,
  Download,
  Trash2,
  Archive,
  RefreshCcw, // Icon for Restore
  Filter,
  CheckSquare
} from "lucide-react";
import { apiFetch } from "@/lib/api";
import { toast } from "react-hot-toast";
import { useAuthGuard } from "@/hooks/use-auth-guard";

type InferenceRecord = {
  _id: string;
  dataset_id: string;
  status: string;
  created_at: string;
};

export default function ArchivePage() {
  const { isLoading } = useAuthGuard();
  const [records, setRecords] = useState<InferenceRecord[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  // State for metadata lookup
  const [modelsList, setModelsList] = useState<any[]>([]);
  const [datasetMap, setDatasetMap] = useState<Record<string, string>>({});

  const [modelFilter, setModelFilter] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState<string | null>(null);

  // 1. Load Archived Results
  useEffect(() => {
    if (isLoading) return;

    async function loadArchived() {
      try {
        const query = modelFilter ? `?archived=true&model_id=${encodeURIComponent(modelFilter)}` : `?archived=true`;
        const response = await apiFetch(`/api/inferences/${query}`);
        if (!response.ok) {
          throw new Error("Failed to load archive");
        }
        const data: InferenceRecord[] = await response.json();
        setRecords(data);
      } catch (error) {
        console.error(error);
        toast.error("Unable to load archived inferences");
      }
    }
    loadArchived();
  }, [isLoading, modelFilter]);

  // 2. Load Models & Datasets for Lookup
  useEffect(() => {
    if (isLoading) return;
    async function loadMetaData() {
      // Fetch Models
      try {
        const resp = await apiFetch('/api/models/');
        if (resp.ok) setModelsList(await resp.json());
      } catch (e) { console.warn('Failed to load models'); }

      // Fetch Datasets
      try {
        const resp = await apiFetch('/api/datasets/');
        if (resp.ok) {
          const data = await resp.json();
          const map: Record<string, string> = {};
          data.forEach((ds: any) => { map[ds._id] = ds.name; });
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

  const handleBulkRestore = async () => {
    try {
      const resp = await apiFetch('/api/inferences/archive', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: selectedIds, archived: false })
      });
      if (!resp.ok) throw new Error('Unarchive failed');

      toast.success(`Restored ${selectedIds.length} inferences`);
      setRecords((r) => r.filter((rec) => !selectedIds.includes(rec._id)));
      setSelectedIds([]);
    } catch (e) {
      console.error(e);
      toast.error('Failed to restore selected items');
    }
  };

  const handleBulkDelete = async () => {
    if (!confirm("This will permanently delete the selected records. This cannot be undone.")) return;
    try {
      for (const id of selectedIds) {
        await apiFetch(`/api/inferences/${id}`, { method: 'DELETE' });
      }
      toast.success('Permanently deleted selected inferences');
      setRecords((r) => r.filter((rec) => !selectedIds.includes(rec._id)));
      setSelectedIds([]);
    } catch (e) {
      console.error(e);
      toast.error('Failed to delete selected items');
    }
  };

  const handleRestoreSingle = async (id: string) => {
    try {
      const resp = await apiFetch('/api/inferences/archive', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: [id], archived: false })
      });
      if (!resp.ok) throw new Error('Unarchive failed');
      toast.success('Inference restored to active list');
      setRecords((r) => r.filter((x) => x._id !== id));
    } catch (e) {
      console.error(e);
      toast.error('Failed to restore');
    }
  };

  const handleDeleteSingle = async (id: string) => {
    if (!confirm("Permanently delete this record?")) return;
    try {
      const resp = await apiFetch(`/api/inferences/${id}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error('Delete failed');
      toast.success('Inference permanently deleted');
      setRecords((r) => r.filter((rec) => rec._id !== id));
    } catch (e) {
      console.error(e);
      toast.error('Failed to delete');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-50">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">

        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Archived Inferences</h1>
            <p className="mt-1 text-slate-500">Restore items to the main dashboard or permanently delete them.</p>
          </div>
          <div className="mt-4 md:mt-0">
            <Link
              href="/results"
              className="inline-flex items-center text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors"
            >
              ← Back to Active Results
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">

          {/* LEFT: Results Column (Span 3) */}
          <div className="lg:col-span-3">
            {records.length === 0 ? (
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-12 text-center">
                <div className="mx-auto h-12 w-12 text-slate-400 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                  <Archive className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-medium text-slate-900">Archive is empty</h3>
                <p className="mt-1 text-slate-500">No archived inferences found matching your criteria.</p>
              </div>
            ) : (
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
                        <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Job ID</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Dataset</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-3 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">Actions</th>
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
                                  current.includes(record._id) ? current.filter((x) => x !== record._id) : [...current, record._id]
                                )
                              }
                            />
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className="text-sm font-medium text-slate-700 font-mono">
                              {record._id.substring(0, 8)}...
                            </span>
                          </td>
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
                                {isDownloading === record._id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                              </button>

                              <button
                                onClick={() => handleRestoreSingle(record._id)}
                                className="text-slate-400 hover:text-emerald-600 transition-colors p-1"
                                title="Restore"
                              >
                                <RefreshCcw className="h-4 w-4" />
                              </button>

                              <button
                                onClick={() => handleDeleteSingle(record._id)}
                                className="text-slate-400 hover:text-rose-600 transition-colors p-1"
                                title="Delete Permanently"
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
            )}
          </div>

          {/* RIGHT: Sticky Sidebar (Span 1) */}
          <div className="lg:col-span-1">
            <div className="sticky top-6 space-y-6">

              {/* Filter */}
              <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                <div className="flex items-center gap-2 mb-4 text-slate-900 font-semibold">
                  <Filter className="w-4 h-4 text-slate-500" />
                  <h2>Filter Archive</h2>
                </div>
                <label className="block text-xs font-medium text-slate-500 mb-1.5 uppercase">By Model</label>
                <select
                  value={modelFilter ?? ''}
                  onChange={(e) => setModelFilter(e.target.value || null)}
                  className="w-full rounded-lg border-slate-200 text-sm focus:border-blue-500 focus:ring-blue-500"
                >
                  <option value="">All models</option>
                  {modelsList.map((m) => (
                    <option key={m._id} value={m._id}>{m.name}</option>
                  ))}
                </select>
              </div>

              {/* Bulk Actions */}
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
                    onClick={handleBulkRestore}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-white bg-emerald-600 rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-all"
                  >
                    <RefreshCcw className="w-4 h-4" />
                    Restore Selected
                  </button>

                  <button
                    disabled={selectedIds.length === 0}
                    onClick={handleBulkDelete}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-rose-600 bg-rose-50 border border-rose-200 rounded-lg hover:bg-rose-100 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete Forever
                  </button>
                </div>
              </div>

            </div>
          </div>

        </div>
      </main>
    </div>
  );
}