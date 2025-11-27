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

export default function ArchivePage() {
  const { isLoading } = useAuthGuard();
  const [records, setRecords] = useState<InferenceRecord[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [isDownloading, setIsDownloading] = useState<string | null>(null);
  const [modelsList, setModelsList] = useState<any[]>([]);
  const [modelFilter, setModelFilter] = useState<string | null>(null);

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

  useEffect(() => {
    if (isLoading) return;
    async function loadModels() {
      try {
        const resp = await apiFetch('/api/models/');
        if (!resp.ok) return;
        const data = await resp.json();
        setModelsList(data);
      } catch (e) {
        console.warn('Failed to load models');
      }
    }
    loadModels();
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
          <h1 className="text-2xl font-semibold text-cvat-text-primary">Archived Inferences</h1>
          <p className="text-sm text-cvat-text-secondary">Previously archived inferences can be restored or deleted permanently.</p>
        </div>

        {records.length === 0 ? (
          <div className="cvat-card p-6 text-center text-cvat-text-secondary">No archived inferences found.</div>
        ) : (
          <div className="cvat-card overflow-hidden md:flex md:items-stretch">
            <div className="md:flex-1 md:overflow-auto">
              <table className="min-w-full divide-y divide-cvat-border">
              <thead className="bg-cvat-bg-tertiary">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-cvat-text-secondary uppercase tracking-wider">
                    <input
                      type="checkbox"
                      checked={selectedIds.length > 0 && selectedIds.length === records.length}
                      onChange={(e) => (e.target.checked ? setSelectedIds(records.map((r) => r._id)) : setSelectedIds([]))}
                    />
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-cvat-text-secondary uppercase tracking-wider">Job</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-cvat-text-secondary uppercase tracking-wider">Dataset</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-cvat-text-secondary uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-cvat-text-secondary uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-cvat-border bg-white">
                {records.map((record) => (
                  <tr key={record._id}>
                    <td className="px-6 py-4 text-sm">
                      <input
                        type="checkbox"
                        checked={selectedIds.includes(record._id)}
                        onChange={() =>
                          setSelectedIds((current) => (current.includes(record._id) ? current.filter((x) => x !== record._id) : [...current, record._id]))
                        }
                      />
                    </td>
                    <td className="px-6 py-4 text-sm font-medium text-cvat-text-primary">
                      <Link href={`/results/${record._id}`} className="text-cvat-primary hover:underline">{record._id}</Link>
                    </td>
                    <td className="px-6 py-4 text-sm text-cvat-text-secondary">{record.dataset_id}</td>
                    <td className="px-6 py-4">
                      <span className={`text-xs font-semibold px-3 py-1 rounded-full ${record.status === "completed" ? "bg-emerald-100 text-emerald-700" : record.status === "failed" ? "bg-rose-100 text-rose-700" : "bg-amber-100 text-amber-700"}`}>
                        {record.status.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right text-sm flex gap-2 justify-end">
                      <button onClick={() => downloadZip(record._id)} className="inline-flex items-center gap-2 rounded-md border border-cvat-border px-3 py-2 text-cvat-text-secondary hover:border-cvat-primary hover:text-cvat-primary disabled:opacity-50" disabled={isDownloading === record._id}>
                        {isDownloading === record._id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />} Download
                      </button>
                      <button
                        onClick={async () => {
                          try {
                            const resp = await apiFetch('/api/inferences/archive', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids: [record._id], archived: false }) });
                            if (!resp.ok) throw new Error('Unarchive failed');
                            toast.success('Inference restored');
                            setRecords((r) => r.filter((x) => x._id !== record._id));
                          } catch (e) {
                            console.error(e);
                            toast.error('Failed to unarchive');
                          }
                        }}
                        className="inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm"
                      >
                        Restore
                      </button>
                      <button
                        onClick={async () => {
                          try {
                            const resp = await apiFetch(`/api/inferences/${record._id}`, { method: 'DELETE' });
                            if (!resp.ok) throw new Error('Delete failed');
                            toast.success('Inference deleted');
                            setRecords((r) => r.filter((rec) => rec._id !== record._id));
                          } catch (e) {
                            console.error(e);
                            toast.error('Failed to delete');
                          }
                        }}
                        className="inline-flex items-center gap-2 rounded-md border px-3 py-2 text-rose-600 hover:bg-rose-50"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
            <aside className="hidden md:block md:w-56 md:sticky md:top-28 md:self-start p-4">
              <select value={modelFilter ?? ''} onChange={(e) => setModelFilter(e.target.value || null)} className="rounded border px-3 py-2 w-full md:w-auto mb-3">
                <option value="">All models</option>
                {modelsList.map((m) => (
                  <option key={m._id} value={m._id}>{m.name}</option>
                ))}
              </select>

              <button
                disabled={selectedIds.length === 0}
                onClick={async () => {
                  try {
                    const resp = await apiFetch('/api/inferences/archive', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids: selectedIds, archived: false }) });
                    if (!resp.ok) throw new Error('Unarchive failed');
                    toast.success('Restored selected inferences');
                    setRecords((r) => r.filter((rec) => !selectedIds.includes(rec._id)));
                    setSelectedIds([]);
                  } catch (e) {
                    console.error(e);
                    toast.error('Failed to restore selected');
                  }
                }}
                className="w-full md:w-auto px-3 py-2 rounded-md bg-cvat-primary text-white mb-2"
              >
                Restore selected
              </button>

              <button
                disabled={selectedIds.length === 0}
                onClick={async () => {
                  try {
                    for (const id of selectedIds) {
                      await apiFetch(`/api/inferences/${id}`, { method: 'DELETE' });
                    }
                    toast.success('Deleted selected inferences');
                    setRecords((r) => r.filter((rec) => !selectedIds.includes(rec._id)));
                    setSelectedIds([]);
                  } catch (e) {
                    console.error(e);
                    toast.error('Failed to delete selected');
                  }
                }}
                className="w-full md:w-auto px-3 py-2 rounded-md border text-rose-600"
              >
                Delete selected
              </button>
            </aside>
          </div>
        )}
      </div>
    </div>
  );
}
