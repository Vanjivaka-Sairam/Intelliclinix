"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { toast } from "react-hot-toast";
import DashboardNav from "@/components/DashboardNav";
import { 
  Upload, 
  Brain, 
  Heart, 
  Loader2, 
  FileArchive, 
  Check, 
  Info, 
  ArrowRight, 
  Database,
  Cpu,
  Microscope,
  Plus,
  Edit,
  Trash2,
  Download,
  Settings
} from "lucide-react";

type Model = {
  _id: string;
  model_id: string;
  name: string;
  description: string;
  type: string;
  version: string;
  supported_formats?: string[];
  input_formats?: string[];
  channels?: number[];
  diameter?: number;
  use_gpu: boolean;
  enabled: boolean;
  is_training: boolean;
  training_data_size?: number;
  accuracy?: number;
  created_at: string;
  created_by: string;
};

type ModelType = {
  name: string;
  description: string;
  supported_formats: string[];
  input_formats: string[];
  default_channels: number[] | null;
  default_diameter: number | null;
  use_gpu: boolean;
  parameters: Record<string, any>;
};

export default function ModelsPage() {
  const router = useRouter();
  const [models, setModels] = useState<Model[]>([]);
  const [modelTypes, setModelTypes] = useState<Record<string, ModelType>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  
  // Model creation state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newModel, setNewModel] = useState({
    name: '',
    description: '',
    type: 'cellpose',
    version: '1.0.0',
    supported_formats: [] as string[],
    input_formats: [] as string[],
    channels: [0, 0],
    diameter: null as number | null,
    use_gpu: true
  });
  
  // File upload state
  const [uploadingFile, setUploadingFile] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  useEffect(() => {
    const skipAuth = localStorage.getItem("skipAuth");
    if (skipAuth === "true") {
      setIsAuthenticated(true);
      return;
    }
    const checkAuth = async () => {
      try {
        const response = await fetch("http://localhost:5328/auth/user", {
          method: "GET",
          credentials: "include",
        });
        if (!response.ok) throw new Error("Authentication failed");
        const data = await response.json();
        if (!data.authenticated) {
          toast.error("Please log in first.");
          router.push("/login");
          return;
        }
        setIsAuthenticated(true);
      } catch (error) {
        console.error("Error checking authentication:", error);
        toast.error("Authentication check failed");
        router.push("/login");
      }
    };
    checkAuth();
  }, [router]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchModels();
      fetchModelTypes();
    }
  }, [isAuthenticated]);

  const fetchModels = async () => {
    try {
      setIsLoading(true);
      const response = await fetch("http://localhost:5328/api/models/", {
        credentials: "include",
      });
      
      if (!response.ok) {
        throw new Error("Failed to fetch models");
      }
      
      const data = await response.json();
      setModels(data.data?.models || []);
    } catch (error) {
      console.error("Error fetching models:", error);
      toast.error("Failed to load AI models");
    } finally {
      setIsLoading(false);
    }
  };

  const fetchModelTypes = async () => {
    try {
      const response = await fetch("http://localhost:5328/api/models/types", {
        credentials: "include",
      });
      
      if (!response.ok) {
        throw new Error("Failed to fetch model types");
      }
      
      const data = await response.json();
      setModelTypes(data.data || {});
    } catch (error) {
      console.error("Error fetching model types:", error);
    }
  };

  const handleCreateModel = async () => {
    try {
      const response = await fetch("http://localhost:5328/api/models/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(newModel),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to create model");
      }
      
      const data = await response.json();
      toast.success("Model created successfully!");
      setShowCreateModal(false);
      setNewModel({
        name: '',
        description: '',
        type: 'cellpose',
        version: '1.0.0',
        supported_formats: [],
        input_formats: [],
        channels: [0, 0],
        diameter: null,
        use_gpu: true
      });
      fetchModels();
    } catch (error: any) {
      console.error("Error creating model:", error);
      toast.error(error.message || "Failed to create model");
    }
  };

  const handleUploadModelFile = async (modelId: string, file: File) => {
    try {
      setUploadingFile(modelId);
      setUploadProgress(0);
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`http://localhost:5328/api/models/${modelId}/upload`, {
        method: "POST",
        body: formData,
        credentials: "include",
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to upload model file");
      }
      
      toast.success("Model file uploaded successfully!");
      fetchModels();
    } catch (error: any) {
      console.error("Error uploading model file:", error);
      toast.error(error.message || "Failed to upload model file");
    } finally {
      setUploadingFile(null);
      setUploadProgress(0);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    if (!confirm("Are you sure you want to delete this model? This action cannot be undone.")) {
      return;
    }
    
    try {
      const response = await fetch(`http://localhost:5328/api/models/${modelId}`, {
        method: "DELETE",
        credentials: "include",
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to delete model");
      }
      
      toast.success("Model deleted successfully!");
      fetchModels();
    } catch (error: any) {
      console.error("Error deleting model:", error);
      toast.error(error.message || "Failed to delete model");
    }
  };

  const handleToggleModel = async (modelId: string, enabled: boolean) => {
    try {
      const response = await fetch(`http://localhost:5328/api/models/${modelId}/enable`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ enabled }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to update model status");
      }
      
      toast.success(`Model ${enabled ? 'enabled' : 'disabled'} successfully!`);
      fetchModels();
    } catch (error: any) {
      console.error("Error updating model status:", error);
      toast.error(error.message || "Failed to update model status");
    }
  };

  const getModelIcon = (type: string) => {
    switch (type) {
      case 'cellpose':
        return <Microscope className="h-5 w-5" />;
      case 'nnunet':
        return <Brain className="h-5 w-5" />;
      default:
        return <Cpu className="h-5 w-5" />;
    }
  };

  const getModelTypeInfo = (type: string) => {
    return modelTypes[type] || {
      name: type.toUpperCase(),
      description: 'Custom model',
      supported_formats: [],
      input_formats: []
    };
  };

  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-cvat-bg-primary">
        <div className="animate-pulse flex items-center space-x-2 text-cvat-primary">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span className="text-lg font-medium">Authenticating...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-cvat-bg-primary">
      <DashboardNav />
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-cvat-text-primary">AI Models</h1>
            <p className="text-cvat-text-secondary">Manage your trained AI models for medical image analysis</p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="cvat-button-primary flex items-center px-4 py-2 rounded-lg"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Model
          </button>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-cvat-primary mr-2" />
            <span className="text-cvat-text-secondary">Loading models...</span>
          </div>
        ) : models.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {models.map((model) => {
              const typeInfo = getModelTypeInfo(model.type);
              return (
                <div key={model.model_id} className="cvat-card p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center">
                      <div className="p-2 bg-cvat-primary/10 rounded-lg mr-3">
                        {getModelIcon(model.type)}
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-cvat-text-primary">{model.name}</h3>
                        <p className="text-sm text-cvat-text-secondary">{typeInfo.name}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handleToggleModel(model.model_id, !model.enabled)}
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          model.enabled 
                            ? 'bg-cvat-success/10 text-cvat-success' 
                            : 'bg-cvat-text-tertiary/10 text-cvat-text-tertiary'
                        }`}
                      >
                        {model.enabled ? 'Enabled' : 'Disabled'}
                      </button>
                      <button
                        onClick={() => handleDeleteModel(model.model_id)}
                        className="p-1 text-cvat-error hover:bg-cvat-error/10 rounded"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  
                  <p className="text-sm text-cvat-text-secondary mb-4">{model.description}</p>
                  
                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-xs">
                      <span className="text-cvat-text-tertiary">Version:</span>
                      <span className="text-cvat-text-primary">{model.version}</span>
                    </div>
                    {model.accuracy && (
                      <div className="flex justify-between text-xs">
                        <span className="text-cvat-text-tertiary">Accuracy:</span>
                        <span className="text-cvat-text-primary">{(model.accuracy * 100).toFixed(1)}%</span>
                      </div>
                    )}
                    {model.training_data_size && (
                      <div className="flex justify-between text-xs">
                        <span className="text-cvat-text-tertiary">Training Data:</span>
                        <span className="text-cvat-text-primary">{model.training_data_size} images</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="mb-4">
                    <p className="text-xs text-cvat-text-tertiary mb-1">Supported Formats:</p>
                    <div className="flex flex-wrap gap-1">
                      {model.supported_formats?.map((format, index) => (
                        <span key={index} className="px-2 py-1 bg-cvat-bg-tertiary text-xs rounded">
                          {format}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    <label className="flex-1">
                      <input
                        type="file"
                        accept=".pth,.pkl,.h5,.pt,.model,.weights"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) {
                            handleUploadModelFile(model.model_id, file);
                          }
                        }}
                        className="hidden"
                      />
                      <div className="w-full px-3 py-2 bg-cvat-bg-tertiary border border-cvat-border rounded text-center text-sm cursor-pointer hover:bg-cvat-bg-secondary">
                        {uploadingFile === model.model_id ? (
                          <div className="flex items-center justify-center">
                            <Loader2 className="h-4 w-4 animate-spin mr-2" />
                            Uploading...
                          </div>
                        ) : (
                          'Upload Model File'
                        )}
                      </div>
                    </label>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-12 bg-cvat-bg-secondary rounded-lg">
            <Cpu className="h-16 w-16 mx-auto text-cvat-text-tertiary mb-4" />
            <h3 className="text-lg font-medium text-cvat-text-primary mb-2">No Models Found</h3>
            <p className="text-cvat-text-secondary mb-6">Upload your first AI model to get started with medical image analysis.</p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="cvat-button-primary px-6 py-2 rounded-lg"
            >
              Add Your First Model
            </button>
          </div>
        )}
      </div>

      {/* Create Model Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-cvat-bg-secondary rounded-lg shadow-xl max-w-md w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-cvat-text-primary">Create New Model</h2>
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="text-cvat-text-tertiary hover:text-cvat-text-primary"
                >
                  <span className="text-2xl">&times;</span>
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-cvat-text-primary mb-1">
                    Model Name
                  </label>
                  <input
                    type="text"
                    value={newModel.name}
                    onChange={(e) => setNewModel({...newModel, name: e.target.value})}
                    className="w-full px-3 py-2 border border-cvat-border rounded bg-cvat-bg-secondary text-cvat-text-primary"
                    placeholder="Enter model name"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-cvat-text-primary mb-1">
                    Description
                  </label>
                  <textarea
                    value={newModel.description}
                    onChange={(e) => setNewModel({...newModel, description: e.target.value})}
                    className="w-full px-3 py-2 border border-cvat-border rounded bg-cvat-bg-secondary text-cvat-text-primary"
                    rows={3}
                    placeholder="Enter model description"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-cvat-text-primary mb-1">
                    Model Type
                  </label>
                  <select
                    value={newModel.type}
                    onChange={(e) => {
                      const type = e.target.value;
                      const typeInfo = modelTypes[type];
                      setNewModel({
                        ...newModel,
                        type,
                        supported_formats: typeInfo?.supported_formats || [],
                        input_formats: typeInfo?.input_formats || [],
                        channels: typeInfo?.default_channels || [0, 0],
                        diameter: typeInfo?.default_diameter || null
                      });
                    }}
                    className="w-full px-3 py-2 border border-cvat-border rounded bg-cvat-bg-secondary text-cvat-text-primary"
                  >
                    {Object.entries(modelTypes).map(([key, type]) => (
                      <option key={key} value={key}>{type.name}</option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-cvat-text-primary mb-1">
                    Version
                  </label>
                  <input
                    type="text"
                    value={newModel.version}
                    onChange={(e) => setNewModel({...newModel, version: e.target.value})}
                    className="w-full px-3 py-2 border border-cvat-border rounded bg-cvat-bg-secondary text-cvat-text-primary"
                    placeholder="1.0.0"
                  />
                </div>
                
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="use_gpu"
                    checked={newModel.use_gpu}
                    onChange={(e) => setNewModel({...newModel, use_gpu: e.target.checked})}
                    className="mr-2"
                  />
                  <label htmlFor="use_gpu" className="text-sm text-cvat-text-primary">
                    Use GPU acceleration
                  </label>
                </div>
              </div>
              
              <div className="flex justify-end space-x-3 mt-6">
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="px-4 py-2 text-cvat-text-secondary hover:text-cvat-text-primary"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateModel}
                  className="cvat-button-primary px-4 py-2 rounded"
                >
                  Create Model
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}