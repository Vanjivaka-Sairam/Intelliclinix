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
  Microscope
} from "lucide-react";

type DatasetConfig = {
  id: string;
  name: string;
  description: string;
  filePattern: string;
  icon: JSX.Element;
  color: string;
};

type Model = {
  _id: string;
  model_id: string;
  name: string;
  description: string;
  type: string;
  supported_formats?: string[];
  input_formats?: string[];
  enabled: boolean;
  created_at: number;
};


const DATASET_CONFIGS: { [key: string]: DatasetConfig } = {
  "Dataset001_BrainTumour": {
    id: "Dataset001_BrainTumour",
    name: "Brain Tumor Dataset",
    description: "BRATS dataset for brain tumor segmentation",
    filePattern: "BRATS_XXX_XXXX.nii.gz or XXXX_0000.nii.gz",
    icon: <Brain className="h-6 w-6" />,
    color: "blue"
  },
  "Dataset002_Heart": {
    id: "Dataset002_Heart",
    name: "Heart Dataset",
    description: "Left atrium segmentation dataset",
    filePattern: "la_XXX_0000.nii.gz",
    icon: <Heart className="h-6 w-6" />,
    color: "red"
  }
};

export default function NewUploadPage() {
  const router = useRouter();
  const [config, setConfig] = useState<"2d" | "3d_fullres">("3d_fullres");
  const [selectedDataset, setSelectedDataset] = useState<string>("Dataset001_BrainTumour");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);

  // Model selection state
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  // Progress bar label state
  const [progressBarText, setProgressBarText] = useState("Upload Files");
  // Stepper state: 1 = Select Model, 2 = Select Dataset, 3 = Upload Files, 4 = Processing
  const [currentStep, setCurrentStep] = useState(1);

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
        if (data.user && data.user.username) {
          localStorage.setItem("username", data.user.username);
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

  // Load available models
  useEffect(() => {
    if (isAuthenticated) {
      fetchModels();
    }
  }, [isAuthenticated]);

  const fetchModels = async () => {
    try {
      setIsLoadingModels(true);
      const response = await fetch("http://localhost:5328/api/inference/models", {
        credentials: "include",
      });
      
      if (!response.ok) {
        throw new Error("Failed to fetch models");
      }
      
      const data = await response.json();
      console.log("Fetched models:", data.models);
      setModels(data.models || []);
      
      // Auto-select first model if available
      if (data.models && data.models.length > 0) {
        console.log("Auto-selecting model:", data.models[0]);
        setSelectedModel(data.models[0]);
      }
    } catch (error) {
      console.error("Error fetching models:", error);
      toast.error("Failed to load AI models");
    } finally {
      setIsLoadingModels(false);
    }
  };

  // Update stepper and progress bar label
  useEffect(() => {
    if (isProcessing) {
      setCurrentStep(3);
      setProgressBarText("Processing");
    } else if (selectedFile) {
      setCurrentStep(2);
      setProgressBarText("Upload Files");
    } else {
      setCurrentStep(1);
      setProgressBarText("Select Model");
    }
  }, [selectedFile, isProcessing, selectedDataset, config, selectedModel]);

  const handleSubmit = async () => {
    if (!isAuthenticated) {
      toast.error("Please log in first");
      router.push("/login");
      return;
    }
    if (!selectedModel) {
      toast.error("Please select an AI model");
      return;
    }
    if (!selectedFile) {
      toast.error("Please select a file to upload");
      return;
    }
    const username = localStorage.getItem("username");
    if (!username) {
      toast.error("User not found. Please log in again.");
      router.push("/login");
      return;
    }
    setIsProcessing(true);
    setUploadProgress(0);
    try {
      const formData = new FormData();
      formData.append("files", selectedFile);
      formData.append("model_id", selectedModel.model_id);
      formData.append("config", config);
      formData.append("username", username);
      formData.append("dataset", selectedDataset);
      // Simulate upload progress (in a real app, use XMLHttpRequest with progress events)
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          return prev + Math.random() * 5; // Slower progress for Cellpose
        });
      }, 1000); // Update every second
      // Debug: Log the form data being sent
      console.log("Selected model:", selectedModel);
      console.log("Selected file:", selectedFile);
      console.log("File name:", selectedFile.name);
      console.log("File type:", selectedFile.type);
      console.log("File size:", selectedFile.size);
      
      const uploadResponse = await fetch("http://localhost:5328/api/inference/upload", {
        method: "POST",
        body: formData,
        credentials: "include",
      });
      clearInterval(progressInterval);
      setUploadProgress(100);
      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        if (uploadResponse.status === 401) {
          toast.error("Session expired. Please log in again.");
          router.push("/login");
          return;
        }
        throw new Error(errorData.error || "Upload failed");
      }
      const uploadData = await uploadResponse.json();
      if (!uploadData.success) throw new Error(uploadData.error);
      
      // Start inference using the generic model system
      const inferenceResponse = await fetch("http://localhost:5328/api/inference/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        // keepalive ensures the request continues even if the page navigates away
        keepalive: true as any,
        body: JSON.stringify({
          job_id: uploadData.job_id,
          model_id: selectedModel.model_id
        }),
      });
      if (!inferenceResponse.ok) {
        const errorData = await inferenceResponse.json();
        if (inferenceResponse.status === 401) {
          toast.error("Session expired. Please log in again.");
          router.push("/login");
          return;
        }
        throw new Error(errorData.error || "Inference failed");
      }
      toast.success("Processing completed! Redirecting to results...");
      router.push("/predictions");
    } catch (error: any) {
      console.error("Processing error:", error);
      toast.error(error.message || "Processing failed");
    } finally {
      setIsProcessing(false);
    }
  };

  // Drag and drop handlers
  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
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

  const selectedDatasetConfig = DATASET_CONFIGS[selectedDataset];
  const datasetIconColor = selectedDatasetConfig.color === "blue" ? "text-blue-600" : "text-red-600";
  const datasetBgColor = selectedDatasetConfig.color === "blue" ? "bg-blue-100" : "bg-red-100";
  const datasetBorderColor = selectedDatasetConfig.color === "blue" ? "border-blue-200" : "border-red-200";

  return (
    <div className="min-h-screen bg-cvat-bg-primary">
      <DashboardNav />
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-cvat-text-primary">Medical Image Upload</h1>
          <p className="text-cvat-text-secondary">Upload medical scans for AI-assisted analysis and annotation</p>
        </div>
        <div className="max-w-3xl mx-auto">
          <div className="cvat-card overflow-hidden">
            {/* Stepper */}
            <div className="bg-cvat-bg-tertiary px-6 py-4 border-b border-cvat-border">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className={`flex items-center justify-center h-8 w-8 rounded-full font-medium
                    ${currentStep === 1 ? "bg-cvat-primary text-cvat-text-white" : "bg-cvat-border text-cvat-text-tertiary"}`}>1</div>
                  <span className={`ml-2 font-medium ${currentStep === 1 ? "text-cvat-primary" : "text-cvat-text-tertiary"}`}>Select Model</span>
                </div>
                <div className="h-0.5 flex-1 mx-4 bg-cvat-border"></div>
                <div className="flex items-center">
                  <div className={`flex items-center justify-center h-8 w-8 rounded-full font-medium
                    ${currentStep === 2 ? "bg-cvat-primary text-cvat-text-white" : "bg-cvat-border text-cvat-text-tertiary"}`}>2</div>
                  <span className={`ml-2 font-medium ${currentStep === 2 ? "text-cvat-primary" : "text-cvat-text-tertiary"}`}>Upload Files</span>
                </div>
                <div className="h-0.5 flex-1 mx-4 bg-cvat-border"></div>
                <div className="flex items-center">
                  <div className={`flex items-center justify-center h-8 w-8 rounded-full font-medium
                    ${currentStep === 3 ? "bg-cvat-primary text-cvat-text-white" : "bg-cvat-border text-cvat-text-tertiary"}`}>3</div>
                  <span className={`ml-2 font-medium ${currentStep === 3 ? "text-cvat-primary" : "text-cvat-text-tertiary"}`}>Processing</span>
                </div>
              </div>
            </div>
            <div className="px-8 py-6">
              {/* Model Selection Dropdown */}
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-cvat-text-primary mb-4 flex items-center">
                  <Cpu className="h-5 w-5 mr-2 text-cvat-primary" />
                  Select Trained Model
                </h3>
                {isLoadingModels ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-cvat-primary mr-2" />
                    <span className="text-cvat-text-secondary">Loading trained models...</span>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <label htmlFor="model-select" className="block text-sm font-medium text-cvat-text-primary mb-2">
                        Choose a trained model for analysis
                      </label>
                      <select
                        id="model-select"
                        value={selectedModel?.model_id || ''}
                        onChange={(e) => {
                          const model = models.find(m => m.model_id === e.target.value);
                          setSelectedModel(model || null);
                          // Auto-set dataset based on model type
                          if (model) {
                            if (model.name.toLowerCase().includes('brain')) {
                              setSelectedDataset('Dataset001_BrainTumour');
                            } else if (model.name.toLowerCase().includes('heart')) {
                              setSelectedDataset('Dataset002_Heart');
                            }
                          }
                        }}
                        className="w-full px-4 py-3 border border-cvat-border rounded-lg bg-cvat-bg-secondary text-cvat-text-primary focus:outline-none focus:ring-2 focus:ring-cvat-primary focus:border-cvat-primary"
                      >
                        <option value="">Select a trained model...</option>
                        {models.map((model) => (
                          <option key={model.model_id} value={model.model_id}>
                            {model.name} - {model.description}
                          </option>
                        ))}
                      </select>
                    </div>
                    
                    {/* Selected Model Info */}
                    {selectedModel && (
                      <div className="p-4 bg-cvat-primary/5 border border-cvat-primary/20 rounded-lg">
                        <div className="flex items-start">
                          <div className="flex-shrink-0 h-10 w-10 rounded-lg bg-cvat-primary/10 flex items-center justify-center">
                            {selectedModel.type === 'cellpose' ? (
                              <Microscope className="h-5 w-5 text-cvat-primary" />
                            ) : (
                              <Brain className="h-5 w-5 text-cvat-primary" />
                            )}
                          </div>
                          <div className="ml-3 flex-1">
                            <h4 className="text-sm font-medium text-cvat-text-primary">{selectedModel.name}</h4>
                            <p className="text-sm text-cvat-text-secondary mt-1">{selectedModel.description}</p>
                            <div className="mt-2 flex items-center space-x-2">
                              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-cvat-primary/10 text-cvat-primary">
                                {selectedModel.type.toUpperCase()}
                              </span>
                              <span className="text-xs text-cvat-text-tertiary">
                                Supported formats: {selectedModel.supported_formats?.join(', ') || selectedModel.input_formats?.join(', ') || 'N/A'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
                <div className={`mt-4 p-4 rounded-lg bg-cvat-primary/5 border-cvat-primary/20 border flex items-start`}>
                  <Info className={`h-5 w-5 text-cvat-primary mt-0.5 flex-shrink-0`} />
                  <div className="ml-3">
                    <p className="text-sm text-cvat-text-primary">
                      <span className="font-medium">Expected file format:</span> {selectedDatasetConfig.filePattern}
                    </p>
                    <p className="text-sm text-cvat-text-secondary mt-1">
                      Make sure your files match the expected naming convention for accurate analysis.
                    </p>
                  </div>
                </div>
              </div>
              {/* File Upload */}
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-cvat-text-primary mb-4 flex items-center">
                  <Upload className="h-5 w-5 mr-2 text-cvat-primary" />
                  Upload Scan Data
                </h3>
                <div 
                  className={`border-2 ${dragActive ? "border-cvat-primary bg-cvat-primary/5" : "border-dashed border-cvat-border"} rounded-lg p-8 text-center transition-all duration-200`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <input
                    type="file"
                    accept=".zip"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  {!selectedFile ? (
                    <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center space-y-4">
                      <div className="p-3 bg-cvat-primary/10 rounded-full">
                        <FileArchive className="h-8 w-8 text-cvat-primary" />
                      </div>
                      <div>
                        <p className="text-cvat-text-primary font-medium">Drag &amp; drop your ZIP file here</p>
                        <p className="text-sm text-cvat-text-secondary mt-1">
                          or <span className="text-cvat-primary">browse files</span>
                        </p>
                      </div>
                      <p className="text-xs text-cvat-text-tertiary bg-cvat-bg-tertiary py-1 px-2 rounded-full">
                        Supported format: .zip containing scan data
                      </p>
                    </label>
                  ) : (
                    <div className="flex flex-col items-center space-y-3">
                      <div className="p-3 bg-cvat-success/10 rounded-full">
                        <Check className="h-8 w-8 text-cvat-success" />
                      </div>
                      <div>
                        <p className="text-cvat-text-primary font-medium">{selectedFile.name}</p>
                        <p className="text-sm text-cvat-text-secondary mt-1">
                          {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                      <button 
                        className="text-sm text-cvat-primary hover:text-cvat-primary-hover underline"
                        onClick={() => setSelectedFile(null)}
                      >
                        Change file
                      </button>
                    </div>
                  )}
                </div>
              </div>
              {/* Scan Configuration */}
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-cvat-text-primary mb-4 flex items-center">
                  <Database className="h-5 w-5 mr-2 text-cvat-primary" />
                  Scan Configuration
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div 
                    className={`cursor-pointer border-2 rounded-lg p-4 transition ${
                      config === "2d" 
                        ? "border-cvat-primary bg-cvat-primary/5"
                        : "border-cvat-border hover:border-cvat-primary/50"
                    }`}
                    onClick={() => setConfig("2d")}
                  >
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-10 w-10 rounded-lg bg-cvat-primary/10 flex items-center justify-center">
                        <span className="font-bold text-cvat-primary">2D</span>
                      </div>
                      <div className="ml-4">
                        <h4 className="font-medium text-cvat-text-primary">2D Slice Collection</h4>
                        <p className="text-sm text-cvat-text-secondary">Analysis of individual 2D slices</p>
                      </div>
                      {config === "2d" && <Check className="ml-auto h-5 w-5 text-cvat-primary" />}
                    </div>
                  </div>
                  <div 
                    className={`cursor-pointer border-2 rounded-lg p-4 transition ${
                      config === "3d_fullres" 
                        ? "border-cvat-primary bg-cvat-primary/5"
                        : "border-cvat-border hover:border-cvat-primary/50"
                    }`}
                    onClick={() => setConfig("3d_fullres")}
                  >
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-10 w-10 rounded-lg bg-cvat-primary/10 flex items-center justify-center">
                        <span className="font-bold text-cvat-primary">3D</span>
                      </div>
                      <div className="ml-4">
                        <h4 className="font-medium text-cvat-text-primary">3D Volume (High Resolution)</h4>
                        <p className="text-sm text-cvat-text-secondary">Full volumetric analysis</p>
                      </div>
                      {config === "3d_fullres" && <Check className="ml-auto h-5 w-5 text-cvat-primary" />}
                    </div>
                  </div>
                </div>
              </div>
              {/* Progress Bar and Submit Button */}
              {selectedFile && (
                <div className="mb-4">
                  <div className="flex justify-between text-sm text-cvat-text-secondary mb-1">
                    <span>{progressBarText}</span>
                    <span>{Math.round(uploadProgress)}%</span>
                  </div>
                  <div className="h-2 w-full bg-cvat-border rounded-full overflow-hidden mb-2">
                    <div 
                      className="h-full bg-cvat-primary transition-all duration-500"
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                </div>
              )}
              <div>
                <button
                  onClick={handleSubmit}
                  disabled={!selectedModel || !selectedFile || isProcessing}
                  className="cvat-button-primary w-full flex items-center justify-center py-3 px-6 font-medium rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <span>Start Analysis</span>
                      <ArrowRight className="ml-2 h-5 w-5" />
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
          <div className="mt-6 text-center text-sm text-cvat-text-tertiary">
            Having issues? <a href="#" className="text-cvat-primary hover:underline">Contact support</a> or check our <a href="#" className="text-cvat-primary hover:underline">documentation</a>.
          </div>
        </div>
      </div>
    </div>
  );
}
