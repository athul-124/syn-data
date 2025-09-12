import React, { useState, useEffect } from 'react';
import { Upload as UploadIcon, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from 'react-hot-toast';
import useStore from '../store/useStore';
import axios from 'axios';

const Upload = () => {
  const { uploadFile, loading, error } = useStore();
  const [dragActive, setDragActive] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');

  // Check backend health on component mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await axios.get('http://localhost:5000/health');
        console.log('Backend health:', response.data);
        setBackendStatus('connected');
      } catch (error) {
        console.error('Backend not reachable:', error);
        setBackendStatus('disconnected');
      }
    };
    checkBackend();
  }, []);

  const handleFile = async (file) => {
    if (!file) return;
    
    console.log('Processing file:', file.name);
    setUploadSuccess(false);
    
    try {
      await uploadFile(file);
      setUploadSuccess(true);
      toast.success('File uploaded successfully!');
    } catch (error) {
      console.error('Upload failed:', error);
      toast.error('Upload failed. Please try again.');
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    const files = e.target.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Dataset</h1>
        <p className="text-gray-600">
          Upload your dataset to generate synthetic data. Supported formats: CSV, JSON, XLSX
        </p>
        
        {/* Backend status indicator */}
        <div className="mt-2">
          {backendStatus === 'checking' && (
            <span className="text-yellow-600">🔄 Checking backend connection...</span>
          )}
          {backendStatus === 'connected' && (
            <span className="text-green-600">✅ Backend connected</span>
          )}
          {backendStatus === 'disconnected' && (
            <span className="text-red-600">❌ Backend not running. Start with: uvicorn main:app --reload --port 8000</span>
          )}
        </div>
      </div>
      
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md flex items-center space-x-2">
          <AlertCircle className="text-red-500" size={20} />
          <span className="text-red-700">{error}</span>
        </div>
      )}

      {uploadSuccess && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-md flex items-center space-x-2">
          <CheckCircle className="text-green-500" size={20} />
          <span className="text-green-700">File uploaded successfully! You can now generate synthetic data.</span>
        </div>
      )}
      
      <div
        className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
          dragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300'
        } ${loading ? 'opacity-50 pointer-events-none' : 'hover:border-gray-400 cursor-pointer'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => document.getElementById('fileInput').click()}
      >
        <input
          id="fileInput"
          type="file"
          className="hidden"
          accept=".csv,.json,.xlsx"
          onChange={handleChange}
          disabled={loading}
        />
        
        <div className="flex flex-col items-center space-y-4">
          <div className="p-4 bg-gray-100 rounded-full">
            <UploadIcon size={32} className="text-gray-600" />
          </div>
          
          {loading ? (
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
              <p className="text-gray-600">Uploading...</p>
            </div>
          ) : (
            <div className="text-center">
              <p className="text-lg font-medium text-gray-900 mb-2">
                {dragActive ? 'Drop your file here' : 'Drag & drop your dataset'}
              </p>
              <p className="text-gray-600 mb-4">or click to browse</p>
              <p className="text-sm text-gray-500">
                Supports CSV, JSON, and XLSX files up to 100MB
              </p>
            </div>
          )}
        </div>
      </div>
      
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-6 bg-white rounded-lg border">
          <FileText className="text-blue-600 mb-3" size={24} />
          <h3 className="font-semibold text-gray-900 mb-2">CSV Files</h3>
          <p className="text-sm text-gray-600">
            Comma-separated values with headers in the first row
          </p>
        </div>
        
        <div className="p-6 bg-white rounded-lg border">
          <FileText className="text-green-600 mb-3" size={24} />
          <h3 className="font-semibold text-gray-900 mb-2">JSON Files</h3>
          <p className="text-sm text-gray-600">
            Structured data in JSON format with consistent schema
          </p>
        </div>
        
        <div className="p-6 bg-white rounded-lg border">
          <FileText className="text-purple-600 mb-3" size={24} />
          <h3 className="font-semibold text-gray-900 mb-2">Excel Files</h3>
          <p className="text-sm text-gray-600">
            XLSX files with data in the first sheet
          </p>
        </div>
      </div>
    </div>
  );
};

export default Upload;














