import React, { useState, useEffect } from 'react';
import { Play, Download, FileText, BarChart3, Clock, CheckCircle, XCircle } from 'lucide-react';
import { toast } from 'react-hot-toast';
import useStore from '../store/useStore';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

const Generate = () => {
  const { 
    uploadedFiles = [], // Add default empty array
    generationTasks = [], // Add default empty array
    currentTask,
    startGeneration, 
    generatePreview,
    checkTaskStatus,
    loading 
  } = useStore();

  const [selectedFile, setSelectedFile] = useState('');
  const [nRows, setNRows] = useState(1000);
  const [targetColumn, setTargetColumn] = useState('');
  const [previewData, setPreviewData] = useState(null);
  const [showPreview, setShowPreview] = useState(false);

  // Poll for task updates
  useEffect(() => {
    if (currentTask && (currentTask.status === 'PENDING' || currentTask.status === 'RUNNING' || currentTask.status === 'processing' || currentTask.status === 'queued')) {
      console.log(`🔄 Starting polling for task:`, currentTask);
      
      const interval = setInterval(async () => {
        try {
          const taskId = currentTask.task_id || currentTask.id;
          console.log(`🔄 Polling task: ${taskId}`);
          
          const updatedTask = await checkTaskStatus(taskId);
          console.log(`📊 Updated task:`, updatedTask);
          
          if (updatedTask.status === 'completed') {
            toast.success('Generation completed!');
            clearInterval(interval);
          } else if (updatedTask.status === 'failed') {
            toast.error(`Generation failed: ${updatedTask.error || 'Unknown error'}`);
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Failed to check task status:', error);
          
          // If task not found (404), stop polling
          if (error.response?.status === 404) {
            console.log('🛑 Task not found, stopping polling');
            toast.error('Task not found - stopping status checks');
            clearInterval(interval);
          }
        }
      }, 2000);
      
      return () => {
        console.log('🛑 Clearing polling interval');
        clearInterval(interval);
      };
    }
  }, [currentTask, checkTaskStatus]);

  const handlePreview = async () => {
    if (!selectedFile) {
      toast.error('Please select a file first');
      return;
    }

    try {
      const result = await generatePreview(selectedFile, Math.min(nRows, 50));
      setPreviewData(result);
      setShowPreview(true);
      toast.success('Preview generated!');
    } catch (error) {
      toast.error('Preview failed');
    }
  };

  const handleGenerate = async () => {
    if (!selectedFile) {
      toast.error('Please select a file first');
      return;
    }

    if (nRows < 1 || nRows > 100000) {
      toast.error('Number of rows must be between 1 and 100,000');
      return;
    }

    try {
      const task = await startGeneration(selectedFile, nRows, targetColumn);
      toast.success(`Generation started! Task ID: ${task.task_id}`);
    } catch (error) {
      toast.error('Failed to start generation');
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-500" size={20} />;
      case 'failed':
        return <XCircle className="text-red-500" size={20} />;
      case 'processing':
        return <Clock className="text-blue-500 animate-spin" size={20} />;
      default:
        return <Clock className="text-gray-500" size={20} />;
    }
  };

  const handleRefreshTasks = async () => {
    if (currentTask) {
      try {
        const taskId = currentTask.task_id || currentTask.id;
        console.log(`🔄 Manual refresh for task: ${taskId}`);
        await checkTaskStatus(taskId);
        toast.success('Task status refreshed');
      } catch (error) {
        console.error('Failed to refresh task status:', error);
        toast.error('Failed to refresh task status');
      }
    }
  };

  const handleDebugTasks = async () => {
    try {
      const response = await axios.get(`${API_BASE}/debug/all-tasks`);
      console.log('🐛 Debug tasks:', response.data);
      toast.success('Check console for task debug info');
    } catch (error) {
      console.error('Debug failed:', error);
      toast.error('Debug failed');
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Generate Synthetic Data</h1>
        <p className="text-gray-600">Configure and generate synthetic datasets from your uploaded files</p>
      </div>

      {/* Configuration Panel */}
      <div className="bg-white p-6 rounded-lg border">
        <h2 className="text-xl font-semibold mb-4">Configuration</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* File Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Dataset
            </label>
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Choose a file...</option>
              {uploadedFiles.map((file) => (
                <option key={file.file_id || file.id} value={file.file_id || file.id}>
                  {file.filename} ({file.rows || 0} rows, {file.columns || 0} columns)
                </option>
              ))}
            </select>
          </div>

          {/* Number of Rows */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Number of Rows to Generate
            </label>
            <input
              type="number"
              value={nRows}
              onChange={(e) => setNRows(parseInt(e.target.value) || 1000)}
              min="10"
              max="100000"
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* Target Column (Optional) */}
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Target Column (Optional)
            </label>
            <input
              type="text"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              placeholder="e.g., target, label, outcome"
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <p className="text-sm text-gray-500 mt-1">
              Specify a target column for enhanced quality analysis
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-4 mt-6">
          <button
            onClick={handlePreview}
            disabled={!selectedFile || loading}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 disabled:opacity-50"
          >
            <FileText size={16} />
            <span>Preview</span>
          </button>
          
          <button
            onClick={handleGenerate}
            disabled={!selectedFile || loading || (currentTask && currentTask.status === 'RUNNING')}
            className="flex items-center space-x-2 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            <Play size={16} />
            <span>Generate</span>
          </button>
        </div>
      </div>

      {/* Preview Panel */}
      {showPreview && previewData && (
        <div className="bg-white p-6 rounded-lg border">
          <h3 className="text-lg font-semibold mb-4">Data Preview</h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Original Data (Sample)</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      {previewData.original_sample[0] && Object.keys(previewData.original_sample[0]).map(key => (
                        <th key={key} className="px-3 py-2 text-left font-medium text-gray-700">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.original_sample.slice(0, 5).map((row, idx) => (
                      <tr key={idx} className="border-t">
                        {Object.values(row).map((value, i) => (
                          <td key={i} className="px-3 py-2 text-gray-600">
                            {String(value).substring(0, 20)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h4 className="font-medium text-gray-700 mb-2">Synthetic Data (Preview)</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      {previewData.preview_data[0] && Object.keys(previewData.preview_data[0]).map(key => (
                        <th key={key} className="px-3 py-2 text-left font-medium text-gray-700">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.preview_data.slice(0, 5).map((row, idx) => (
                      <tr key={idx} className="border-t">
                        {Object.values(row).map((value, i) => (
                          <td key={i} className="px-3 py-2 text-gray-600">
                            {String(value).substring(0, 20)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Current Task Status */}
      {currentTask && (
        <div className="bg-white p-6 rounded-lg border">
          <h3 className="text-lg font-semibold mb-4">Generation Status</h3>
          
          <div className="flex items-center space-x-3 mb-4">
            {getStatusIcon(currentTask.status)}
            <span className="font-medium">{currentTask.status}</span>
            {currentTask.progress > 0 && (
              <span className="text-gray-500">({currentTask.progress}%)</span>
            )}
          </div>

          {currentTask.status === 'processing' && (
            <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${currentTask.progress}%` }}
              ></div>
            </div>
          )}

          {currentTask.status === 'completed' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-green-50 rounded-md">
                <div>
                  <p className="font-medium text-green-800">Generation Complete!</p>
                  <p className="text-sm text-green-600">
                    Generated {currentTask.n_rows} rows
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <a
                    href={`http://localhost:8000/download/${currentTask.task_id || currentTask.id}`}
                    className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                  >
                    <Download size={16} />
                    <span>Data</span>
                  </a>
                  <a
                    href={`http://localhost:8000/download/report/${currentTask.task_id || currentTask.id}`}
                    className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
                  >
                    <FileText size={16} />
                    <span>Report</span>
                  </a>
                </div>
              </div>

              {/* Quality Report */}
              {currentTask.quality_report && (
                <div className="p-4 bg-gray-50 rounded-md">
                  <h4 className="font-medium mb-2 flex items-center space-x-2">
                    <BarChart3 size={16} />
                    <span>Quality Report</span>
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">Overall Score</p>
                      <p className="font-medium">
                        {currentTask.quality_report.overall_score?.overall_quality_score?.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-600">Fidelity</p>
                      <p className="font-medium">
                        {currentTask.quality_report.overall_score?.fidelity_score?.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-600">Utility</p>
                      <p className="font-medium">
                        {currentTask.quality_report.overall_score?.utility_score?.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-600">Privacy</p>
                      <p className="font-medium">
                        {currentTask.quality_report.overall_score?.privacy_score?.toFixed(2)}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {currentTask.status === 'failed' && (
            <div className="p-4 bg-red-50 rounded-md">
              <p className="text-red-800 font-medium">Generation Failed</p>
              <p className="text-red-600 text-sm">{currentTask.error}</p>
            </div>
          )}
        </div>
      )}

      {/* Recent Tasks */}
      {generationTasks.length > 0 && (
        <div className="bg-white p-6 rounded-lg border">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Recent Tasks</h3>
            <button
              onClick={handleRefreshTasks}
              className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded"
            >
              Refresh
            </button>
          </div>
          <div className="space-y-3">
            {generationTasks.slice(-5).reverse().map((task) => (
              <div key={task.task_id || task.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(task.status)}
                  <div>
                    <p className="font-medium">File: {task.file_id}</p>
                    <p className="text-sm text-gray-500">
                      {task.n_rows} rows • Status: {task.status} • {new Date(task.created_at).toLocaleString()}
                    </p>
                {task.status === 'processing' && (
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${task.progress || 0}%` }}
                    ></div>
                  </div>
                )}
              </div>
            </div>
            
            {task.status === 'completed' && (
              <div className="flex items-center space-x-2">
                <a
                  href={`http://localhost:8000/download/${task.task_id || task.id}`}
                  className="flex items-center space-x-1 px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                >
                  <Download size={14} />
                  <span>Data</span>
                </a>
                <a
                  href={`http://localhost:8000/download/report/${task.task_id || task.id}`}
                  className="flex items-center space-x-1 px-3 py-1 bg-gray-600 text-white text-sm rounded hover:bg-gray-700"
                >
                  <FileText size={14} />
                  <span>Report</span>
                </a>
              </div>
            )}
          </div>
            ))}
          </div>
        </div>
      )}

      {/* Debug Button */}
      <div className="mt-6">
        <button
          onClick={handleDebugTasks}
          className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
        >
          Debug Tasks
        </button>
      </div>
    </div>
  );
};

export default Generate;
