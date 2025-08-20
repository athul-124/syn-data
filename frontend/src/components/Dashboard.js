import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Upload, Zap, BarChart3, Clock, Download, FileText, CheckCircle, XCircle } from 'lucide-react';
import useStore from '../store/useStore';

const Dashboard = () => {
  const { 
    uploadedFiles = [], 
    generationTasks = [], 
    loading 
  } = useStore();

  const [stats, setStats] = useState({
    totalUploads: 0,
    totalGenerations: 0,
    totalSyntheticRows: 0,
    successRate: 0
  });

  useEffect(() => {
    // Calculate stats when data changes
    const completedTasks = generationTasks.filter(task => task.status === 'completed');
    const totalSyntheticRows = completedTasks.reduce((sum, task) => sum + (task.n_rows || 0), 0);
    const successRate = generationTasks.length > 0 ? (completedTasks.length / generationTasks.length) * 100 : 0;

    setStats({
      totalUploads: uploadedFiles.length,
      totalGenerations: generationTasks.length,
      totalSyntheticRows,
      successRate
    });
  }, [uploadedFiles, generationTasks]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-500" size={16} />;
      case 'failed':
        return <XCircle className="text-red-500" size={16} />;
      case 'processing':
        return <Clock className="text-blue-500 animate-spin" size={16} />;
      default:
        return <Clock className="text-gray-500" size={16} />;
    }
  };

  const recentTasks = generationTasks.slice(-5).reverse();
  const recentFiles = uploadedFiles.slice(-3).reverse();

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Welcome to SynData
        </h1>
        <p className="text-gray-600">
          Generate high-quality synthetic tabular data with automated quality assessment
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Datasets Uploaded</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalUploads}</p>
            </div>
            <Upload className="text-blue-600" size={24} />
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Synthetic Rows</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalSyntheticRows.toLocaleString()}</p>
            </div>
            <BarChart3 className="text-green-600" size={24} />
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Generations</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalGenerations}</p>
            </div>
            <Zap className="text-purple-600" size={24} />
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Success Rate</p>
              <p className="text-2xl font-bold text-gray-900">{stats.successRate.toFixed(1)}%</p>
            </div>
            <Clock className="text-orange-600" size={24} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Quick Start */}
        <div className="bg-white p-6 rounded-lg border">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Start</h2>
          <div className="space-y-4">
            <Link
              to="/upload"
              className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-gray-50 transition-colors"
            >
              <Upload className="text-blue-600" size={20} />
              <div>
                <p className="font-medium text-gray-900">Upload Dataset</p>
                <p className="text-sm text-gray-600">Start by uploading your CSV, JSON, or Excel file</p>
              </div>
            </Link>

            <Link
              to="/generate"
              className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-gray-50 transition-colors"
            >
              <Zap className="text-green-600" size={20} />
              <div>
                <p className="font-medium text-gray-900">Generate Data</p>
                <p className="text-sm text-gray-600">Create synthetic data with quality assessment</p>
              </div>
            </Link>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white p-6 rounded-lg border">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Recent Activity</h2>
          {recentTasks.length === 0 && recentFiles.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-gray-500">No recent activity</p>
              <p className="text-sm text-gray-400 mt-2">
                Upload a dataset to get started
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Recent Generations */}
              {recentTasks.map((task) => (
                <div key={task.task_id || task.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(task.status)}
                    <div>
                      <p className="font-medium text-sm">Generated {task.n_rows} rows</p>
                      <p className="text-xs text-gray-500">
                        {new Date(task.created_at).toLocaleDateString()} • File: {task.file_id}
                      </p>
                    </div>
                  </div>
                  {task.status === 'completed' && (
                    <div className="flex items-center space-x-2">
                      <a
                        href={`http://localhost:8000/download/${task.task_id || task.id}`}
                        className="flex items-center space-x-1 px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                      >
                        <Download size={12} />
                        <span>Data</span>
                      </a>
                      <a
                        href={`http://localhost:8000/download/report/${task.task_id || task.id}`}
                        className="flex items-center space-x-1 px-2 py-1 bg-gray-600 text-white text-xs rounded hover:bg-gray-700"
                      >
                        <FileText size={12} />
                        <span>Report</span>
                      </a>
                    </div>
                  )}
                </div>
              ))}
              
              {/* Recent Uploads */}
              {recentFiles.map((file) => (
                <div key={file.file_id || file.id} className="flex items-center space-x-3 p-3 bg-green-50 rounded-md">
                  <FileText className="text-green-600" size={16} />
                  <div>
                    <p className="font-medium text-sm">{file.filename}</p>
                    <p className="text-xs text-gray-500">
                      Uploaded • {file.rows} rows, {file.columns} columns
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Recent Generations Table */}
      {generationTasks.length > 0 && (
        <div className="mt-8 bg-white p-6 rounded-lg border">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Generation History</h2>
            <Link 
              to="/generate" 
              className="text-blue-600 hover:text-blue-700 text-sm font-medium"
            >
              View All →
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Status</th>
                  <th className="text-left py-2">File ID</th>
                  <th className="text-left py-2">Rows Generated</th>
                  <th className="text-left py-2">Quality</th>
                  <th className="text-left py-2">Created</th>
                  <th className="text-left py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {generationTasks.slice(-10).reverse().map((task) => (
                  <tr key={task.task_id || task.id} className="border-b hover:bg-gray-50">
                    <td className="py-2">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(task.status)}
                        <span className="text-xs">{task.status}</span>
                      </div>
                    </td>
                    <td className="py-2 font-mono text-xs">{task.file_id}</td>
                    <td className="py-2">{task.n_rows?.toLocaleString()}</td>
                    <td className="py-2">
                      {task.quality_report?.overall_score?.overall_quality_score?.toFixed(2) ||
                       (task.status === 'completed' ? 'Good' : '-')}
                    </td>
                    <td className="py-2 text-xs text-gray-500">
                      {new Date(task.created_at).toLocaleDateString()}
                    </td>
                    <td className="py-2">
                      {task.status === 'completed' && (
                        <div className="flex items-center space-x-2">
                          <a
                            href={`http://localhost:8000/download/${task.task_id || task.id}`}
                            className="text-blue-600 hover:text-blue-700 text-xs"
                          >
                            Data
                          </a>
                          <a
                            href={`http://localhost:8000/download/report/${task.task_id || task.id}`}
                            className="text-gray-600 hover:text-gray-700 text-xs"
                          >
                            Report
                          </a>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
