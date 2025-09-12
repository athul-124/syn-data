import { create } from 'zustand';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_BACKEND_URL || `https://${process.env.REPLIT_DEV_DOMAIN || '7414eacd-dce4-4151-9146-c6512c89d7a0-00-18om4ild42nry.pike.replit.dev'}:8000`;

const useStore = create((set, get) => ({
  // Auth state
  user: null,
  isAuthenticated: false,
  
  // Data state
  uploadedFile: null,
  uploadedFiles: [], // Add this array
  previewData: null,
  generationTasks: [],
  currentTask: null,
  
  // UI state
  loading: false,
  error: null,
  
  // Actions
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  
  // File upload with increased timeout and better error handling
  uploadFile: async (file) => {
    set({ loading: true, error: null });
    
    try {
      console.log('Uploading file:', file.name, 'Size:', file.size, 'Type:', file.type);
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000, // Increase to 2 minutes
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log(`Upload progress: ${percentCompleted}%`);
        }
      });
      
      console.log('Upload response:', response.data);
      
      set(state => ({ 
        uploadedFile: response.data,
        uploadedFiles: [...(state.uploadedFiles || []), response.data],
        loading: false 
      }));
      
      return response.data;
    } catch (error) {
      console.error('Upload error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        statusText: error.response?.statusText,
        code: error.code
      });
      
      let errorMessage = 'Upload failed';
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Upload timed out. The file might be too large or the server is slow.';
      } else if (error.code === 'ERR_NETWORK') {
        errorMessage = 'Network error. Check if the backend is running on port 8000.';
      } else if (error.response?.status === 413) {
        errorMessage = 'File too large. Maximum size is 100MB.';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      set({ 
        error: errorMessage,
        loading: false 
      });
      throw error;
    }
  },
  
  // Generate preview
  generatePreview: async (fileId, nRows = 10) => {
    set({ loading: true, error: null });
    
    try {
      const response = await axios.post(`${API_BASE}/preview`, {
        file_id: fileId,
        n_rows: nRows
      });
      
      set({ loading: false });
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Preview failed';
      set({ error: errorMessage, loading: false });
      throw error;
    }
  },
  
  // Start generation task
  startGeneration: async (fileId, nRows, targetColumn = '') => {
    set({ loading: true, error: null });
    
    try {
      console.log('🚀 Starting generation with:', { fileId, nRows, targetColumn });
      
      const formData = new FormData();
      formData.append('file_id', fileId);
      formData.append('n_rows', nRows.toString());
      formData.append('target_column', targetColumn);
      
      console.log('📤 Sending request to /generate-async');
      const response = await axios.post(`${API_BASE}/generate-async`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const task = response.data;
      console.log('✅ Generation response:', task);
      
      set(state => ({
        currentTask: task,
        generationTasks: [...(state.generationTasks || []), task],
        loading: false
      }));
      
      return task;
    } catch (error) {
      console.error('❌ Generation failed:', error);
      console.error('Error details:', {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message
      });
      
      const errorMessage = error.response?.data?.detail || 'Generation failed';
      set({ error: errorMessage, loading: false });
      throw error;
    }
  },
  
  // Poll task status
  pollTaskStatus: async (taskId) => {
    try {
      console.log(`🔄 Polling task: ${taskId}`);
      const response = await axios.get(`${API_BASE}/tasks/${taskId}/status`);
      const updatedTask = response.data;
      
      set(state => ({
        generationTasks: state.generationTasks.map(task => 
          (task.task_id === taskId || task.id === taskId) ? updatedTask : task
        ),
        currentTask: (state.currentTask?.task_id === taskId || state.currentTask?.id === taskId) ? updatedTask : state.currentTask
      }));
      
      return updatedTask;
    } catch (error) {
      console.error('Failed to poll task status:', error);
      return null;
    }
  },
  
  // Get all tasks
  fetchTasks: async () => {
    try {
      const response = await axios.get(`${API_BASE}/tasks`);
      set({ generationTasks: response.data });
      return response.data;
    } catch (error) {
      set({ error: 'Failed to fetch tasks' });
    }
  },
  
  // Check task status
  checkTaskStatus: async (taskId) => {
    try {
      console.log(`🔍 Checking status for task: ${taskId}`);
      const response = await axios.get(`${API_BASE}/tasks/${taskId}/status`);
      const updatedTask = response.data;
      
      console.log(`✅ Task status response:`, updatedTask);
      
      set(state => ({
        currentTask: state.currentTask?.task_id === taskId ? updatedTask : state.currentTask,
        generationTasks: state.generationTasks.map(task => 
          (task.task_id === taskId || task.id === taskId) ? updatedTask : task
        )
      }));
      
      return updatedTask;
    } catch (error) {
      console.error('Failed to check task status:', error);
      console.error('Error details:', {
        status: error.response?.status,
        data: error.response?.data,
        url: error.config?.url
      });
      throw error;
    }
  },
  
  fetchGenerationTasks: async () => {
    try {
      const response = await axios.get(`${API_BASE}/tasks`);
      set({ generationTasks: response.data });
      return response.data;
    } catch (error) {
      console.error('Failed to fetch generation tasks:', error);
      set({ generationTasks: [] });
      return [];
    }
  },
  
  fetchUploadedFiles: async () => {
    try {
      // For now, we'll use the existing uploadedFiles array
      // In a real app, you'd fetch from an API endpoint
      const { uploadedFiles } = get();
      return uploadedFiles;
    } catch (error) {
      console.error('Failed to fetch uploaded files:', error);
      set({ error: error.message });
    }
  }
}));

export default useStore;














