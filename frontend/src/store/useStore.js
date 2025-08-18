import { create } from 'zustand';
import axios from 'axios';

const API_BASE = 'http://localhost:8000'; // Make sure this points to your FastAPI backend

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
  
  // File upload
  uploadFile: async (file) => {
    set({ loading: true, error: null });
    
    try {
      console.log('Uploading file:', file.name, 'Size:', file.size, 'Type:', file.type);
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000 // Increase timeout
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
        statusText: error.response?.statusText
      });
      
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.message || 
                          error.message || 
                          'Upload failed';
      
      set({ 
        error: 'Upload failed: ' + errorMessage,
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
      const formData = new FormData();
      formData.append('file_id', fileId);
      formData.append('n_rows', nRows.toString());
      formData.append('target_column', targetColumn);
      
      const response = await axios.post(`${API_BASE}/generate-async`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const task = response.data;
      
      set(state => ({
        currentTask: task,
        generationTasks: [...(state.generationTasks || []), task],
        loading: false
      }));
      
      return task;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Generation failed';
      set({ error: errorMessage, loading: false });
      throw error;
    }
  },
  
  // Poll task status
  pollTaskStatus: async (taskId) => {
    try {
      const response = await axios.get(`${API_BASE}/tasks/${taskId}/status`);
      const updatedTask = response.data;
      
      set(state => ({
        generationTasks: state.generationTasks.map(task => 
          task.id === taskId ? updatedTask : task
        ),
        currentTask: state.currentTask?.id === taskId ? updatedTask : state.currentTask
      }));
      
      return updatedTask;
    } catch (error) {
      console.error('Failed to poll task status:', error);
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
      const response = await axios.get(`${API_BASE}/tasks/${taskId}/status`);
      const updatedTask = response.data;
      
      set(state => ({
        currentTask: state.currentTask?.task_id === taskId ? updatedTask : state.currentTask,
        generationTasks: state.generationTasks.map(task => 
          task.task_id === taskId ? updatedTask : task
        )
      }));
      
      return updatedTask;
    } catch (error) {
      console.error('Failed to check task status:', error);
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











