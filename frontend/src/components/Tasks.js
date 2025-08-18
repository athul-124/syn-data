import React from 'react';

const Tasks = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-2">Generation Tasks</h1>
      <p className="text-gray-600 mb-8">Monitor your data generation tasks</p>
      
      <div className="bg-white p-8 rounded-lg border text-center">
        <p className="text-gray-500">No tasks yet</p>
      </div>
    </div>
  );
};

export default Tasks;