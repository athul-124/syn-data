import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Database, Upload, Zap, List } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: Database },
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/generate', label: 'Generate', icon: Zap },
    { path: '/tasks', label: 'Tasks', icon: List }
  ];
  
  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-8">
            <Link to="/" className="text-xl font-bold text-blue-600">
              SynData
            </Link>
            
            <div className="flex space-x-4">
              {navItems.map(({ path, label, icon: Icon }) => (
                <Link
                  key={path}
                  to={path}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    location.pathname === path
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Icon size={16} />
                  <span>{label}</span>
                </Link>
              ))}
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-600">Free Plan</span>
            <button className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
              Upgrade
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;