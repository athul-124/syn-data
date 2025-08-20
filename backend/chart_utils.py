"""
Chart utilities for quality reporting visualization
"""
import os
import base64
from typing import Dict, Optional
import matplotlib.pyplot as plt

def save_charts_to_base64(charts: Dict[str, str]) -> Dict[str, str]:
    """Convert chart files to base64 strings for API response"""
    
    base64_charts = {}
    
    for chart_name, file_path in charts.items():
        if chart_name == "error":
            continue
            
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    img_data = f.read()
                    base64_str = base64.b64encode(img_data).decode('utf-8')
                    base64_charts[chart_name] = f"data:image/png;base64,{base64_str}"
                
                # Clean up temporary file
                os.remove(file_path)
            
        except Exception as e:
            print(f"Error processing chart {chart_name}: {str(e)}")
            base64_charts[chart_name] = f"Error: {str(e)}"
    
    return base64_charts

def cleanup_temp_charts():
    """Clean up any remaining temporary chart files"""
    temp_files = [
        'temp_distributions.png',
        'temp_correlations.png', 
        'temp_means.png',
        'temp_categorical.png'
    ]
    
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove {file_path}: {str(e)}")