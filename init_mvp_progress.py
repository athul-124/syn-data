#!/usr/bin/env python3
"""Initialize MVP progress tracking"""

from mvp_tools.progress_tracker import create_syndata_mvp_tracker

def main():
    """Initialize MVP progress file"""
    print("🚀 Initializing MVP progress tracking...")
    
    # Create tracker with sample data
    tracker = create_syndata_mvp_tracker()
    
    # Save to file
    tracker.save_to_file("mvp_progress.json")
    
    print("✅ MVP progress file created: mvp_progress.json")
    print("📊 Sample tasks loaded for SynData MVP")

if __name__ == "__main__":
    main()