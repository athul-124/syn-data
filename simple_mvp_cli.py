#!/usr/bin/env python3
"""
Simplified MVP CLI for SynData project
"""

import sys
import json
import argparse
from datetime import datetime

# Add mvp_tools to path
sys.path.append('mvp_tools')

try:
    from progress_tracker import MVPProgressTracker, TaskStatus, create_syndata_mvp_tracker
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def show_status():
    """Show current project status"""
    try:
        tracker = MVPProgressTracker("SynData MVP")
        tracker.load_from_file("mvp_progress.json")
        
        report = tracker.generate_status_report()
        
        print("=" * 60)
        print("📊 MVP PROJECT STATUS")
        print("=" * 60)
        
        print(f"🎯 Project: {report['project_name']}")
        print(f"📈 Current Phase: {report['current_phase']}")
        print(f"✅ Progress: {report['completion_percentage']:.1f}%")
        print(f"📋 Tasks: {report['completed_tasks']}/{report['total_tasks']} completed")
        
        # Show task details
        print(f"\n📋 TASK BREAKDOWN:")
        for task_data in report['tasks']:
            status_icon = "✅" if task_data['status'] == 'Completed' else "🔄" if task_data['status'] == 'In Progress' else "⏳"
            print(f"   {status_icon} {task_data['name']} ({task_data['status']})")
        
        # Health check
        overdue_tasks = [t for t in tracker.tasks if t.is_overdue]
        if overdue_tasks:
            print(f"\n⚠️  OVERDUE TASKS: {len(overdue_tasks)}")
            for task in overdue_tasks:
                print(f"   🔴 {task.name}")
        else:
            print(f"\n🟢 No overdue tasks!")
        
    except FileNotFoundError:
        print("❌ Progress file not found. Run 'python init_mvp_progress.py' first")
    except Exception as e:
        print(f"❌ Error: {e}")

def init_project(project_name: str, with_sample: bool = False):
    """Initialize MVP project"""
    print(f"🚀 Initializing MVP project: {project_name}")
    
    if with_sample:
        tracker = create_syndata_mvp_tracker()
        tracker.project_name = project_name
    else:
        tracker = MVPProgressTracker(project_name)
    
    tracker.save_to_file("mvp_progress.json")
    print(f"✅ Project '{project_name}' initialized!")
    print("📁 Created: mvp_progress.json")

def complete_task(task_id: str, notes: str = ""):
    """Complete a specific task"""
    try:
        tracker = MVPProgressTracker("SynData MVP")
        tracker.load_from_file("mvp_progress.json")
        
        tracker.complete_task(task_id, notes)
        tracker.save_to_file("mvp_progress.json")
        
        print(f"💾 Progress saved")
        
    except FileNotFoundError:
        print("❌ Progress file not found. Run 'python init_mvp_progress.py' first")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Simplified MVP CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show project status')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize project')
    init_parser.add_argument('project_name', help='Project name')
    init_parser.add_argument('--sample', action='store_true', help='Use sample data')
    
    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Complete a task')
    complete_parser.add_argument('task_id', help='Task ID (e.g., T003)')
    complete_parser.add_argument('--notes', help='Completion notes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'status':
        show_status()
    elif args.command == 'init':
        init_project(args.project_name, args.sample)
    elif args.command == 'complete':
        complete_task(args.task_id, args.notes or "")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()



