#!/usr/bin/env python3
"""
Progress Tracker for MVP Development
Tracks tasks, milestones, and overall project progress through MVP phases.
"""

import json
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

class TaskStatus(Enum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    BLOCKED = "Blocked"
    ON_HOLD = "On Hold"

class MVPPhase(Enum):
    DISCOVERY = "Phase I: Discovery and Research"
    PRIORITIZATION = "Phase II: Prioritization and Planning"
    DEVELOPMENT = "Phase III: Development and Implementation"
    TESTING = "Phase IV: Testing and Validation"
    LAUNCH = "Phase V: Launch and Iteration"

class MilestoneType(Enum):
    PHASE_GATE = "Phase Gate"
    DELIVERABLE = "Deliverable"
    REVIEW = "Review"
    RELEASE = "Release"

@dataclass
class Milestone:
    id: str
    name: str
    description: str
    milestone_type: MilestoneType
    target_date: str
    completion_date: Optional[str] = None
    success_criteria: List[str] = None
    is_completed: bool = False
    
    def __post_init__(self):
        if self.success_criteria is None:
            self.success_criteria = []

@dataclass
class MVPMetric:
    name: str
    description: str
    target_value: float
    current_value: float = 0.0
    unit: str = ""
    category: str = "General"

@dataclass
class Task:
    id: str
    name: str
    description: str
    phase: str
    status: TaskStatus = TaskStatus.NOT_STARTED
    assigned_to: Optional[str] = None
    estimated_hours: int = 0
    actual_hours: int = 0
    start_date: Optional[str] = None
    due_date: Optional[str] = None
    completion_date: Optional[str] = None
    dependencies: List[str] = None
    blockers: List[str] = None
    notes: str = ""
    created_date: str = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.blockers is None:
            self.blockers = []
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()
    
    @property
    def is_overdue(self) -> bool:
        if not self.due_date or self.status == TaskStatus.COMPLETED:
            return False
        return datetime.fromisoformat(self.due_date) < datetime.now()
    
    @property
    def progress_percentage(self) -> float:
        if self.status == TaskStatus.COMPLETED:
            return 100.0
        elif self.status == TaskStatus.IN_PROGRESS:
            if self.estimated_hours > 0:
                return min((self.actual_hours / self.estimated_hours) * 100, 95.0)
            return 50.0
        return 0.0

class MVPProgressTracker:
    """Main progress tracker for MVP development"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_start_date = datetime.now().isoformat()
        self.current_phase = "Phase III: Development and Implementation"
        self.tasks: List[Task] = []
        self.phases = []
        self._initialize_sample_tasks()
    
    def _initialize_sample_tasks(self):
        """Initialize with sample SynData MVP tasks"""
        sample_tasks = [
            Task(
                id="T003",
                name="Feature Prioritization Workshop",
                description="Finalize MVP scope with RICE scoring",
                phase="Phase II: Prioritization and Planning",
                status=TaskStatus.IN_PROGRESS,
                estimated_hours=12,
                actual_hours=8,
                due_date=(datetime.now() + timedelta(days=2)).isoformat()
            ),
            Task(
                id="T005",
                name="Synthetic Data Generator",
                description="Enhanced generator with SDV integration",
                phase="Phase III: Development and Implementation",
                status=TaskStatus.IN_PROGRESS,
                estimated_hours=32,
                actual_hours=20,
                due_date=(datetime.now() + timedelta(days=5)).isoformat()
            ),
            Task(
                id="T006",
                name="Quality Report System",
                description="Comprehensive quality scoring with fidelity analysis",
                phase="Phase III: Development and Implementation",
                status=TaskStatus.NOT_STARTED,
                estimated_hours=24,
                actual_hours=0,
                due_date=(datetime.now() + timedelta(days=8)).isoformat()
            )
        ]
        self.tasks.extend(sample_tasks)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def complete_task(self, task_id: str, completion_notes: str = ""):
        """Mark task as completed and update progress"""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.actual_hours = task.estimated_hours
            task.completion_date = datetime.now().isoformat()
            task.notes = completion_notes
            print(f"✅ Task {task_id} completed: {task.name}")
        else:
            print(f"❌ Task {task_id} not found")
    
    def generate_status_report(self) -> Dict:
        """Generate comprehensive status report"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        
        return {
            "project_name": self.project_name,
            "current_phase": self.current_phase,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_percentage": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "tasks": [asdict(task) for task in self.tasks]
        }
    
    def save_to_file(self, filename: str):
        """Save progress to JSON file"""
        data = {
            "project_name": self.project_name,
            "project_start_date": self.project_start_date,
            "current_phase": self.current_phase,
            "tasks": []
        }
        
        for task in self.tasks:
            task_dict = asdict(task)
            task_dict["status"] = task.status.value
            data["tasks"].append(task_dict)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str):
        """Load progress from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.project_name = data.get("project_name", self.project_name)
        self.project_start_date = data.get("project_start_date", self.project_start_date)
        self.current_phase = data.get("current_phase", self.current_phase)
        
        self.tasks = []
        for task_data in data.get("tasks", []):
            # Remove computed properties that shouldn't be in constructor
            task_dict = task_data.copy()
            task_dict.pop('is_overdue', None)
            task_dict.pop('progress_percentage', None)
            
            # Convert status string back to enum
            task_dict["status"] = TaskStatus(task_dict["status"])
            
            self.tasks.append(Task(**task_dict))

    def load_from_json(self, filename: str):
        """Load progress from JSON file (alias for load_from_file)"""
        self.load_from_file(filename)

    def save_to_json(self, filename: str):
        """Save progress to JSON file (alias for save_to_file)"""
        self.save_to_file(filename)

    def get_overall_progress(self) -> Dict:
        """Get overall progress summary"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        
        return {
            'project_name': self.project_name,
            'current_phase': self.current_phase,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'task_progress_percentage': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'overdue_tasks': len([t for t in self.tasks if t.is_overdue])
        }

    def get_upcoming_milestones(self, days: int = 30) -> List:
        """Get upcoming milestones (placeholder)"""
        return []

def create_syndata_mvp_tracker() -> MVPProgressTracker:
    """Create a sample MVP tracker with SynData project data"""
    return MVPProgressTracker("SynData MVP Demo")
