"""
MVP Dashboard - Comprehensive Overview and Management Interface

Provides a unified dashboard for monitoring MVP development progress,
metrics, feedback, and key insights as outlined in the Strategic Blueprint.
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
from dataclasses import asdict

from .feature_prioritization import FeaturePrioritizer, Priority as FeaturePriority
from .user_research import UserResearchManager
from .progress_tracker import MVPProgressTracker, MVPPhase
from .feedback_system import FeedbackManager, FeedbackType, Priority as FeedbackPriority


class MVPDashboard:
    """Main dashboard class that aggregates all MVP tools and provides unified insights"""
    
    def __init__(self):
        self.feature_prioritizer = FeaturePrioritizer()
        self.user_research = UserResearchManager()
        self.progress_tracker = MVPProgressTracker("Default MVP Project")
        self.feedback_manager = FeedbackManager()
        
        # Dashboard metadata
        self.dashboard_name = "MVP Development Dashboard"
        self.last_updated = datetime.now().isoformat()
    
    def load_all_data(self, feature_file: str = None, research_file: str = None, 
                     progress_file: str = None, feedback_file: str = None) -> None:
        """Load data from all MVP tools"""
        try:
            if feature_file:
                self.feature_prioritizer.load_from_json(feature_file)
            if research_file:
                self.user_research.load_from_json(research_file)
            if progress_file:
                self.progress_tracker.load_from_json(progress_file)
            if feedback_file:
                self.feedback_manager.load_from_json(feedback_file)
            
            self.last_updated = datetime.now().isoformat()
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def save_all_data(self, feature_file: str = "mvp_features.json", 
                     research_file: str = "user_research_data.json",
                     progress_file: str = "mvp_progress.json", 
                     feedback_file: str = "feedback_data.json") -> None:
        """Save data from all MVP tools"""
        try:
            self.feature_prioritizer.save_to_json(feature_file)
            self.user_research.save_to_json(research_file)
            self.progress_tracker.save_to_json(progress_file)
            self.feedback_manager.save_to_json(feedback_file)
            print(f"All MVP data saved successfully")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def get_executive_summary(self) -> Dict:
        """Generate executive summary for stakeholders"""
        # Get data from all components
        feature_report = self.feature_prioritizer.generate_prioritization_report()
        progress_report = self.progress_tracker.get_overall_progress()
        feedback_analysis = self.feedback_manager.analyze_feedback() if self.feedback_manager.feedback_list else {}
        
        # Calculate key metrics
        mvp_completion = progress_report.get('task_progress_percentage', 0)
        feature_completion = len([f for f in self.feature_prioritizer.features 
                                if f.status == 'done']) / len(self.feature_prioritizer.features) * 100 if self.feature_prioritizer.features else 0
        
        # Risk assessment
        risks = []
        overdue_tasks = len([t for t in self.progress_tracker.tasks if t.is_overdue])
        if overdue_tasks > 0:
            risks.append(f"{overdue_tasks} overdue tasks")
        
        if feedback_analysis and feedback_analysis.get('sentiment_analysis', {}).get('average_sentiment', 0) < -0.2:
            risks.append("Declining user sentiment")
        
        critical_feedback = len([f for f in self.feedback_manager.feedback_list 
                               if f.priority == FeedbackPriority.CRITICAL])
        if critical_feedback > 0:
            risks.append(f"{critical_feedback} critical feedback items")
        
        return {
            'dashboard_name': self.dashboard_name,
            'last_updated': self.last_updated,
            'project_status': {
                'current_phase': progress_report.get('current_phase', 'Unknown'),
                'mvp_completion_percentage': mvp_completion,
                'feature_completion_percentage': feature_completion,
                'total_features': feature_report.get('total_features', 0),
                'mvp_features': feature_report.get('mvp_features_count', 0)
            },
            'key_metrics': {
                'total_tasks': progress_report.get('total_tasks', 0),
                'completed_tasks': progress_report.get('completed_tasks', 0),
                'total_feedback': len(self.feedback_manager.feedback_list),
                'user_personas': len(self.user_research.personas),
                'user_stories': len(self.user_research.user_stories)
            },
            'health_indicators': {
                'on_track': mvp_completion >= 70 and len(risks) == 0,
                'at_risk': len(risks) > 0 and len(risks) <= 2,
                'critical': len(risks) > 2 or mvp_completion < 50,
                'risks': risks
            },
            'next_milestones': [
                m.to_dict() for m in self.progress_tracker.get_upcoming_milestones(14)[:3]
            ]
        }
    
    def get_feature_insights(self) -> Dict:
        """Get insights about feature prioritization and development"""
        feature_report = self.feature_prioritizer.generate_prioritization_report()
        
        # Analyze feature distribution
        must_haves = len(self.feature_prioritizer.get_moscow_features(FeaturePriority.MUST_HAVE))
        should_haves = len(self.feature_prioritizer.get_moscow_features(FeaturePriority.SHOULD_HAVE))
        could_haves = len(self.feature_prioritizer.get_moscow_features(FeaturePriority.COULD_HAVE))
        
        # Feature status analysis
        status_counts = {}
        for feature in self.feature_prioritizer.features:
            status_counts[feature.status] = status_counts.get(feature.status, 0) + 1
        
        # Top RICE features
        top_rice_features = self.feature_prioritizer.get_features_by_rice_score()[:5]
        
        return {
            'total_features': len(self.feature_prioritizer.features),
            'moscow_distribution': {
                'must_have': must_haves,
                'should_have': should_haves,
                'could_have': could_haves,
                'wont_have': len(self.feature_prioritizer.get_moscow_features(FeaturePriority.WONT_HAVE))
            },
            'status_distribution': status_counts,
            'mvp_scope': {
                'features_count': feature_report.get('mvp_features_count', 0),
                'estimated_effort': feature_report.get('estimated_mvp_effort', 0)
            },
            'top_priority_features': [
                {
                    'name': f.name,
                    'rice_score': f.rice_score,
                    'moscow_priority': f.moscow_priority.value,
                    'status': f.status
                } for f in top_rice_features
            ]
        }
    
    def get_user_insights(self) -> Dict:
        """Get insights about users and their needs"""
        persona_summary = self.user_research.generate_persona_summary()
        story_backlog = self.user_research.generate_user_story_backlog()
        
        # Analyze pain points and goals
        top_pain_points = persona_summary.get('top_pain_points', [])[:5]
        top_goals = persona_summary.get('top_goals', [])[:5]
        
        # Story analysis
        critical_stories = len(story_backlog.get('backlog_by_priority', {}).get('Critical', []))
        high_stories = len(story_backlog.get('backlog_by_priority', {}).get('High', []))
        
        return {
            'total_personas': persona_summary.get('total_personas', 0),
            'segment_distribution': persona_summary.get('segment_distribution', {}),
            'top_pain_points': [{'pain_point': pp[0], 'frequency': pp[1]} for pp in top_pain_points],
            'top_goals': [{'goal': g[0], 'frequency': g[1]} for g in top_goals],
            'user_stories': {
                'total': story_backlog.get('total_stories', 0),
                'critical': critical_stories,
                'high_priority': high_stories,
                'story_points_total': story_backlog.get('story_points_total', 0)
            }
        }
    
    def get_progress_insights(self) -> Dict:
        """Get insights about development progress"""
        overall_progress = self.progress_tracker.get_overall_progress()
        
        # Phase analysis
        phase_progress = {}
        for phase in MVPPhase:
            phase_data = self.progress_tracker.get_phase_progress(phase)
            phase_progress[phase.name] = {
                'progress_percentage': phase_data.get('progress_percentage', 0),
                'total_tasks': phase_data.get('total_tasks', 0),
                'completed_tasks': phase_data.get('completed_tasks', 0),
                'overdue_tasks': phase_data.get('overdue_tasks', 0)
            }
        
        # Timeline analysis
        project_start = datetime.fromisoformat(self.progress_tracker.project_start_date)
        days_elapsed = (datetime.now() - project_start).days
        
        # Milestone analysis
        upcoming_milestones = self.progress_tracker.get_upcoming_milestones(30)
        overdue_milestones = [m for m in self.progress_tracker.milestones if m.is_overdue]
        
        return {
            'overall_progress': overall_progress['task_progress_percentage'],
            'current_phase': overall_progress['current_phase'],
            'project_timeline': {
                'days_elapsed': days_elapsed,
                'start_date': self.progress_tracker.project_start_date
            },
            'phase_breakdown': phase_progress,
            'milestones': {
                'total': len(self.progress_tracker.milestones),
                'completed': len([m for m in self.progress_tracker.milestones if m.is_completed]),
                'upcoming': len(upcoming_milestones),
                'overdue': len(overdue_milestones)
            },
            'task_health': {
                'total_tasks': overall_progress['total_tasks'],
                'completed_tasks': overall_progress['completed_tasks'],
                'overdue_tasks': overall_progress['overdue_tasks']
            }
        }
    
    def get_feedback_insights(self) -> Dict:
        """Get insights about user feedback"""
        if not self.feedback_manager.feedback_list:
            return {'message': 'No feedback data available'}
        
        feedback_analysis = self.feedback_manager.analyze_feedback()
        
        # Recent feedback trends
        recent_feedback = [f for f in self.feedback_manager.feedback_list 
                          if (datetime.now() - datetime.fromisoformat(f.created_date)).days <= 7]
        
        # Priority distribution
        priority_counts = {}
        for feedback in self.feedback_manager.feedback_list:
            priority_counts[feedback.priority.value] = priority_counts.get(feedback.priority.value, 0) + 1
        
        return {
            'total_feedback': len(self.feedback_manager.feedback_list),
            'recent_feedback': len(recent_feedback),
            'resolution_rate': feedback_analysis.get('resolution_rate', 0),
            'sentiment_analysis': feedback_analysis.get('sentiment_analysis', {}),
            'type_distribution': feedback_analysis.get('type_distribution', {}),
            'priority_distribution': priority_counts,
            'common_themes': feedback_analysis.get('common_themes', [])[:5],
            'key_insights': [i['title'] for i in feedback_analysis.get('insights', [])],
            'urgent_items': len([f for f in self.feedback_manager.feedback_list 
                               if f.priority == FeedbackPriority.CRITICAL])
        }
    
    def get_build_measure_learn_status(self) -> Dict:
        """Assess the build-measure-learn feedback loop status"""
        # Build phase assessment
        development_tasks = self.progress_tracker.get_tasks_by_phase(MVPPhase.DEVELOPMENT)
        build_progress = sum(t.progress_percentage for t in development_tasks) / len(development_tasks) if development_tasks else 0
        
        # Measure phase assessment
        metrics_count = len(self.progress_tracker.metrics)
        metrics_with_data = len([m for m in self.progress_tracker.metrics if m.current_value > 0])
        
        # Learn phase assessment
        feedback_count = len(self.feedback_manager.feedback_list)
        insights_count = len(self.feedback_manager.insights)
        
        # Determine current loop stage
        if build_progress < 50:
            current_stage = "Build"
        elif metrics_with_data < metrics_count * 0.5:
            current_stage = "Measure"
        else:
            current_stage = "Learn"
        
        return {
            'current_stage': current_stage,
            'build_status': {
                'progress_percentage': build_progress,
                'development_tasks': len(development_tasks),
                'completed_features': len([f for f in self.feature_prioritizer.features if f.status == 'done'])
            },
            'measure_status': {
                'total_metrics': metrics_count,
                'active_metrics': metrics_with_data,
                'measurement_coverage': (metrics_with_data / metrics_count * 100) if metrics_count > 0 else 0
            },
            'learn_status': {
                'feedback_collected': feedback_count,
                'insights_generated': insights_count,
                'learning_rate': (insights_count / feedback_count * 100) if feedback_count > 0 else 0
            },
            'loop_health': {
                'is_active': build_progress > 0 and feedback_count > 0,
                'cycle_completeness': min(build_progress, metrics_with_data / max(metrics_count, 1) * 100, feedback_count * 10)
            }
        }
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate a comprehensive MVP dashboard report"""
        return {
            'executive_summary': self.get_executive_summary(),
            'feature_insights': self.get_feature_insights(),
            'user_insights': self.get_user_insights(),
            'progress_insights': self.get_progress_insights(),
            'feedback_insights': self.get_feedback_insights(),
            'build_measure_learn': self.get_build_measure_learn_status(),
            'generated_at': datetime.now().isoformat()
        }
    
    def get_action_items(self) -> List[Dict]:
        """Generate prioritized action items based on current status"""
        action_items = []
        
        # Get insights from all components
        executive_summary = self.get_executive_summary()
        progress_insights = self.get_progress_insights()
        feedback_insights = self.get_feedback_insights()
        
        # Critical issues first
        if executive_summary['health_indicators']['critical']:
            action_items.append({
                'priority': 'Critical',
                'category': 'Project Health',
                'title': 'Address Critical Project Issues',
                'description': f"Project has {len(executive_summary['health_indicators']['risks'])} critical risks",
                'action': 'Review and address all identified risks immediately',
                'due_date': (datetime.now() + timedelta(days=1)).isoformat()
            })
        
        # Overdue tasks
        if progress_insights['task_health']['overdue_tasks'] > 0:
            action_items.append({
                'priority': 'High',
                'category': 'Task Management',
                'title': 'Resolve Overdue Tasks',
                'description': f"{progress_insights['task_health']['overdue_tasks']} tasks are overdue",
                'action': 'Review overdue tasks and update timelines or reassign resources',
                'due_date': (datetime.now() + timedelta(days=3)).isoformat()
            })
        
        # Critical feedback
        if feedback_insights.get('urgent_items', 0) > 0:
            action_items.append({
                'priority': 'High',
                'category': 'User Feedback',
                'title': 'Address Critical Feedback',
                'description': f"{feedback_insights['urgent_items']} critical feedback items need attention",
                'action': 'Review and respond to critical feedback items',
                'due_date': (datetime.now() + timedelta(days=2)).isoformat()
            })
        
        # Low user feedback
        if feedback_insights.get('total_feedback', 0) < 5:
            action_items.append({
                'priority': 'Medium',
                'category': 'User Research',
                'title': 'Increase User Feedback Collection',
                'description': 'Limited user feedback available for learning',
                'action': 'Implement feedback collection mechanisms and reach out to users',
                'due_date': (datetime.now() + timedelta(days=7)).isoformat()
            })
        
        # Feature completion
        feature_insights = self.get_feature_insights()
        if feature_insights['mvp_scope']['features_count'] > 0:
            completed_features = len([f for f in self.feature_prioritizer.features if f.status == 'done'])
            if completed_features / feature_insights['mvp_scope']['features_count'] < 0.8:
                action_items.append({
                    'priority': 'Medium',
                    'category': 'Feature Development',
                    'title': 'Focus on MVP Feature Completion',
                    'description': f"Only {completed_features} of {feature_insights['mvp_scope']['features_count']} MVP features completed",
                    'action': 'Prioritize completion of remaining MVP features',
                    'due_date': (datetime.now() + timedelta(days=14)).isoformat()
                })
        
        return sorted(action_items, key=lambda x: {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}[x['priority']])
    
    def save_dashboard_report(self, filename: str = "mvp_dashboard_report.json") -> None:
        """Save comprehensive dashboard report to file"""
        report = self.generate_comprehensive_report()
        report['action_items'] = self.get_action_items()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Dashboard report saved to {filename}")


def create_sample_mvp_dashboard() -> MVPDashboard:
    """Create a sample MVP dashboard with test data"""
    from .feature_prioritization import create_syndata_mvp_features
    from .user_research import create_syndata_user_research
    from .progress_tracker import create_syndata_mvp_tracker
    from .feedback_system import create_syndata_feedback_system
    
    dashboard = MVPDashboard()
    
    # Load sample data
    dashboard.feature_prioritizer = create_syndata_mvp_features()
    dashboard.user_research = create_syndata_user_research()
    dashboard.progress_tracker = create_syndata_mvp_tracker()
    dashboard.feedback_manager = create_syndata_feedback_system()
    
    return dashboard


if __name__ == "__main__":
    # Example usage
    dashboard = create_sample_mvp_dashboard()
    
    # Generate executive summary
    summary = dashboard.get_executive_summary()
    print("=== MVP Dashboard Executive Summary ===")
    print(f"Project Status: {summary['project_status']['current_phase']}")
    print(f"MVP Completion: {summary['project_status']['mvp_completion_percentage']:.1f}%")
    print(f"Feature Completion: {summary['project_status']['feature_completion_percentage']:.1f}%")
    
    health = summary['health_indicators']
    if health['critical']:
        print("🔴 Project Status: CRITICAL")
    elif health['at_risk']:
        print("🟡 Project Status: AT RISK")
    else:
        print("🟢 Project Status: ON TRACK")
    
    if health['risks']:
        print("⚠️  Risks:")
        for risk in health['risks']:
            print(f"   - {risk}")
    
    # Show action items
    action_items = dashboard.get_action_items()
    if action_items:
        print(f"\n📋 Action Items ({len(action_items)}):")
        for item in action_items[:5]:  # Show top 5
            print(f"   {item['priority']}: {item['title']}")
    
    # Build-Measure-Learn status
    bml_status = dashboard.get_build_measure_learn_status()
    print(f"\n🔄 Build-Measure-Learn: Currently in '{bml_status['current_stage']}' phase")
    print(f"   Build: {bml_status['build_status']['progress_percentage']:.1f}%")
    print(f"   Measure: {bml_status['measure_status']['measurement_coverage']:.1f}%")
    print(f"   Learn: {bml_status['learn_status']['learning_rate']:.1f}%")
    
    # Save comprehensive report
    dashboard.save_dashboard_report()
    print("\nComprehensive dashboard report saved to mvp_dashboard_report.json")
