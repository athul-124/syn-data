#!/usr/bin/env python3
"""
MVP Development CLI Tool

Command-line interface for managing MVP development using the Strategic Blueprint methodology.
Provides easy access to all MVP tools: feature prioritization, user research, progress tracking,
feedback management, and dashboard reporting.
"""

import sys
import json
import argparse
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED
from datetime import datetime, timedelta
from typing import Optional

from mvp_tools.feature_prioritization import (
    FeaturePrioritizer, Feature, Priority as FeaturePriority,
    create_syndata_mvp_features
)
from mvp_tools.user_research import (
    UserResearchManager, UserPersona, UserStory, ResearchInsight,
    UserSegment, ResearchMethod, create_syndata_user_research
)
from mvp_tools.progress_tracker import (
    MVPProgressTracker, Task, Milestone, MVPMetric, MVPPhase, TaskStatus,
    MilestoneType, create_syndata_mvp_tracker
)
from mvp_tools.feedback_system import (
    FeedbackManager, UserFeedback, FeedbackType, FeedbackSource,
    Priority as FeedbackPriority, create_syndata_feedback_system
)
from mvp_tools.mvp_dashboard import MVPDashboard, create_sample_mvp_dashboard

console = Console()


class MVPCLIManager:
    """Main CLI manager for MVP development tools"""
    
    def __init__(self):
        self.console = console
        self.dashboard = MVPDashboard()
    
    def init_project(self, project_name: str, with_sample_data: bool = False) -> None:
        """Initialize a new MVP project"""
        self.console.print(f"🚀 Initializing MVP project: [bold cyan]{project_name}[/bold cyan]")
        
        if with_sample_data:
            self.console.print("📊 Loading sample data for SynData project...")
            self.dashboard = create_sample_mvp_dashboard()
            self.dashboard.progress_tracker.project_name = project_name
        else:
            self.dashboard.progress_tracker.project_name = project_name
        
        # Save initial project files
        self.dashboard.save_all_data()
        self.console.print(f"✅ Project '[bold green]{project_name}[/bold green]' initialized successfully!")
        self.console.print("📁 Created files:")
        self.console.print("   - [green]mvp_features.json[/green]")
        self.console.print("   - [green]user_research_data.json[/green]")
        self.console.print("   - [green]mvp_progress.json[/green]")
        self.console.print("   - [green]feedback_data.json[/green]")
    
    def load_project(self) -> None:
        """Load existing project data"""
        try:
            self.dashboard.load_all_data(
                "mvp_features.json",
                "user_research_data.json", 
                "mvp_progress.json",
                "feedback_data.json"
            )
            self.console.print("✅ Project data loaded successfully!")
        except Exception as e:
            self.console.print(f"❌ [bold red]Error loading project data:[/] {e}")
            self.console.print("💡 Try running '[bold]mvp init --sample[/bold]' to create a new project.")
            sys.exit(1)
    
    def show_status(self) -> None:
        """Show current project status"""
        self.load_project()
        summary = self.dashboard.get_executive_summary()
        
        title = f"📊 MVP PROJECT STATUS - {summary['project_status']['current_phase']}"
        self.console.print(Panel(title, title_align="left", border_style="bold blue"))
        
        # Project health
        health = summary['health_indicators']
        if health['critical']:
            status_icon = "🔴"
            status_text = "CRITICAL"
            style = "bold red"
        elif health['at_risk']:
            status_icon = "🟡"
            status_text = "AT RISK"
            style = "bold yellow"
        else:
            status_icon = "🟢"
            status_text = "ON TRACK"
            style = "bold green"
        
        self.console.print(f"{status_icon} [b]Project Health:[/] [{style}]{status_text}[/]")
        
        # Progress metrics
        self.console.print(f"📈 [b]MVP Completion:[/] {summary['project_status']['mvp_completion_percentage']:.1f}%")
        self.console.print(f"🎯 [b]Features:[/] {summary['project_status']['mvp_features']}/{summary['project_status']['total_features']} MVP features")
        self.console.print(f"✅ [b]Tasks:[/] {summary['key_metrics']['completed_tasks']}/{summary['key_metrics']['total_tasks']} completed")
        self.console.print(f"💬 [b]Feedback:[/] {summary['key_metrics']['total_feedback']} items collected")
        
        # Risks and issues
        if health['risks']:
            self.console.print(f"\n⚠️  [bold]RISKS IDENTIFIED:[/]")
            for risk in health['risks']:
                self.console.print(f"   • [yellow]{risk}[/yellow]")
        
        # Next milestones
        if summary['next_milestones']:
            milestone_table = Table.grid(padding=(0, 2))
            milestone_table.add_column()
            for milestone in summary['next_milestones']:
                days_until = milestone['days_until_due']
                if days_until < 0:
                    text = f"🔴 {milestone['name']} (OVERDUE)"

    
    def show_features(self) -> None:
        """Show feature prioritization summary"""
        try:
            self.load_project()
            insights = self.dashboard.get_feature_insights()
            
            print("=" * 60)
            print("🎯 FEATURE PRIORITIZATION")
            print("=" * 60)
            
            print(f"📊 Total Features: {insights['total_features']}")
            print(f"🎯 MVP Scope: {insights['mvp_scope']['features_count']} features")
            print(f"⏱️  Estimated Effort: {insights['mvp_scope']['estimated_effort']} hours")
            
            # MoSCoW distribution
            moscow = insights['moscow_distribution']
            print(f"\n📋 MoSCoW BREAKDOWN:")
            print(f"   🔴 Must Have: {moscow['must_have']}")
            print(f"   🟡 Should Have: {moscow['should_have']}")
            print(f"   🟢 Could Have: {moscow['could_have']}")
            print(f"   ⚪ Won't Have: {moscow['wont_have']}")
            
            # Top priority features
            print(f"\n⭐ TOP PRIORITY FEATURES:")
            for i, feature in enumerate(insights['top_priority_features'], 1):
                status_icon = "✅" if feature['status'] == 'done' else "🔄" if feature['status'] == 'in_progress' else "⏳"
                print(f"   {i}. {status_icon} {feature['name']} (RICE: {feature['rice_score']:.1f})")
            
        except Exception as e:
            print(f"❌ Error showing features: {e}")
    
    def show_feedback(self) -> None:
        """Show feedback analysis summary"""
        try:
            self.load_project()
            insights = self.dashboard.get_feedback_insights()
            
            if 'message' in insights:
                print("📭 No feedback data available")
                return
            
            print("=" * 60)
            print("💬 FEEDBACK ANALYSIS")
            print("=" * 60)
            
            print(f"📊 Total Feedback: {insights['total_feedback']}")
            print(f"📈 Recent (7 days): {insights['recent_feedback']}")
            print(f"✅ Resolution Rate: {insights['resolution_rate']:.1f}%")
            
            # Sentiment
            sentiment = insights.get('sentiment_analysis', {})
            if sentiment:
                avg_sentiment = sentiment.get('average_sentiment', 0)
                if avg_sentiment > 0.2:
                    sentiment_icon = "😊"
                    sentiment_text = "Positive"
                elif avg_sentiment < -0.2:
                    sentiment_icon = "😟"
                    sentiment_text = "Negative"
                else:
                    sentiment_icon = "😐"
                    sentiment_text = "Neutral"
                
                print(f"😊 Sentiment: {sentiment_icon} {sentiment_text} ({avg_sentiment:.2f})")
            
            # Urgent items
            if insights['urgent_items'] > 0:
                print(f"🚨 Critical Items: {insights['urgent_items']}")
            
            # Common themes
            if insights['common_themes']:
                print(f"\n🔍 TOP THEMES:")
                for i, theme in enumerate(insights['common_themes'], 1):
                    print(f"   {i}. {theme['theme']} ({theme['frequency']} occurrences)")
            
            # Key insights
            if insights['key_insights']:
                print(f"\n💡 KEY INSIGHTS:")
                for insight in insights['key_insights']:
                    print(f"   • {insight}")
            
        except Exception as e:
            print(f"❌ Error showing feedback: {e}")
    
    def show_progress(self) -> None:
        """Show development progress"""
        try:
            self.load_project()
            insights = self.dashboard.get_progress_insights()
            
            print("=" * 60)
            print("📈 DEVELOPMENT PROGRESS")
            print("=" * 60)
            
            print(f"🎯 Overall Progress: {insights['overall_progress']:.1f}%")
            print(f"📅 Current Phase: {insights['current_phase']}")
            print(f"⏰ Days Elapsed: {insights['project_timeline']['days_elapsed']}")
            
            # Task health
            task_health = insights['task_health']
            print(f"\n📋 TASK SUMMARY:")
            print(f"   ✅ Completed: {task_health['completed_tasks']}/{task_health['total_tasks']}")
            if task_health['overdue_tasks'] > 0:
                print(f"   🔴 Overdue: {task_health['overdue_tasks']}")
            
            # Milestone summary
            milestones = insights['milestones']
            print(f"\n🎯 MILESTONES:")
            print(f"   ✅ Completed: {milestones['completed']}/{milestones['total']}")
            if milestones['overdue'] > 0:
                print(f"   🔴 Overdue: {milestones['overdue']}")
            if milestones['upcoming'] > 0:
                print(f"   ⏰ Upcoming: {milestones['upcoming']}")
            
            # Phase breakdown
            print(f"\n📊 PHASE BREAKDOWN:")
            for phase_name, phase_data in insights['phase_breakdown'].items():
                if phase_data['total_tasks'] > 0:
                    progress = phase_data['progress_percentage']
                    if progress >= 100:
                        icon = "✅"
                    elif progress >= 50:
                        icon = "🔄"
                    else:
                        icon = "⏳"
                    
                    phase_display = phase_name.replace('_', ' ').title()
                    print(f"   {icon} {phase_display}: {progress:.1f}% ({phase_data['completed_tasks']}/{phase_data['total_tasks']})")
            
        except Exception as e:
            print(f"❌ Error showing progress: {e}")
    
    def show_actions(self) -> None:
        """Show prioritized action items"""
        try:
            self.load_project()
            action_items = self.dashboard.get_action_items()
            
            print("=" * 60)
            print("📋 ACTION ITEMS")
            print("=" * 60)
            
            if not action_items:
                print("🎉 No urgent action items - project is on track!")
                return
            
            for i, item in enumerate(action_items, 1):
                priority_icons = {
                    'Critical': '🔴',
                    'High': '🟡', 
                    'Medium': '🟢',
                    'Low': '⚪'
                }
                
                icon = priority_icons.get(item['priority'], '⚪')
                due_date = datetime.fromisoformat(item['due_date']).strftime('%Y-%m-%d')
                
                print(f"{i}. {icon} {item['title']} ({item['priority']})")
                print(f"   📝 {item['description']}")
                print(f"   🎯 {item['action']}")
                print(f"   📅 Due: {due_date}")
                print()
            
        except Exception as e:
            print(f"❌ Error showing actions: {e}")
    
    def generate_report(self, output_file: Optional[str] = None) -> None:
        """Generate comprehensive dashboard report"""
        try:
            self.load_project()
            
            if not output_file:
                output_file = f"mvp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            self.dashboard.save_dashboard_report(output_file)
            print(f"📊 Comprehensive report generated: {output_file}")
            
            # Also show summary
            print("\n" + "=" * 60)
            print("📊 REPORT SUMMARY")
            print("=" * 60)
            
            summary = self.dashboard.get_executive_summary()
            print(f"🎯 MVP Completion: {summary['project_status']['mvp_completion_percentage']:.1f}%")
            print(f"📈 Task Progress: {summary['key_metrics']['completed_tasks']}/{summary['key_metrics']['total_tasks']}")
            print(f"💬 Feedback Items: {summary['key_metrics']['total_feedback']}")
            
            health = summary['health_indicators']
            if health['critical']:
                print("🔴 Status: CRITICAL - Immediate attention required")
            elif health['at_risk']:
                print("🟡 Status: AT RISK - Monitor closely")
            else:
                print("🟢 Status: ON TRACK - Good progress")
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="MVP Development CLI Tool - Strategic Blueprint Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mvp init "My MVP Project"              # Initialize new project
  mvp init "SynData MVP" --sample        # Initialize with sample data
  mvp status                             # Show project status
  mvp features                           # Show feature prioritization
  mvp feedback                           # Show feedback analysis
  mvp progress                           # Show development progress
  mvp actions                            # Show action items
  mvp report                             # Generate comprehensive report
  mvp report --output my_report.json     # Generate report with custom filename
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize new MVP project')
    init_parser.add_argument('project_name', help='Name of the MVP project')
    init_parser.add_argument('--sample', action='store_true', 
                           help='Initialize with sample SynData project data')
    
    # Status command
    subparsers.add_parser('status', help='Show current project status')
    
    # Features command
    subparsers.add_parser('features', help='Show feature prioritization summary')
    
    # Feedback command
    subparsers.add_parser('feedback', help='Show feedback analysis summary')
    
    # Progress command
    subparsers.add_parser('progress', help='Show development progress')
    
    # Actions command
    subparsers.add_parser('actions', help='Show prioritized action items')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive report')
    report_parser.add_argument('--output', '-o', help='Output filename for report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = MVPCLIManager()
    
    try:
        if args.command == 'init':
            cli.init_project(args.project_name, args.sample)
        elif args.command == 'status':
            cli.show_status()
        elif args.command == 'features':
            cli.show_features()
        elif args.command == 'feedback':
            cli.show_feedback()
        elif args.command == 'progress':
            cli.show_progress()
        elif args.command == 'actions':
            cli.show_actions()
        elif args.command == 'report':
            cli.generate_report(args.output)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
