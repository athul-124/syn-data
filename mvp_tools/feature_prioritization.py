"""Feature Prioritization Module"""

import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

class Priority(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

@dataclass
class Feature:
    id: str
    name: str
    description: str
    priority: Priority
    status: str = "planned"

class FeaturePrioritizer:
    def __init__(self):
        self.features: List[Feature] = []
    
    def load_from_json(self, filename: str):
        """Load features from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            # Handle loading logic here
        except FileNotFoundError:
            pass  # File doesn't exist yet
    
    def save_to_json(self, filename: str):
        """Save features to JSON file"""
        data = {"features": []}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_prioritization_report(self) -> Dict:
        """Generate feature prioritization report"""
        total_features = len(self.features)
        mvp_features = [f for f in self.features if f.priority == Priority.HIGH]
        
        return {
            'total_features': total_features,
            'mvp_features_count': len(mvp_features),
            'estimated_mvp_effort': len(mvp_features) * 8,  # Rough estimate
            'moscow_breakdown': {
                'must_have': len([f for f in self.features if f.priority == Priority.HIGH]),
                'should_have': len([f for f in self.features if f.priority == Priority.MEDIUM]),
                'could_have': len([f for f in self.features if f.priority == Priority.LOW]),
                'wont_have': 0
            }
        }

    def get_features_by_rice_score(self) -> List[Feature]:
        """Get features sorted by priority (placeholder for RICE)"""
        return sorted(self.features, key=lambda f: f.priority.value, reverse=True)

    def get_moscow_features(self, priority: Priority) -> List[Feature]:
        """Get features by MoSCoW priority"""
        return [f for f in self.features if f.priority == priority]

def create_syndata_mvp_features():
    return FeaturePrioritizer()
