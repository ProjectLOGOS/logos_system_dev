"""
JSON Success Criteria Handler for NULL_STATE
Manages external criteria configuration and learning packet rotation
"""

import json
import os
import glob
import time
import copy
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

class CriteriaManager:
    """Manages success criteria stored in JSON files."""
    
    def __init__(self, criteria_dir="./criteria"):
        self.criteria_dir = criteria_dir
        self.current_criteria = None
        self.current_criteria_path = None
        
        # Create criteria directory if it doesn't exist
        os.makedirs(criteria_dir, exist_ok=True)
        
        # Create default criteria if none exist
        if not glob.glob(os.path.join(criteria_dir, "*.json")):
            self._create_default_criteria()
            
        # Load most recent criteria
        self._load_most_recent()
    
    def _create_default_criteria(self):
        """Create default success criteria file."""
        default_criteria = {
            "metadata": {
                "name": "Standard Universe",
                "description": "Default universe parameters with Earth-like conditions",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "display": {
                    "color_schemes": {
                        "constants": ["#e6f7ff", "#1890ff", "#003a8c"],
                        "forces": ["#f6ffed", "#52c41a", "#135200"],
                        "properties": ["#fff1f0", "#ff4d4f", "#820014"],
                        "life_criteria": ["#f9f0ff", "#722ed1", "#120338"]
                    }
                }
            },
            "parameters": {
                "constants": {
                    "fine_structure": {
                        "value": 0.0072973525693,
                        "tolerance": 0.01,
                        "description": "Determines electromagnetic interaction strength"
                    },
                    "gravitational_constant": {
                        "value": 6.67430e-11,
                        "tolerance": 0.01,
                        "description": "Determines gravitational interaction strength"
                    },
                    "speed_of_light": {
                        "value": 299792458.0,
                        "tolerance": 0.01,
                        "description": "Maximum speed limit in universe"
                    },
                    "planck_constant": {
                        "value": 6.62607015e-34,
                        "tolerance": 0.01,
                        "description": "Quantum action scale"
                    },
                    "cosmological_constant": {
                        "value": 1.0e-52,
                        "tolerance": 0.01,
                        "description": "Energy density of vacuum space"
                    }
                },
                "forces": {
                    "strong_nuclear": {
                        "value": 1.0,
                        "tolerance": 0.05,
                        "description": "Relative strength of strong nuclear force"
                    },
                    "weak_nuclear": {
                        "value": 1.0e-13,
                        "tolerance": 0.05,
                        "description": "Relative strength of weak nuclear force"
                    },
                    "electromagnetic": {
                        "value": 1.0e-2,
                        "tolerance": 0.05,
                        "description": "Relative strength of electromagnetic force"
                    },
                    "gravitational": {
                        "value": 1.0e-38,
                        "tolerance": 0.05,
                        "description": "Relative strength of gravitational force"
                    }
                },
                "properties": {
                    "matter_energy_ratio": {
                        "value": 0.315,
                        "tolerance": 0.1,
                        "description": "Ratio of matter to total energy density"
                    },
                    "entropy_gradient": {
                        "value": 1.0e-5, 
                        "tolerance": 0.1,
                        "description": "Rate of entropy increase"
                    },
                    "expansion_rate": {
                        "value": 67.4,
                        "tolerance": 0.1,
                        "description": "Universe expansion rate (km/s/Mpc)"
                    },
                    "dimensionality": {
                        "value": 3,
                        "tolerance": 0,
                        "description": "Number of spatial dimensions"
                    }
                },
                "life_criteria": {
                    "carbon_oxygen_formation": {
                        "value": 0.7,
                        "min_threshold": 0.5,
                        "description": "Ability to form carbon and oxygen atoms"
                    },
                    "stellar_stability": {
                        "value": 0.85,
                        "min_threshold": 0.7,
                        "description": "Stability of star formation and lifecycle"
                    },
                    "chemical_complexity": {
                        "value": 0.75, 
                        "min_threshold": 0.6,
                        "description": "Potential for complex chemical interactions"
                    }
                },
                "logical": {
                    "non_contradiction": {
                        "value": True,
                        "description": "Logical requirement of non-contradiction"
                    },
                    "identity": {
                        "value": True,
                        "description": "Logical requirement of identity"
                    },
                    "excluded_middle": {
                        "value": True,
                        "description": "Logical requirement of excluded middle"
                    },
                    "causality": {
                        "value": True,
                        "description": "Logical requirement of causality"
                    }
                }
            }
        }
        
        filepath = os.path.join(self.criteria_dir, "standard_universe.json")
        with open(filepath, 'w') as f:
            json.dump(default_criteria, f, indent=2)
    
    def _load_most_recent(self):
        """Load the most recently modified criteria file."""
        criteria_files = glob.glob(os.path.join(self.criteria_dir, "*.json"))
        if not criteria_files:
            return False
            
        # Sort by modification time, newest first
        newest_file = max(criteria_files, key=os.path.getmtime)
        return self.load_criteria(newest_file)
    
    def load_criteria(self, filepath):
        """Load criteria from specified JSON file."""
        try:
            with open(filepath, 'r') as f:
                self.current_criteria = json.load(f)
                self.current_criteria_path = filepath
                return True
        except Exception as e:
            print(f"Error loading criteria: {e}")
            return False
    
    def save_criteria(self, criteria, filepath=None):
        """Save criteria to JSON file."""
        if filepath is None:
            timestamp = int(time.time())
            name = criteria.get("metadata", {}).get("name", "universe").lower().replace(" ", "_")
            filepath = os.path.join(self.criteria_dir, f"{name}_{timestamp}.json")
            
        with open(filepath, 'w') as f:
            json.dump(criteria, f, indent=2)
            
        self.current_criteria = criteria
        self.current_criteria_path = filepath
        return filepath
    
    def update_criteria(self, updates, save=True):
        """Update existing criteria with new values."""
        if not self.current_criteria:
            return False
            
        # Deep copy to avoid modifying original
        updated_criteria = copy.deepcopy(self.current_criteria)
        
        # Apply updates
        for category, params in updates.get("parameters", {}).items():
            if category in updated_criteria["parameters"]:
                for param, data in params.items():
                    if param in updated_criteria["parameters"][category]:
                        updated_criteria["parameters"][category][param].update(data)
        
        # Update metadata
        if "metadata" in updates:
            updated_criteria["metadata"].update(updates["metadata"])
            
        # Update version and timestamp
        updated_criteria["metadata"]["version"] = str(float(updated_criteria["metadata"]["version"]) + 0.1)
        updated_criteria["metadata"]["updated"] = datetime.now().isoformat()
        
        if save:
            self.save_criteria(updated_criteria)
            
        return updated_criteria
    
    def convert_to_universe_parameters(self):
        """Convert JSON criteria to UniverseParameters object."""
        if not self.current_criteria:
            return None
            
        from dataclasses import asdict
        
        # Import here to avoid circular imports
        from null_state import (
            PhysicalConstants, ForceRelationships, UniverseProperties,
            LifePermittingCriteria, LogicalRequirements, UniverseParameters
        )
        
        # Extract parameters
        params = self.current_criteria["parameters"]
        
        # Create individual parameter objects
        constants = PhysicalConstants(
            fine_structure=params["constants"]["fine_structure"]["value"],
            gravitational_constant=params["constants"]["gravitational_constant"]["value"],
            speed_of_light=params["constants"]["speed_of_light"]["value"],
            planck_constant=params["constants"]["planck_constant"]["value"],
            cosmological_constant=params["constants"]["cosmological_constant"]["value"]
        )
        
        forces = ForceRelationships(
            strong_nuclear=params["forces"]["strong_nuclear"]["value"],
            weak_nuclear=params["forces"]["weak_nuclear"]["value"],
            electromagnetic=params["forces"]["electromagnetic"]["value"],
            gravitational=params["forces"]["gravitational"]["value"]
        )
        
        properties = UniverseProperties(
            matter_energy_ratio=params["properties"]["matter_energy_ratio"]["value"],
            entropy_gradient=params["properties"]["entropy_gradient"]["value"],
            expansion_rate=params["properties"]["expansion_rate"]["value"],
            dimensionality=params["properties"]["dimensionality"]["value"]
        )
        
        life_criteria = LifePermittingCriteria(
            carbon_oxygen_formation=params["life_criteria"]["carbon_oxygen_formation"]["value"],
            stellar_stability=params["life_criteria"]["stellar_stability"]["value"],
            chemical_complexity=params["life_criteria"]["chemical_complexity"]["value"]
        )
        
        logical = LogicalRequirements(
            non_contradiction=params["logical"]["non_contradiction"]["value"],
            identity=params["logical"]["identity"]["value"],
            excluded_middle=params["logical"]["excluded_middle"]["value"],
            causality=params["logical"]["causality"]["value"]
        )
        
        # Create and return complete UniverseParameters
        return UniverseParameters(
            constants=constants,
            forces=forces,
            properties=properties,
            life_criteria=life_criteria,
            logical=logical
        )
    
    def evaluate_universe(self, current_params):
        """Evaluate universe against current criteria."""
        if not self.current_criteria:
            return {}
            
        from dataclasses import asdict
        current_values = asdict(current_params)
        criteria = self.current_criteria["parameters"]
        results = {}
        
        # Evaluate constants
        constants_match = True
        constant_results = {}
        for name, value in asdict(current_params.constants).items():
            if name in criteria["constants"]:
                target = criteria["constants"][name]["value"]
                tolerance = criteria["constants"][name]["tolerance"]
                diff_ratio = abs(value - target) / target
                success = diff_ratio <= tolerance
                constants_match = constants_match and success
                constant_results[name] = {
                    "target": target,
                    "current": value,
                    "difference": diff_ratio,
                    "tolerance": tolerance,
                    "success": success
                }
        results["constants"] = {
            "success": constants_match,
            "details": constant_results
        }
        
        # Evaluate forces
        forces_match = True
        force_results = {}
        for name, value in asdict(current_params.forces).items():
            if name in criteria["forces"]:
                target = criteria["forces"][name]["value"]
                tolerance = criteria["forces"][name]["tolerance"]
                diff_ratio = abs(value - target) / target
                success = diff_ratio <= tolerance
                forces_match = forces_match and success
                force_results[name] = {
                    "target": target,
                    "current": value,
                    "difference": diff_ratio,
                    "tolerance": tolerance,
                    "success": success
                }
        results["forces"] = {
            "success": forces_match,
            "details": force_results
        }
        
        # Evaluate properties
        properties_match = True
        property_results = {}
        for name, value in asdict(current_params.properties).items():
            if name in criteria["properties"]:
                target = criteria["properties"][name]["value"]
                tolerance = criteria["properties"][name]["tolerance"]
                # Special case for dimensionality which must be exact
                if name == "dimensionality":
                    success = value == target
                    diff_ratio = 0 if success else 1
                else:
                    diff_ratio = abs(value - target) / target
                    success = diff_ratio <= tolerance
                properties_match = properties_match and success
                property_results[name] = {
                    "target": target,
                    "current": value,
                    "difference": diff_ratio,
                    "tolerance": tolerance,
                    "success": success
                }
        results["properties"] = {
            "success": properties_match,
            "details": property_results
        }
        
        # Evaluate life criteria
        life_possible = True
        life_results = {}
        for name, value in asdict(current_params.life_criteria).items():
            if name in criteria["life_criteria"]:
                threshold = criteria["life_criteria"][name]["min_threshold"]
                success = value >= threshold
                life_possible = life_possible and success
                life_results[name] = {
                    "threshold": threshold,
                    "current": value,
                    "success": success
                }
        results["life_criteria"] = {
            "success": life_possible,
            "details": life_results
        }
        
        # Evaluate logical requirements
        logic_valid = True
        logic_results = {}
        for name, value in asdict(current_params.logical).items():
            if name in criteria["logical"]:
                target = criteria["logical"][name]["value"]
                success = value == target
                logic_valid = logic_valid and success
                logic_results[name] = {
                    "target": target,
                    "current": value,
                    "success": success
                }
        results["logical"] = {
            "success": logic_valid,
            "details": logic_results
        }
        
        # Overall success
        results["overall_success"] = (
            constants_match and forces_match and properties_match and 
            life_possible and logic_valid
        )
        
        return results
    
    def get_gui_friendly_data(self):
        """Get GUI-friendly representation of criteria."""
        if not self.current_criteria:
            return {}
            
        return {
            "metadata": self.current_criteria["metadata"],
            "parameters": self.current_criteria["parameters"],
            "file_path": self.current_criteria_path,
            "available_criteria": self.list_available_criteria()
        }
    
    def list_available_criteria(self):
        """List all available criteria files."""
        criteria_files = glob.glob(os.path.join(self.criteria_dir, "*.json"))
        result = []
        
        for filepath in criteria_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    result.append({
                        "name": data.get("metadata", {}).get("name", "Unknown"),
                        "description": data.get("metadata", {}).get("description", ""),
                        "version": data.get("metadata", {}).get("version", ""),
                        "created": data.get("metadata", {}).get("created", ""),
                        "updated": data.get("metadata", {}).get("updated", ""),
                        "file_path": filepath
                    })
            except:
                pass
                
        return sorted(result, key=lambda x: os.path.getmtime(x["file_path"]), reverse=True)


class LearningPacketManager:
    """Manages learning packet files with rotation."""
    
    def __init__(self, packets_dir="./learning_packets", rotation_size=5):
        self.packets_dir = packets_dir
        self.rotation_size = rotation_size
        self.current_index = 0
        
        # Create packets directory if it doesn't exist
        os.makedirs(packets_dir, exist_ok=True)
        
        # Initialize rotation index
        self._initialize_rotation()
    
    def _initialize_rotation(self):
        """Initialize rotation index."""
        packet_files = glob.glob(os.path.join(self.packets_dir, "packet_*.json"))
        
        if not packet_files:
            self.current_index = 0
            return
            
        # Extract indices from filenames
        indices = []
        for filepath in packet_files:
            try:
                filename = os.path.basename(filepath)
                index = int(filename.split("_")[1].split(".")[0])
                indices.append(index)
            except:
                pass
                
        if indices:
            self.current_index = (max(indices) + 1) % self.rotation_size
        else:
            self.current_index = 0
    
    def get_next_packet_path(self):
        """Get path for next packet in rotation."""
        filepath = os.path.join(self.packets_dir, f"packet_{self.current_index}.json")
        self.current_index = (self.current_index + 1) % self.rotation_size
        return filepath
    
    def save_learning_packet(self, packet_data):
        """Save learning packet with rotation."""
        filepath = self.get_next_packet_path()
        
        # Add timestamp and rotation index
        packet_data["metadata"] = packet_data.get("metadata", {})
        packet_data["metadata"]["timestamp"] = time.time()
        packet_data["metadata"]["rotation_index"] = self.current_index
        
        with open(filepath, 'w') as f:
            json.dump(packet_data, f, indent=2)
            
        return filepath
    
    def load_learning_packet(self, index=None):
        """Load learning packet by index."""
        if index is None:
            # Load latest packet
            packet_files = glob.glob(os.path.join(self.packets_dir, "packet_*.json"))
            if not packet_files:
                return None
                
            filepath = max(packet_files, key=os.path.getmtime)
        else:
            # Load specific index
            filepath = os.path.join(self.packets_dir, f"packet_{index}.json")
            if not os.path.exists(filepath):
                return None
                
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def get_all_packets(self):
        """Get all learning packets."""
        packet_files = glob.glob(os.path.join(self.packets_dir, "packet_*.json"))
        result = []
        
        for filepath in packet_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    result.append({
                        "file_path": filepath,
                        "metadata": data.get("metadata", {}),
                        "data": data
                    })
            except:
                pass
                
        return sorted(result, key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)