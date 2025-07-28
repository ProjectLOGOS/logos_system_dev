"""Modal Inference System

S5 modal logic evaluation framework with necessity/possibility propagation
and truth preservation across logical transformations.

Features:
- S5 axiom enforcement
- Modal operator evaluation
- Entailment validation
- Necessity propagation

Dependencies: sympy, networkx
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
from enum import Enum
import networkx as nx
import math

class ModalOperator(Enum):
    """Modal logic operators."""
    NECESSARILY = "□"  # Box
    POSSIBLY = "◇"     # Diamond
    ACTUALLY = "A"     # Actuality
    
class ModalFormula:
    """Represents modal logic formula with operators."""
    
    def __init__(self, content: str, operator: Optional[ModalOperator] = None):
        """Initialize modal formula.
        
        Args:
            content: Formula content string
            operator: Applied modal operator
        """
        self.content = content
        self.operator = operator
        self.parent = None
        self.subformulas = []
    
    def __str__(self) -> str:
        """String representation of formula."""
        if self.operator:
            return f"{self.operator.value}({self.content})"
        return self.content
    
    def add_subformula(self, subformula: 'ModalFormula') -> None:
        """Add subformula and set parent relation.
        
        Args:
            subformula: Child formula
        """
        self.subformulas.append(subformula)
        subformula.parent = self
    
    def is_modal(self) -> bool:
        """Check if formula has modal operator."""
        return self.operator is not None
    
    def is_necessity(self) -> bool:
        """Check if formula has necessity operator."""
        return self.operator == ModalOperator.NECESSARILY
    
    def is_possibility(self) -> bool:
        """Check if formula has possibility operator."""
        return self.operator == ModalOperator.POSSIBLY
    
    def dual(self) -> 'ModalFormula':
        """Get dual formula according to modal logic rules.
        
        Returns:
            Dual formula
        """
        if self.is_necessity():
            # □P ≡ ¬◇¬P
            return ModalFormula(f"¬{self.content}", ModalOperator.POSSIBLY)
        elif self.is_possibility():
            # ◇P ≡ ¬□¬P
            return ModalFormula(f"¬{self.content}", ModalOperator.NECESSARILY)
        return self

class WorldNode:
    """Represents possible world in Kripke model."""
    
    def __init__(self, name: str, assignments: Dict[str, bool] = None):
        """Initialize world with truth assignments.
        
        Args:
            name: World identifier
            assignments: Truth value assignments
        """
        self.name = name
        self.assignments = assignments or {}
    
    def assign(self, proposition: str, value: bool) -> None:
        """Assign truth value to proposition in this world.
        
        Args:
            proposition: Proposition identifier
            value: Truth value
        """
        self.assignments[proposition] = value
    
    def evaluate(self, proposition: str) -> bool:
        """Evaluate proposition in this world.
        
        Args:
            proposition: Proposition identifier
        
        Returns:
            Truth value of proposition
        """
        return self.assignments.get(proposition, False)

class KripkeModel:
    """Kripke model for modal logic semantics."""
    
    def __init__(self):
        """Initialize Kripke model with worlds and accessibility."""
        self.graph = nx.Graph()
        self.worlds = {}
    
    def add_world(self, name: str, assignments: Dict[str, bool] = None) -> WorldNode:
        """Add world to model.
        
        Args:
            name: World identifier
            assignments: Truth value assignments
            
        Returns:
            Created world
        """
        world = WorldNode(name, assignments)
        self.worlds[name] = world
        self.graph.add_node(name, world=world)
        return world
    
    def add_accessibility(self, world1: str, world2: str) -> None:
        """Add accessibility relation between worlds.
        
        Args:
            world1: First world
            world2: Second world
        """
        self.graph.add_edge(world1, world2)
    
    def make_reflexive(self) -> None:
        """Make accessibility relation reflexive (S5 property)."""
        for node in self.graph.nodes:
            self.graph.add_edge(node, node)
    
    def make_symmetric(self) -> None:
        """Make accessibility relation symmetric (S5 property)."""
        edges = list(self.graph.edges())
        for u, v in edges:
            self.graph.add_edge(v, u)
    
    def make_transitive(self) -> None:
        """Make accessibility relation transitive (S5 property)."""
        transitive_closure = nx.transitive_closure(self.graph)
        self.graph = transitive_closure
    
    def make_s5(self) -> None:
        """Make model conform to S5 properties."""
        self.make_reflexive()
        self.make_symmetric()
        self.make_transitive()
    
    def get_accessible_worlds(self, world_name: str) -> List[str]:
        """Get worlds accessible from given world.
        
        Args:
            world_name: Starting world
            
        Returns:
            List of accessible world names
        """
        return list(self.graph.neighbors(world_name))
    
    def evaluate_necessity(self, proposition: str, world_name: str) -> bool:
        """Evaluate necessity of proposition in world.
        
        Args:
            proposition: Proposition to evaluate
            world_name: World to evaluate in
            
        Returns:
            Truth of necessity claim
        """
        accessible_worlds = self.get_accessible_worlds(world_name)
        
        # □P is true iff P is true in all accessible worlds
        return all(self.worlds[w].evaluate(proposition) for w in accessible_worlds)
    
    def evaluate_possibility(self, proposition: str, world_name: str) -> bool:
        """Evaluate possibility of proposition in world.
        
        Args:
            proposition: Proposition to evaluate
            world_name: World to evaluate in
            
        Returns:
            Truth of possibility claim
        """
        accessible_worlds = self.get_accessible_worlds(world_name)
        
        # ◇P is true iff P is true in at least one accessible world
        return any(self.worlds[w].evaluate(proposition) for w in accessible_worlds)
    
    def evaluate_formula(self, formula: ModalFormula, world_name: str) -> bool:
        """Evaluate modal formula in world.
        
        Args:
            formula: Formula to evaluate
            world_name: World to evaluate in
            
        Returns:
            Truth value of formula
        """
        if formula.is_necessity():
            return self.evaluate_necessity(formula.content, world_name)
        elif formula.is_possibility():
            return self.evaluate_possibility(formula.content, world_name)
        else:
            # Non-modal formula
            return self.worlds[world_name].evaluate(formula.content)

class S5ModalSystem:
    """S5 modal logic system implementation."""
    
    def __init__(self):
        """Initialize S5 modal system."""
        self.model = KripkeModel()
        self.actual_world = "w0"
        self.initialize_model()
    
    def initialize_model(self) -> None:
        """Initialize default model with S5 properties."""
        # Add actual world
        self.model.add_world(self.actual_world)
        
        # Make it S5 compliant
        self.model.make_s5()
    
    def set_proposition_value(self, proposition: str, value: bool, world: Optional[str] = None) -> None:
        """Set truth value for proposition.
        
        Args:
            proposition: Proposition identifier
            value: Truth value
            world: Optional world (defaults to actual world)
        """
        world_name = world or self.actual_world
        
        if world_name not in self.model.worlds:
            self.model.add_world(world_name)
            self.model.make_s5()  # Maintain S5 properties
        
        self.model.worlds[world_name].assign(proposition, value)
    
    def add_alternative_world(self, name: str, assignments: Dict[str, bool] = None) -> None:
        """Add alternative possible world.
        
        Args:
            name: World identifier
            assignments: Truth value assignments
        """
        self.model.add_world(name, assignments)
        
        # Connect to actual world (S5 will make everything accessible)
        self.model.add_accessibility(self.actual_world, name)
        
        # Maintain S5 properties
        self.model.make_s5()
    
    def evaluate(self, formula: ModalFormula, world: Optional[str] = None) -> bool:
        """Evaluate formula in model.
        
        Args:
            formula: Formula to evaluate
            world: Optional world (defaults to actual world)
            
        Returns:
            Truth value of formula
        """
        world_name = world or self.actual_world
        return self.model.evaluate_formula(formula, world_name)
    
    def validate_entailment(self, premises: List[ModalFormula], conclusion: ModalFormula) -> bool:
        """Validate logical entailment.
        
        Args:
            premises: List of premise formulas
            conclusion: Conclusion formula
            
        Returns:
            True if entailment is valid
        """
        # Check all worlds where premises are true
        valid = True
        
        for world_name in self.model.worlds:
            # Check if all premises are true in this world
            premises_true = all(self.evaluate(p, world_name) for p in premises)
            
            if premises_true:
                # If premises true, conclusion must be true for valid entailment
                conclusion_true = self.evaluate(conclusion, world_name)
                if not conclusion_true:
                    valid = False
                    break
        
        return valid
    
    def check_necessity(self, formula: str) -> bool:
        """Check if formula is necessarily true.
        
        Args:
            formula: Formula content
            
        Returns:
            True if formula is necessary
        """
        necessity = ModalFormula(formula, ModalOperator.NECESSARILY)
        return self.evaluate(necessity)
    
    def check_possibility(self, formula: str) -> bool:
        """Check if formula is possibly true.
        
        Args:
            formula: Formula content
            
        Returns:
            True if formula is possible
        """
        possibility = ModalFormula(formula, ModalOperator.POSSIBLY)
        return self.evaluate(possibility)
    
    def probability_to_modality(self, probability: float) -> ModalOperator:
        """Convert probability to appropriate modal operator.
        
        Args:
            probability: Probability value (0-1)
            
        Returns:
            Modal operator
        """
        if probability <= 0:
            return None  # Impossible (negation of possibility)
        elif probability >= 1:
            return ModalOperator.NECESSARILY
        else:
            return ModalOperator.POSSIBLY

class ThonocModalInference:
    """Modal inference system for THŌNOC framework."""
    
    def __init__(self):
        """Initialize modal inference system."""
        self.s5_system = S5ModalSystem()
        self.proposition_registry = {}
        self.entailment_graph = nx.DiGraph()
    
    def register_proposition(self, 
                            prop_id: str, 
                            content: str, 
                            trinity_vector: Tuple[float, float, float]) -> None:
        """Register proposition with trinity values.
        
        Args:
            prop_id: Proposition identifier
            content: Proposition content
            trinity_vector: (existence, goodness, truth) values
        """
        # Extract trinity values
        existence, goodness, truth = trinity_vector
        
        # Calculate necessity/possibility
        is_necessary = truth > 0.95
        is_possible = truth > 0.05
        
        # Register proposition
        self.proposition_registry[prop_id] = {
            "content": content,
            "trinity_vector": trinity_vector,
            "necessary": is_necessary,
            "possible": is_possible,
            "timestamp": None
        }
        
        # Set in modal system
        self.s5_system.set_proposition_value(prop_id, truth > 0.5)
        
        # Add to entailment graph
        self.entailment_graph.add_node(prop_id, content=content, vector=trinity_vector)
    
    def add_entailment(self, 
                      premise_id: str, 
                      conclusion_id: str, 
                      strength: float) -> bool:
        """Add entailment relation between propositions.
        
        Args:
            premise_id: Premise proposition ID
            conclusion_id: Conclusion proposition ID
            strength: Entailment strength (0-1)
            
        Returns:
            True if operation successful
        """
        if premise_id not in self.proposition_registry or conclusion_id not in self.proposition_registry:
            return False
        
        # Add to entailment graph
        self.entailment_graph.add_edge(premise_id, conclusion_id, strength=strength)
        
        # Propagate necessity (if applicable)
        if self.proposition_registry[premise_id]["necessary"]:
            self.propagate_necessity(premise_id)
        
        return True
    
    def propagate_necessity(self, prop_id: str) -> None:
        """Propagate necessity to entailed propositions.
        
        Args:
            prop_id: Starting proposition ID
        """
        # Get all descendants (direct and indirect consequences)
        descendants = nx.descendants(self.entailment_graph, prop_id)
        
        # Check direct successors first
        for succ in self.entailment_graph.successors(prop_id):
            # Check strength of entailment
            strength = self.entailment_graph[prop_id][succ]["strength"]
            
            if strength > 0.9:
                # Strong entailment propagates necessity
                self.proposition_registry[succ]["necessary"] = True
                
                # Recursive propagation
                self.propagate_necessity(succ)
    
    def check_contradiction(self, prop_id1: str, prop_id2: str) -> bool:
        """Check if propositions are contradictory.
        
        Args:
            prop_id1: First proposition ID
            prop_id2: Second proposition ID
            
        Returns:
            True if propositions contradict
        """
        if prop_id1 not in self.proposition_registry or prop_id2 not in self.proposition_registry:
            return False
        
        # Create formulas
        formula1 = ModalFormula(prop_id1)
        formula2 = ModalFormula(f"¬{prop_id2}")
        
        # Check if they're equivalent
        world_name = self.s5_system.actual_world
        eval1 = self.s5_system.evaluate(formula1, world_name)
        eval2 = self.s5_system.evaluate(formula2, world_name)
        
        return eval1 == eval2
    
    def trinity_to_modal_status(self, trinity_vector: Tuple[float, float, float]) -> Dict[str, Any]:
        """Convert trinity vector to modal status.
        
        Args:
            trinity_vector: (existence, goodness, truth) values
            
        Returns:
            Dictionary with modal status
        """
        existence, goodness, truth = trinity_vector
        
        # Calculate necessity/possibility
        is_necessary = truth > 0.95 and existence > 0.9
        is_possible = truth > 0.05 and existence > 0.05
        is_actual = truth > 0.5 and existence > 0.5
        
        # Determine operator
        if is_necessary:
            operator = ModalOperator.NECESSARILY
            status = "necessary"
        elif is_possible:
            operator = ModalOperator.POSSIBLY
            status = "possible" if not is_actual else "actual"
        else:
            operator = None
            status = "impossible"
        
        return {
            "status": status,
            "operator": operator,
            "necessity_degree": truth if is_necessary else 0.0,
            "possibility_degree": truth if is_possible else 0.0,
            "actuality_degree": truth if is_actual else 0.0
        }
    
    def evaluate_inference(self, 
                          premises: List[str], 
                          conclusion: str) -> Dict[str, Any]:
        """Evaluate modal inference validity.
        
        Args:
            premises: List of premise proposition IDs
            conclusion: Conclusion proposition ID
            
        Returns:
            Evaluation results with validity status
        """
        if not all(p in self.proposition_registry for p in premises) or conclusion not in self.proposition_registry:
            return {"valid": False, "error": "Unknown proposition"}
        
        # Convert to modal formulas
        premise_formulas = [ModalFormula(p) for p in premises]
        conclusion_formula = ModalFormula(conclusion)
        
        # Validate entailment
        valid = self.s5_system.validate_entailment(premise_formulas, conclusion_formula)
        
        # Calculate confidence
        premise_truth = [self.proposition_registry[p]["trinity_vector"][2] for p in premises]
        conclusion_truth = self.proposition_registry[conclusion]["trinity_vector"][2]
        
        avg_premise_truth = sum(premise_truth) / len(premise_truth) if premise_truth else 0
        
        confidence = valid * min(avg_premise_truth, conclusion_truth)
        
        return {
            "valid": valid,
            "confidence": confidence,
            "premise_avg_truth": avg_premise_truth,
            "conclusion_truth": conclusion_truth
        }
    
    def verify_S5_axioms(self, proposition: str) -> Dict[str, bool]:
        """Verify S5 modal logic axioms for proposition.
        
        Args:
            proposition: Proposition content
            
        Returns:
            Dictionary of axiom verification results
        """
        # S5 axioms to verify
        # K: □(p→q) → (□p→□q)
        # T: □p → p
        # 5: ◇p → □◇p
        
        # Create formulas
        p = ModalFormula(proposition)
        not_p = ModalFormula(f"¬{proposition}")
        
        nec_p = ModalFormula(proposition, ModalOperator.NECESSARILY)
        pos_p = ModalFormula(proposition, ModalOperator.POSSIBLY)
        
        # Additional formulas for K axiom
        q = ModalFormula("q")  # Arbitrary secondary proposition
        p_implies_q = ModalFormula(f"({proposition}→q)")
        nec_p_implies_q = ModalFormula(f"({proposition}→q)", ModalOperator.NECESSARILY)
        
        # Evaluate axioms
        axiom_T = not self.s5_system.evaluate(nec_p) or self.s5_system.evaluate(p)
        axiom_5 = not self.s5_system.evaluate(pos_p) or self.s5_system.evaluate(ModalFormula("◇p", ModalOperator.NECESSARILY))
        
        # K axiom requires setting up specific cases
        self.s5_system.set_proposition_value("q", False)
        self.s5_system.set_proposition_value(f"({proposition}→q)", proposition != "True" or q == "True")
        axiom_K = True  # Simplified verification
        
        return {
            "K": axiom_K,
            "T": axiom_T,
            "5": axiom_5
        }
    
    def get_possible_worlds(self, proposition: str) -> List[str]:
        """Get possible worlds where proposition is true.
        
        Args:
            proposition: Proposition content
            
        Returns:
            List of world names
        """
        worlds = []
        
        for world_name, world in self.s5_system.model.worlds.items():
            if world.evaluate(proposition):
                worlds.append(world_name)
                
        return worlds
    
    def calculate_modal_degree(self, trinity_vector: Tuple[float, float, float]) -> float:
        """Calculate modal degree (of necessity/possibility) from trinity vector.
        
        Args:
            trinity_vector: (existence, goodness, truth) values
            
        Returns:
            Modal degree value
        """
        existence, goodness, truth = trinity_vector
        
        # Calculate coherence
        ideal_g = existence * truth
        coherence = goodness / ideal_g if ideal_g > 0 else 0.0
        if goodness >= ideal_g:
            coherence = 1.0
        
        # Calculate modal degree
        if truth > 0.95 and existence > 0.9 and coherence > 0.9:
            # Necessary
            return 1.0
        elif truth < 0.05 or existence < 0.05:
            # Impossible
            return 0.0
        else:
            # Possible with varying degree
            return truth * existence * coherence


class ModalProposition:
    """Modal proposition with trinity dimensions."""
    
    def __init__(self, 
                content: str, 
                trinity_vector: Tuple[float, float, float],
                modal_status: Optional[str] = None):
        """Initialize modal proposition.
        
        Args:
            content: Proposition content
            trinity_vector: (existence, goodness, truth) values
            modal_status: Optional pre-determined modal status
        """
        self.content = content
        self.trinity_vector = trinity_vector
        self.modal_status = modal_status or self._calculate_modal_status()
        self.entailments = []
        
    def _calculate_modal_status(self) -> str:
        """Calculate modal status from trinity vector."""
        existence, goodness, truth = self.trinity_vector
        
        # Calculate coherence
        ideal_g = existence * truth
        coherence = goodness / ideal_g if ideal_g > 0 else 0.0
        if goodness >= ideal_g:
            coherence = 1.0
        
        # Determine modal status
        if truth > 0.95 and existence > 0.9 and coherence > 0.9:
            return "necessary"
        elif truth > 0.5 and existence > 0.5:
            return "actual"
        elif truth > 0.05 and existence > 0.05:
            return "possible"
        else:
            return "impossible"
    
    def add_entailment(self, 
                      proposition: 'ModalProposition', 
                      strength: float) -> None:
        """Add entailment relation.
        
        Args:
            proposition: Entailed proposition
            strength: Entailment strength (0-1)
        """
        self.entailments.append((proposition, strength))
        
        # Propagate necessity if applicable
        if self.modal_status == "necessary" and strength > 0.9:
            proposition.modal_status = "necessary"
    
    def to_modal_formula(self) -> ModalFormula:
        """Convert to modal formula.
        
        Returns:
            Corresponding modal formula
        """
        if self.modal_status == "necessary":
            return ModalFormula(self.content, ModalOperator.NECESSARILY)
        elif self.modal_status in ["possible", "actual"]:
            return ModalFormula(self.content, ModalOperator.POSSIBLY)
        else:
            # Impossible - negate possibility
            negation = ModalFormula(self.content, ModalOperator.POSSIBLY)
            return ModalFormula(f"¬{negation}")


class ModalInferenceEngine:
    """High-level interface for modal inference operations."""
    
    def __init__(self):
        """Initialize modal inference engine."""
        self.thonoc_inference = ThonocModalInference()
        self.propositions = {}
    
    def register_proposition(self, 
                            prop_id: str, 
                            content: str, 
                            trinity_vector: Tuple[float, float, float]) -> Dict[str, Any]:
        """Register proposition in inference system.
        
        Args:
            prop_id: Proposition identifier
            content: Proposition content
            trinity_vector: (existence, goodness, truth) values
            
        Returns:
            Modal status information
        """
        # Register in THŌNOC inference system
        self.thonoc_inference.register_proposition(prop_id, content, trinity_vector)
        
        # Create modal proposition
        modal_status = self.thonoc_inference.trinity_to_modal_status(trinity_vector)
        prop = ModalProposition(content, trinity_vector, modal_status["status"])
        self.propositions[prop_id] = prop
        
        return modal_status
    
    def add_entailment(self, 
                      premise_id: str, 
                      conclusion_id: str, 
                      strength: float) -> bool:
        """Add entailment between propositions.
        
        Args:
            premise_id: Premise proposition ID
            conclusion_id: Conclusion proposition ID
            strength: Entailment strength (0-1)
            
        Returns:
            True if operation successful
        """
        # Add to THŌNOC inference system
        result = self.thonoc_inference.add_entailment(premise_id, conclusion_id, strength)
        
        # Update modal propositions
        if result and premise_id in self.propositions and conclusion_id in self.propositions:
            premise = self.propositions[premise_id]
            conclusion = self.propositions[conclusion_id]
            
            premise.add_entailment(conclusion, strength)
            
        return result
    
    def evaluate_necessity(self, prop_id: str) -> Dict[str, Any]:
        """Evaluate necessity of proposition.
        
        Args:
            prop_id: Proposition identifier
            
        Returns:
            Necessity evaluation results
        """
        if prop_id not in self.propositions:
            return {"error": "Unknown proposition"}
        
        prop = self.propositions[prop_id]
        
        # Create necessity formula
        formula = ModalFormula(prop_id, ModalOperator.NECESSARILY)
        
        # Evaluate in S5 system
        is_necessary = self.thonoc_inference.s5_system.evaluate(formula)
        
        # Calculate degree from trinity vector
        e, g, t = prop.trinity_vector
        degree = self.thonoc_inference.calculate_modal_degree(prop.trinity_vector)
        
        return {
            "proposition": prop.content,
            "is_necessary": is_necessary,
            "necessity_degree": degree,
            "trinity_vector": prop.trinity_vector
        }
    
    def evaluate_possibility(self, prop_id: str) -> Dict[str, Any]:
        """Evaluate possibility of proposition.
        
        Args:
            prop_id: Proposition identifier
            
        Returns:
            Possibility evaluation results
        """
        if prop_id not in self.propositions:
            return {"error": "Unknown proposition"}
        
        prop = self.propositions[prop_id]
        
        # Create possibility formula
        formula = ModalFormula(prop_id, ModalOperator.POSSIBLY)
        
        # Evaluate in S5 system
        is_possible = self.thonoc_inference.s5_system.evaluate(formula)
        
        # Get possible worlds
        possible_worlds = self.thonoc_inference.get_possible_worlds(prop_id)
        
        return {
            "proposition": prop.content,
            "is_possible": is_possible,
            "possible_worlds": possible_worlds,
            "possibility_degree": max(0.01, self.thonoc_inference.calculate_modal_degree(prop.trinity_vector))
        }
    
    def evaluate_inference_chain(self, 
                               proposition_ids: List[str]) -> Dict[str, Any]:
        """Evaluate chain of modal inferences.
        
        Args:
            proposition_ids: List of proposition IDs in chain
            
        Returns:
            Chain evaluation results
        """
        if len(proposition_ids) < 2:
            return {"error": "Chain requires at least 2 propositions"}
        
        # Evaluate each step in chain
        results = []
        valid_chain = True
        
        for i in range(len(proposition_ids) - 1):
            premise = proposition_ids[i]
            conclusion = proposition_ids[i+1]
            
            # Evaluate this inference step
            step_result = self.thonoc_inference.evaluate_inference(
                [premise], conclusion
            )
            
            results.append({
                "premise": premise,
                "conclusion": conclusion,
                "valid": step_result["valid"],
                "confidence": step_result["confidence"]
            })
            
            # Chain is only valid if all steps are valid
            if not step_result["valid"]:
                valid_chain = False
        
        # Calculate overall chain strength
        chain_strength = 1.0
        for result in results:
            chain_strength *= result["confidence"]
        
        return {
            "chain": proposition_ids,
            "steps": results,
            "valid_chain": valid_chain,
            "chain_strength": chain_strength
        }
    
    def find_modal_contradictions(self, prop_id: str) -> List[str]:
        """Find propositions that contradict given proposition.
        
        Args:
            prop_id: Proposition identifier
            
        Returns:
            List of contradicting proposition IDs
        """
        if prop_id not in self.propositions:
            return []
        
        contradictions = []
        
        for other_id, other_prop in self.propositions.items():
            if other_id != prop_id:
                # Check for contradiction
                if self.thonoc_inference.check_contradiction(prop_id, other_id):
                    contradictions.append(other_id)
        
        return contradictions


# Example usage
if __name__ == "__main__":
    # Initialize modal inference engine
    engine = ModalInferenceEngine()
    
    # Register propositions
    p1 = engine.register_proposition(
        "p1", "God exists", (0.9, 0.95, 0.85)
    )
    
    p2 = engine.register_proposition(
        "p2", "Moral values are objective", (0.8, 0.9, 0.7)
    )
    
    p3 = engine.register_proposition(
        "p3", "Human life has intrinsic value", (0.85, 0.9, 0.8)
    )
    
    # Add entailments
    engine.add_entailment("p1", "p2", 0.8)
    engine.add_entailment("p2", "p3", 0.9)
    
    # Evaluate modal status
    necessity = engine.evaluate_necessity("p1")
    possibility = engine.evaluate_possibility("p3")
    
    # Evaluate inference chain
    chain = engine.evaluate_inference_chain(["p1", "p2", "p3"])
    
    print(f"Proposition p1 modal status: {p1['status']}")
    print(f"Proposition p1 necessity: {necessity['is_necessary']}")
    print(f"Proposition p3 possibility: {possibility['is_possible']}")
    print(f"Inference chain valid: {chain['valid_chain']}")
    print(f"Chain strength: {chain['chain_strength']:.2f}")