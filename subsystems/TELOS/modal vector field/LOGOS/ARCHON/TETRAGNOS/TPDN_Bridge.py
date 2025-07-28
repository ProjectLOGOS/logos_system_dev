"""3PDN-Lambda Bridge Module

Implements the bridge between natural language processing (3PDN Translation Engine)
and the Lambda Logos core, providing bidirectional translation and addressing
the 3PDN bottleneck.

Key components:
- Natural language to Lambda conversion
- Lambda to 3PDN representation
- 3PDN bottleneck optimization
- Translation target generation

Dependencies: typing, json, lambda_logos_core, ontological_node
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import json

# Import from Lambda Logos core (adjust imports as needed)
try:
    from lambda_logos_core import (
        LambdaLogosEngine, LogosExpr, Variable, Value, Abstraction,
        Application, SufficientReason, OntologicalType, Constant
    )
except ImportError:
    # Mock classes for standalone use
    class OntologicalType:
        EXISTENCE = "𝔼"
        GOODNESS = "𝔾"
        TRUTH = "𝕋"
        PROP = "Prop"
    
    class LogosExpr:
        def to_dict(self): return {}
        def __str__(self): return "LogosExpr"
    
    class Variable(LogosExpr):
        def __init__(self, name, ont_type): 
            self.name = name
            self.ont_type = ont_type
    
    class Value(LogosExpr):
        def __init__(self, value, ont_type): 
            self.value = value
            self.ont_type = ont_type
    
    class Constant(LogosExpr):
        def __init__(self, name, const_type): 
            self.name = name
            self.const_type = const_type
    
    class Application(LogosExpr):
        def __init__(self, func, arg): 
            self.func = func
            self.arg = arg
    
    class SufficientReason(LogosExpr):
        def __init__(self, source_type, target_type, value): 
            self.source_type = source_type
            self.target_type = target_type
            self.value = value
    
    class LambdaLogosEngine:
        def evaluate(self, expr): return expr

# Try to import from ontological node (adjust imports as needed)
try:
    from ontological_node import OntologicalNode
except ImportError:
    # Placeholder if not available
    class OntologicalNode:
        def __init__(self, c_value): 
            self.c = c_value
            self.trinity_vector = (0.5, 0.5, 0.5)
        
        def update_lambda_term(self, term, env=None): pass

class PDNBridge:
    """Bridge between 3PDN Translation Engine and Lambda Logos."""
    
    def __init__(self, lambda_engine: Optional[LambdaLogosEngine] = None):
        """Initialize PDN bridge.
        
        Args:
            lambda_engine: Lambda Logos engine instance (optional)
        """
        self.lambda_engine = lambda_engine or LambdaLogosEngine()
        
        # Dictionary of common terms for quick translation
        self.common_terms = self._initialize_common_terms()
    
    def _initialize_common_terms(self) -> Dict[str, LogosExpr]:
        """Initialize dictionary of common Lambda terms.
        
        Returns:
            Dictionary of common terms
        """
        # Ontological values
        ei_val = Value("ei", OntologicalType.EXISTENCE)
        og_val = Value("og", OntologicalType.GOODNESS)
        at_val = Value("at", OntologicalType.TRUTH)
        
        # Sufficient reason operators
        sr_eg = SufficientReason(OntologicalType.EXISTENCE, OntologicalType.GOODNESS, 3)
        sr_gt = SufficientReason(OntologicalType.GOODNESS, OntologicalType.TRUTH, 2)
        
        # Basic applications
        eg_app = Application(sr_eg, ei_val)
        gt_app = Application(sr_gt, og_val)
        
        # Connect dictionary
        return {
            "existence": ei_val,
            "goodness": og_val,
            "truth": at_val,
            "sr_eg": sr_eg,
            "sr_gt": sr_gt,
            "existence_implies_goodness": eg_app,
            "goodness_implies_truth": gt_app
        }
    
    def natural_to_lambda(self, query: str, translation_result: Optional[Dict[str, Any]] = None) -> Tuple[LogosExpr, Dict[str, Any]]:
        """Convert natural language to Lambda expression.
        
        Args:
            query: Natural language query
            translation_result: Optional external translation result
            
        Returns:
            (Lambda expression, Translation result) tuple
        """
        # If translation result provided, use it
        if translation_result:
            return self._translation_to_lambda(translation_result), translation_result
        
        # Otherwise, create a mock translation result
        mock_translation = self._mock_translate(query)
        lambda_expr = self._translation_to_lambda(mock_translation)
        
        return lambda_expr, mock_translation
    
    def _mock_translate(self, query: str) -> Dict[str, Any]:
        """Create mock 3PDN translation for query.
        
        Args:
            query: Natural language query
            
        Returns:
            Mock translation result
        """
        # Extract keywords for simple categorization
        tokens = query.lower().split()
        
        # Very basic keyword analysis (real 3PDN would be sophisticated)
        existential = any(word in tokens for word in ["exist", "existence", "reality", "being"])
        moral = any(word in tokens for word in ["good", "moral", "ethical", "right", "wrong"])
        truth = any(word in tokens for word in ["true", "truth", "knowledge", "fact"])
        
        # Set default EGT vector
        e_value = 0.7 if existential else 0.5
        g_value = 0.7 if moral else 0.5
        t_value = 0.7 if truth else 0.5
        
        # Create basic 3PDN structure
        return {
            "query": query,
            "trinity_vector": (e_value, g_value, t_value),
            "layers": {
                "sign": tokens,
                "mind": {
                    "ontological": 0.6 if existential else 0.3,
                    "moral": 0.7 if moral else 0.2,
                    "epistemic": 0.8 if truth else 0.3
                },
                "bridge": {
                    "existence": e_value,
                    "goodness": g_value,
                    "truth": t_value
                }
            }
        }
    
    def _translation_to_lambda(self, translation_result: Dict[str, Any]) -> LogosExpr:
        """Convert 3PDN translation to Lambda expression.
        
        Args:
            translation_result: 3PDN translation result
            
        Returns:
            Lambda expression
        """
        # Extract trinity vector
        trinity = translation_result.get("trinity_vector", (0.5, 0.5, 0.5))
        
        # Determine strongest dimension
        dims = [("existence", trinity[0]), ("goodness", trinity[1]), ("truth", trinity[2])]
        primary_dim = max(dims, key=lambda x: x[1])[0]
        
        # Return corresponding Lambda expression
        if primary_dim == "existence":
            return self.common_terms["existence"]
        elif primary_dim == "goodness":
            return self.common_terms["goodness"]
        elif primary_dim == "truth":
            return self.common_terms["truth"]
        
        # Default fallback
        return self.common_terms["existence"]
    
    def lambda_to_natural(self, expr: LogosExpr) -> str:
        """Convert Lambda expression to natural language.
        
        Args:
            expr: Lambda expression
            
        Returns:
            Natural language representation
        """
        # Basic conversion based on expression type
        if isinstance(expr, Variable):
            if expr.ont_type == OntologicalType.EXISTENCE:
                return "a concept of existence"
            elif expr.ont_type == OntologicalType.GOODNESS:
                return "a concept of goodness"
            elif expr.ont_type == OntologicalType.TRUTH:
                return "a concept of truth"
            else:
                return f"a variable named {expr.name}"
        
        elif isinstance(expr, Value):
            if expr.value == "ei":
                return "existence itself"
            elif expr.value == "og":
                return "objective goodness"
            elif expr.value == "at":
                return "absolute truth"
            else:
                return f"the value {expr.value}"
        
        elif isinstance(expr, SufficientReason):
            if (expr.source_type == OntologicalType.EXISTENCE and 
                expr.target_type == OntologicalType.GOODNESS):
                return "the principle that existence implies goodness"
            elif (expr.source_type == OntologicalType.GOODNESS and 
                  expr.target_type == OntologicalType.TRUTH):
                return "the principle that goodness implies truth"
            else:
                return f"a sufficient reason operator from {expr.source_type} to {expr.target_type}"
        
        elif isinstance(expr, Application):
            # Handle common applications
            if str(expr) == str(self.common_terms["existence_implies_goodness"]):
                return "existence implies goodness"
            elif str(expr) == str(self.common_terms["goodness_implies_truth"]):
                return "goodness implies truth"
            else:
                return f"the application of {self.lambda_to_natural(expr.func)} to {self.lambda_to_natural(expr.arg)}"
        
        # Default fallback
        return str(expr)
    
    def lambda_to_3pdn(self, expr: LogosExpr) -> Dict[str, Any]:
        """Convert Lambda expression to 3PDN representation.
        
        Args:
            expr: Lambda expression
            
        Returns:
            3PDN representation with SIGN, MIND, BRIDGE layers
        """
        # Extract type information
        type_info = self._extract_type_info(expr)
        
        # Generate semantic categories
        semantic = self._map_to_semantic(type_info)
        
        # Generate ontological dimensions
        ontological = self._map_to_ontological(semantic)
        
        # Create 3PDN representation
        return {
            "layers": {
                "sign": self._expr_to_sign(expr),
                "mind": semantic,
                "bridge": ontological
            },
            "trinity_vector": (
                ontological.get("existence", 0.5),
                ontological.get("goodness", 0.5),
                ontological.get("truth", 0.5)
            ),
            "expr": str(expr)
        }
    
    def _extract_type_info(self, expr: LogosExpr) -> Dict[str, Any]:
        """Extract type information from Lambda expression.
        
        Args:
            expr: Lambda expression
            
        Returns:
            Type information dictionary
        """
        # Real implementation would use the Lambda engine's type checker
        # This is a simplified version based on expression class
        
        if isinstance(expr, Variable) or isinstance(expr, Value):
            if hasattr(expr, 'ont_type'):
                return {"type": "simple", "value": expr.ont_type}
        
        elif isinstance(expr, SufficientReason):
            return {
                "type": "sr",
                "source": expr.source_type,
                "target": expr.target_type
            }
        
        elif isinstance(expr, Application):
            # Recursive type extraction
            func_type = self._extract_type_info(expr.func)
            arg_type = self._extract_type_info(expr.arg)
            
            return {
                "type": "application",
                "func_type": func_type,
                "arg_type": arg_type
            }
        
        # Default type info
        return {"type": "unknown"}
    
    def _map_to_semantic(self, type_info: Dict[str, Any]) -> Dict[str, float]:
        """Map type information to semantic categories.
        
        Args:
            type_info: Type information
            
        Returns:
            Semantic category weights
        """
        # Initialize with default values
        semantic = {
            "ontological": 0.0,
            "moral": 0.0,
            "epistemic": 0.0,
            "causal": 0.0,
            "modal": 0.0,
            "logical": 0.0
        }
        
        # Map simple types directly
        if type_info.get("type") == "simple":
            value = type_info.get("value")
            if value == OntologicalType.EXISTENCE:
                semantic["ontological"] = 0.8
                semantic["causal"] = 0.2
            elif value == OntologicalType.GOODNESS:
                semantic["moral"] = 0.9
                semantic["ontological"] = 0.1
            elif value == OntologicalType.TRUTH:
                semantic["epistemic"] = 0.7
                semantic["logical"] = 0.3
        
        # Map SR operators
        elif type_info.get("type") == "sr":
            source = type_info.get("source")
            target = type_info.get("target")
            
            if source == OntologicalType.EXISTENCE and target == OntologicalType.GOODNESS:
                semantic["ontological"] = 0.5
                semantic["moral"] = 0.5
            elif source == OntologicalType.GOODNESS and target == OntologicalType.TRUTH:
                semantic["moral"] = 0.4
                semantic["epistemic"] = 0.6
        
        # Map applications
        elif type_info.get("type") == "application":
            # Combine function and argument semantics
            func_type = type_info.get("func_type", {})
            arg_type = type_info.get("arg_type", {})
            
            if func_type.get("type") == "sr" and arg_type.get("type") == "simple":
                # Specific handling for SR applications
                source = func_type.get("source")
                target = func_type.get("target")
                arg_value = arg_type.get("value")
                
                if source == arg_value:
                    # Valid SR application - emphasize target dimension
                    if target == OntologicalType.GOODNESS:
                        semantic["moral"] = 0.7
                        semantic["ontological"] = 0.3
                    elif target == OntologicalType.TRUTH:
                        semantic["epistemic"] = 0.7
                        semantic["moral"] = 0.3
        
        return semantic
    
    def _map_to_ontological(self, semantic: Dict[str, float]) -> Dict[str, float]:
        """Map semantic categories to ontological dimensions.
        
        Args:
            semantic: Semantic categories
            
        Returns:
            Ontological dimension values
        """
        # Initialize with neutral values
        ontological = {
            "existence": 0.5,
            "goodness": 0.5,
            "truth": 0.5
        }
        
        # Apply semantic weights to dimensions
        if semantic.get("ontological", 0) > 0:
            ontological["existence"] = 0.5 + 0.5 * semantic["ontological"]
        
        if semantic.get("moral", 0) > 0:
            ontological["goodness"] = 0.5 + 0.5 * semantic["moral"]
        
        if semantic.get("epistemic", 0) > 0:
            ontological["truth"] = 0.5 + 0.5 * semantic["epistemic"]
        
        if semantic.get("logical", 0) > 0:
            ontological["truth"] = max(ontological["truth"], 0.5 + 0.4 * semantic["logical"])
        
        if semantic.get("causal", 0) > 0:
            ontological["existence"] = max(ontological["existence"], 0.5 + 0.3 * semantic["causal"])
        
        # Ensure values are within [0, 1]
        for key in ontological:
            ontological[key] = min(max(ontological[key], 0), 1)
        
        return ontological
    
    def _expr_to_sign(self, expr: LogosExpr) -> List[str]:
        """Convert expression to SIGN layer tokens.
        
        Args:
            expr: Lambda expression
            
        Returns:
            List of tokens
        """
        # Convert to string and tokenize
        expr_str = str(expr)
        tokens = expr_str.replace('(', ' ( ').replace(')', ' ) ').replace('.', ' . ').split()
        
        # Filter and clean
        return [token for token in tokens if token.strip()]