"""
File: tetragnos/system_init.py
Title: System Initialization & Compliance Framework

This module ensures each subsystem (ARCHON, THONOC, TELOS, TETRAGNOS) must pass
ODBC validation during startup and maintain periodic compliance validation.
No subsystem can execute without a valid TLM token.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
from threading import Thread, Event, Lock
from enum import Enum
import time
import logging

from tetragnos.odbc_kernel import run_odbc_kernel, LockContext


class SubsystemState(Enum):
    UNINITIALIZED = "uninitialized"
    VALIDATING = "validating" 
    AUTHORIZED = "authorized"
    COMPLIANCE_CHECK = "compliance_check"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


@dataclass
class SubsystemAuth:
    name: str
    state: SubsystemState
    tlm_token: Optional[str] = None
    policy_version: str = "v1"
    last_validation: Optional[datetime] = None
    compliance_interval: int = 300  # 5 minutes default
    next_check: Optional[datetime] = None
    failure_count: int = 0
    max_failures: int = 3


class SystemInitializer:
    """Manages startup authorization and ongoing compliance for all subsystems."""
    
    def __init__(self):
        self.subsystems: Dict[str, SubsystemAuth] = {}
        self.compliance_thread: Optional[Thread] = None
        self.shutdown_event = Event()
        self.auth_lock = Lock()
        self.logger = logging.getLogger(__name__)
        
    def register_subsystem(self, name: str, compliance_interval: int = 300) -> None:
        """Register a subsystem for initialization and compliance monitoring."""
        with self.auth_lock:
            self.subsystems[name] = SubsystemAuth(
                name=name,
                state=SubsystemState.UNINITIALIZED,
                compliance_interval=compliance_interval
            )
            self.logger.info(f"Registered subsystem: {name}")
    
    def initialize_subsystem(self, name: str, subsystem_config: Dict[str, Any]) -> bool:
        """
        Initialize a subsystem by validating it through ODBC kernel.
        Returns True if authorized, False otherwise.
        """
        if name not in self.subsystems:
            self.logger.error(f"Unknown subsystem: {name}")
            return False
            
        auth = self.subsystems[name]
        
        with self.auth_lock:
            auth.state = SubsystemState.VALIDATING
            
        self.logger.info(f"Initializing subsystem: {name}")
        
        # Create initialization request with subsystem-specific context
        init_request = {
            "request_id": f"init_{name}_{datetime.now().isoformat()}",
            "proposition_or_plan": {
                "subsystem": name,
                "initialization": True,
                "config": subsystem_config,
                "compliance_requirements": {
                    "etgc_compliance": True,
                    "mesh_compliance": True, 
                    "commutation_compliance": True
                }
            },
            "policy_version": auth.policy_version,
            "ttl_seconds": auth.compliance_interval
        }
        
        # Run ODBC validation
        odbc_result = run_odbc_kernel(init_request)
        
        with self.auth_lock:
            if odbc_result["decision"] == "locked":
                auth.state = SubsystemState.AUTHORIZED
                auth.tlm_token = odbc_result["tlm"]["token"]
                auth.last_validation = datetime.now(timezone.utc)
                auth.next_check = auth.last_validation + timedelta(seconds=auth.compliance_interval)
                auth.failure_count = 0
                
                self.logger.info(f"Subsystem {name} AUTHORIZED with token: {auth.tlm_token[:8]}...")
                return True
                
            elif odbc_result["decision"] == "quarantine":
                auth.state = SubsystemState.QUARANTINED
                self.logger.warning(f"Subsystem {name} QUARANTINED: {odbc_result.get('reason', 'ETGC failure')}")
                return False
                
            else:  # rejected
                auth.state = SubsystemState.REJECTED
                auth.failure_count += 1
                self.logger.error(f"Subsystem {name} REJECTED: {odbc_result.get('reason', 'MESH/commutation failure')}")
                return False
    
    def is_subsystem_authorized(self, name: str) -> bool:
        """Check if a subsystem is currently authorized to execute."""
        if name not in self.subsystems:
            return False
            
        auth = self.subsystems[name]
        return auth.state == SubsystemState.AUTHORIZED and auth.tlm_token is not None
    
    def get_subsystem_token(self, name: str) -> Optional[str]:
        """Get the current TLM token for a subsystem."""
        if not self.is_subsystem_authorized(name):
            return None
        return self.subsystems[name].tlm_token
    
    def start_compliance_monitoring(self) -> None:
        """Start the background compliance checking thread."""
        if self.compliance_thread and self.compliance_thread.is_alive():
            return
            
        self.compliance_thread = Thread(
            target=self._compliance_monitor_loop,
            name="ComplianceMonitor",
            daemon=True
        )
        self.compliance_thread.start()
        self.logger.info("Started compliance monitoring thread")
    
    def _compliance_monitor_loop(self) -> None:
        """Background thread that periodically revalidates all subsystems."""
        while not self.shutdown_event.is_set():
            try:
                now = datetime.now(timezone.utc)
                
                for name, auth in self.subsystems.items():
                    if auth.state == SubsystemState.AUTHORIZED and auth.next_check and now >= auth.next_check:
                        self._revalidate_subsystem(name)
                        
                # Check every 30 seconds
                self.shutdown_event.wait(30)
                
            except Exception as e:
                self.logger.error(f"Compliance monitor error: {e}")
                time.sleep(60)  # Back off on errors
    
    def _revalidate_subsystem(self, name: str) -> None:
        """Revalidate a subsystem's compliance with LOGOS/ODBC."""
        auth = self.subsystems[name]
        
        with self.auth_lock:
            auth.state = SubsystemState.COMPLIANCE_CHECK
            
        self.logger.info(f"Revalidating subsystem: {name}")
        
        # Create compliance check request
        compliance_request = {
            "request_id": f"compliance_{name}_{datetime.now().isoformat()}",
            "proposition_or_plan": {
                "subsystem": name,
                "compliance_check": True,
                "current_token": auth.tlm_token,
                "last_validation": auth.last_validation.isoformat() if auth.last_validation else None
            },
            "policy_version": auth.policy_version,
            "ttl_seconds": auth.compliance_interval
        }
        
        # Run ODBC validation
        odbc_result = run_odbc_kernel(compliance_request)
        
        with self.auth_lock:
            if odbc_result["decision"] == "locked":
                # Successful revalidation - update token and schedule next check
                auth.state = SubsystemState.AUTHORIZED
                auth.tlm_token = odbc_result["tlm"]["token"]
                auth.last_validation = datetime.now(timezone.utc)
                auth.next_check = auth.last_validation + timedelta(seconds=auth.compliance_interval)
                auth.failure_count = 0
                
                self.logger.info(f"Subsystem {name} compliance RENEWED")
                
            else:
                # Compliance failure
                auth.failure_count += 1
                
                if auth.failure_count >= auth.max_failures:
                    auth.state = SubsystemState.QUARANTINED
                    auth.tlm_token = None
                    self.logger.critical(f"Subsystem {name} QUARANTINED after {auth.failure_count} failures")
                else:
                    auth.state = SubsystemState.REJECTED
                    auth.tlm_token = None
                    # Schedule retry in 1 minute
                    auth.next_check = datetime.now(timezone.utc) + timedelta(minutes=1)
                    self.logger.warning(f"Subsystem {name} compliance failure {auth.failure_count}/{auth.max_failures}")
    
    def shutdown(self) -> None:
        """Shutdown the compliance monitoring system."""
        self.shutdown_event.set()
        if self.compliance_thread:
            self.compliance_thread.join(timeout=5)
        self.logger.info("System initializer shutdown complete")


# Global system initializer instance
system_init = SystemInitializer()


# Decorator for protecting subsystem methods
def require_authorization(subsystem_name: str):
    """Decorator that ensures a subsystem method can only run with valid authorization."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if not system_init.is_subsystem_authorized(subsystem_name):
                raise PermissionError(f"Subsystem {subsystem_name} not authorized - cannot execute {func.__name__}")
            
            # Inject TLM token into kwargs if not present
            if 'tlm_token' not in kwargs:
                kwargs['tlm_token'] = system_init.get_subsystem_token(subsystem_name)
                
            return func(*args, **kwargs)
        return wrapper
    return decorator


# -----------------------------------------------------------------------------
# EXAMPLE SUBSYSTEM IMPLEMENTATIONS
# -----------------------------------------------------------------------------

class ArchonSubsystem:
    """Example ARCHON subsystem with authorization requirements."""
    
    def __init__(self):
        self.name = "ARCHON"
        
    def initialize(self) -> bool:
        """Initialize ARCHON subsystem."""
        config = {
            "interface_type": "natural_language",
            "user_facing": True,
            "requires_etgc": True,
            "requires_mesh": True
        }
        return system_init.initialize_subsystem(self.name, config)
    
    @require_authorization("ARCHON")
    def handle_user_request(self, user_input: str, tlm_token: str = None) -> str:
        """Process user request with TLM authorization."""
        # Implementation here would use tlm_token for validation
        return f"Processing: {user_input} with token {tlm_token[:8]}..."


class TetragnasSubsystem:
    """Example TETRAGNOS translation engine with authorization."""
    
    def __init__(self):
        self.name = "TETRAGNOS"
        
    def initialize(self) -> bool:
        """Initialize TETRAGNOS subsystem."""
        config = {
            "translation_engine": True,
            "logos_logic": True,
            "bidirectional": True,
            "requires_etgc": True,
            "requires_mesh": True,
            "requires_commutation": True
        }
        return system_init.initialize_subsystem(self.name, config)
    
    @require_authorization("TETRAGNOS")
    def translate_nl_to_computation(self, text: str, tlm_token: str = None) -> Dict[str, Any]:
        """Translate natural language to computational form."""
        # Implementation here would apply LOGOS logic with TLM validation
        return {"computation": f"Translated: {text}", "token": tlm_token[:8]}
    
    @require_authorization("TETRAGNOS") 
    def translate_computation_to_nl(self, data: Dict[str, Any], tlm_token: str = None) -> str:
        """Translate computation back to natural language."""
        # Implementation here would reverse translate with TLM validation
        return f"Natural language result with token {tlm_token[:8]}..."


# System startup sequence
def initialize_all_subsystems() -> bool:
    """Initialize all subsystems in proper order."""
    # Register all subsystems
    system_init.register_subsystem("TETRAGNOS", compliance_interval=300)  # 5 min
    system_init.register_subsystem("ARCHON", compliance_interval=300)
    system_init.register_subsystem("THONOC", compliance_interval=300) 
    system_init.register_subsystem("TELOS", compliance_interval=300)
    
    # Initialize each subsystem
    subsystems = [
        ("TETRAGNOS", TetragnasSubsystem()),
        ("ARCHON", ArchonSubsystem()),
        # Add THONOC and TELOS when implemented
    ]
    
    all_initialized = True
    for name, subsystem in subsystems:
        if not subsystem.initialize():
            logging.error(f"Failed to initialize {name}")
            all_initialized = False
    
    if all_initialized:
        # Start compliance monitoring
        system_init.start_compliance_monitoring()
        logging.info("All subsystems initialized and compliance monitoring started")
    
    return all_initialized


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = initialize_all_subsystems()
    if success:
        print("System ready - all subsystems authorized")
    else:
        print("System initialization failed")