"""
File: tetragnos/odbc_kernel.py
Title: Logos MetaLogic Kernel (LMK) — ODBC Kernel orchestrator

This module hardwires the Orthogonal Dual‑Bijection Confluence (ODBC) into a
single callable that produces a Transcendental Lock (TLM) capability when both
lines (ETGC and MESH) pass and commute. It is designed to be dropped into the
existing framework alongside previously created modules:
  - tetragnos/axioms/trinitarian_bijection.py  (ETGC §5 checks)
  - tetragnos/axioms/meta_bijections.py        (MESH line + commutation)
  - tetragnos/axioms/tlm_lock.py               (TLM issuing)

Integration stubs are provided at the bottom for ARCHON, TELOS, and THONOC.
All names follow the lowercase_with_underscores convention.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timedelta, timezone

# Import the previously created building blocks
from tetragnos.axioms.trinitarian_bijection import verify_trinitarian_bijection
from tetragnos.axioms.meta_bijections import verify_meta_bijection
from tetragnos.axioms.tlm_lock import acquire_tlm


# -----------------------------
# Capability: LockContext
# -----------------------------
@dataclass(frozen=True)
class LockContext:
    token: str
    policy_version: str
    issued_at: datetime
    ttl_seconds: int = 300  # short TTL for continual conformity (5 minutes default)

    def is_valid(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        expiry = self.issued_at + timedelta(seconds=self.ttl_seconds)
        return now <= expiry


# -----------------------------
# Kernel: run_odbc_kernel
# -----------------------------

def run_odbc_kernel(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the ODBC Kernel for a single proposition/plan request.

    Expected request keys (extend as needed):
      - request_id: str
      - proposition_or_plan: Any
      - priors, policy_version, thresholds_version (optional)

    Returns a dict with fields:
      - decision: 'locked' | 'reject' | 'quarantine'
      - etgc_line: {...}
      - mesh_line: {...}
      - commutation: {...}
      - tlm: { locked: bool, token: str }
    """
    # 1) ETGC §5 checks (Unity=1, Trinity=3, ratio=1/3, bijection f, grounding & identity tests)
    etgc_inv, f_ok, grounding_ok, identity_ok = verify_trinitarian_bijection()
    etgc_ok = (
        etgc_inv.unity == 1
        and etgc_inv.trinity == 3
        and etgc_inv.ratio == 1/3
        and f_ok
        and grounding_ok
        and identity_ok
    )

    if not etgc_ok:
        return {
            "decision": "quarantine",
            "reason": "ETGC line failed (normative inadmissibility)",
            "etgc_line": {
                "unity": etgc_inv.unity,
                "trinity": etgc_inv.trinity,
                "ratio": etgc_inv.ratio,
                "bijection_ok": f_ok,
                "grounding_ok": grounding_ok,
                "identity_ok": identity_ok,
            },
            "mesh_line": None,
            "commutation": None,
            "tlm": {"locked": False, "token": ""},
        }

    # 2) MESH line + commutation via meta‑bijection aggregator
    meta = verify_meta_bijection()

    # Shape normalized outputs
    mesh_line = {
        "unity": meta.get("mesh_invariants", {}).get("unity"),
        "trinity": meta.get("mesh_invariants", {}).get("trinity"),
        "ratio": meta.get("mesh_invariants", {}).get("ratio"),
        "bijection_ok": meta.get("mesh_bijection_ok"),
        "sign_ok": meta.get("sign_ok"),
        "mind_ok": meta.get("mind_ok"),
        "bridge_ok": meta.get("bridge_ok"),
    }
    comm = {
        "t_to_o_ok": meta.get("commute_T_to_O"),
        "p_to_o_ok": meta.get("commute_P_to_O"),
    }

    # If MESH or commutation fails → reject
    if not (mesh_line["unity"] == 1 and mesh_line["trinity"] == 3 and mesh_line["ratio"] == 1/3 and mesh_line["bijection_ok"] and comm["t_to_o_ok"] and comm["p_to_o_ok"]):
        return {
            "decision": "reject",
            "reason": "MESH line or commutation failed (not instantiable / misconfigured)",
            "etgc_line": {
                "unity": etgc_inv.unity,
                "trinity": etgc_inv.trinity,
                "ratio": etgc_inv.ratio,
                "bijection_ok": f_ok,
                "grounding_ok": grounding_ok,
                "identity_ok": identity_ok,
            },
            "mesh_line": mesh_line,
            "commutation": comm,
            "tlm": {"locked": False, "token": ""},
        }

    # 3) All good → request TLM
    tlm = acquire_tlm({
        "all_ok": True,
        "etgc_bijection_ok": True,
        "mesh_bijection_ok": True,
        "commute_T_to_O": True,
        "commute_P_to_O": True,
        "etgc_invariants": {"unity": 1, "trinity": 3, "ratio": 1/3},
        "mesh_invariants": {"unity": 1, "trinity": 3, "ratio": 1/3},
        "request_id": request.get("request_id"),
        "policy_version": request.get("policy_version", "v1"),
    })

    if not tlm.locked:
        return {
            "decision": "reject",
            "reason": f"TLM failed to lock: {tlm.reasons}",
            "etgc_line": {
                "unity": etgc_inv.unity,
                "trinity": etgc_inv.trinity,
                "ratio": etgc_inv.ratio,
                "bijection_ok": f_ok,
                "grounding_ok": grounding_ok,
                "identity_ok": identity_ok,
            },
            "mesh_line": mesh_line,
            "commutation": comm,
            "tlm": {"locked": False, "token": ""},
        }

    # 4) Success: produce LockContext and canonical response
    lock_ctx = LockContext(
        token=tlm.token,
        policy_version=request.get("policy_version", "v1"),
        issued_at=datetime.now(timezone.utc),
        ttl_seconds=request.get("ttl_seconds", 300),
    )

    return {
        "decision": "locked",
        "etgc_line": {
            "unity": etgc_inv.unity,
            "trinity": etgc_inv.trinity,
            "ratio": etgc_inv.ratio,
            "bijection_ok": f_ok,
            "grounding_ok": grounding_ok,
            "identity_ok": identity_ok,
        },
        "mesh_line": mesh_line,
        "commutation": comm,
        "tlm": {"locked": True, "token": lock_ctx.token, "policy_version": lock_ctx.policy_version, "issued_at": lock_ctx.issued_at.isoformat()},
    }


# ---------------------------------
# Integration Stubs (copy as needed)
# ---------------------------------

def archon_gateway_middleware(request: Dict[str, Any]) -> Dict[str, Any]:
    """Example ARCHON gateway hook enforcing the kernel before handling a request."""
    kernel_result = run_odbc_kernel(request)
    if kernel_result["decision"] != "locked":
        # Return human‑readable reasons to the user; no execution permitted
        return {"status": "denied", "odbc": kernel_result}
    # Attach lock info and continue to business logic
    request["lock"] = kernel_result["tlm"]
    return {"status": "allowed", "odbc": kernel_result}


def telos_write_guard(node: Dict[str, Any], lock: Optional[Dict[str, Any]]) -> None:
    """Rejects writes to TELOS unless a valid lock is present; send ETGC‑failed items to quarantine."""
    if not lock or not lock.get("locked"):
        raise PermissionError("TELOS write rejected: missing/invalid TLM token")
    # Attach provenance fields expected by TELOS schema
    node.setdefault("tlm_token", lock.get("token", ""))
    node.setdefault("policy_version", lock.get("policy_version", "v1"))
    node.setdefault("issued_at", lock.get("issued_at", ""))


def thonoc_guard(plan_request: Dict[str, Any]) -> Dict[str, Any]:
    """Blocks planning unless a lock is present; downstream engines must respect BRIDGE/SIGN/MIND flags."""
    lock = plan_request.get("lock")
    if not lock or not lock.get("locked"):
        return {"status": "denied", "reason": "no TLM"}
    # Here you would filter search space by BRIDGE (¬◇) and enforce SIGN/MIND constraints
    return {"status": "allowed"}
