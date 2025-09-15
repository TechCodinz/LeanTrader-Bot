from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import click

# Try to load repo config for defaults
try:
    import config as _cfg  # type: ignore
except Exception:
    _cfg = None  # type: ignore


def _singleline(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _backend_info(b) -> Tuple[Optional[str], Optional[int], bool]:
    name = None
    nq = None
    sim = False
    try:
        name = getattr(b, "name", None) or getattr(b, "backend_name", None) or str(b)
    except Exception:
        name = None
    # num_qubits
    try:
        nq = getattr(b, "num_qubits", None)
        if nq is None and hasattr(b, "configuration"):
            cfg = b.configuration()
            nq = getattr(cfg, "n_qubits", None) or getattr(cfg, "num_qubits", None)
    except Exception:
        nq = None
    # simulator flag
    try:
        sim = bool(getattr(b, "simulator", False))
        if not sim and hasattr(b, "configuration"):
            cfg = b.configuration()
            sim = bool(getattr(cfg, "simulator", False))
    except Exception:
        sim = False
    return name, (int(nq) if isinstance(nq, (int, float)) else None), bool(sim)


def run_ibm_check(
    api_key: Optional[str],
    instance: Optional[str] = None,
    region: Optional[str] = None,
    min_qubits: int = 127,
    resilience_level: int = 1,
    timeout: int = 30,
) -> Tuple[Dict[str, Any], int]:
    # Step 1: import qiskit_ibm_runtime
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler  # type: ignore
    except Exception:
        res = {
            "ok": False,
            "account_ok": False,
            "region": region,
            "instance": instance,
            "backend": {"name": None, "num_qubits": None},
            "has_simulators": False,
            "backends_sample": [],
            "sampler_ok": False,
            "queue_info": None,
            "resilience_level": int(resilience_level),
            "error": "qiskit-ibm-runtime not installed",
        }
        return res, 1

    # Try import for building the test circuit
    try:
        from qiskit import QuantumCircuit  # type: ignore
    except Exception:
        QuantumCircuit = None  # type: ignore

    # Step 2: resolve API key
    key = (api_key or os.getenv("IBM_QUANTUM_API_KEY", "").strip())
    if not key:
        res = {
            "ok": False,
            "account_ok": False,
            "region": region,
            "instance": instance,
            "backend": {"name": None, "num_qubits": None},
            "has_simulators": False,
            "backends_sample": [],
            "sampler_ok": False,
            "queue_info": None,
            "resilience_level": int(resilience_level),
            "error": "Missing IBM_QUANTUM_API_KEY",
        }
        return res, 1

    account_ok = False
    # Step 3: save account
    try:
        QiskitRuntimeService.save_account(channel="ibm_quantum", token=key, overwrite=False)  # type: ignore
    except Exception:
        pass

    # Step 4: create service and set instance if provided
    backend = None
    backends_sample: List[Dict[str, Any]] = []
    has_simulators = False
    queue_info = None
    sampler_ok = False
    backend_name = None
    backend_nq = None

    try:
        service = QiskitRuntimeService()  # type: ignore
        account_ok = True
        if instance:
            try:
                # Some versions support set_default_instance
                if hasattr(service, "set_default_instance"):
                    service.set_default_instance(instance)
            except Exception:
                pass
        # Step 5: list backends
        try:
            bks = service.backends()  # type: ignore
        except Exception:
            bks = []
        for b in bks[:10]:
            nm, nq, sim = _backend_info(b)
            backends_sample.append({"name": nm, "num_qubits": nq, "simulator": sim})
        has_simulators = any(x.get("simulator") for x in backends_sample)
        # least busy
        try:
            backend = service.least_busy(min_num_qubits=int(min_qubits))  # type: ignore
        except Exception:
            backend = None
        if backend is not None:
            backend_name, backend_nq, _ = _backend_info(backend)
            # queue size
            try:
                st = backend.status()
                q = getattr(st, "pending_jobs", None)
                queue_info = int(q) if isinstance(q, (int, float)) else None
            except Exception:
                queue_info = None
    except Exception as e:
        # service init failed
        res = {
            "ok": False,
            "account_ok": account_ok,
            "region": region,
            "instance": instance,
            "backend": {"name": backend_name, "num_qubits": backend_nq},
            "has_simulators": has_simulators,
            "backends_sample": backends_sample,
            "sampler_ok": False,
            "queue_info": queue_info,
            "resilience_level": int(resilience_level),
            "error": str(e),
        }
        return res, 1

    # Step 6: run a tiny sampler job
    try:
        if backend is not None and QuantumCircuit is not None:
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            # Use session if available
            with Session(service=service, backend=backend):  # type: ignore
                sampler = Sampler(options={"resilience_level": int(resilience_level)})  # type: ignore
                job = sampler.run(circuits=[qc])
                # attempt to wait
                try:
                    res_obj = job.result(timeout=int(timeout))
                except TypeError:
                    res_obj = job.result()
                # a quick sanity: check existence of quasi_dists or results
                sampler_ok = bool(res_obj)
    except Exception:
        sampler_ok = False

    ok = bool(account_ok and (backend is not None) and sampler_ok)
    res = {
        "ok": ok,
        "account_ok": account_ok,
        "region": region,
        "instance": instance,
        "backend": {"name": backend_name, "num_qubits": backend_nq},
        "has_simulators": has_simulators,
        "backends_sample": backends_sample,
        "sampler_ok": sampler_ok,
        "queue_info": queue_info,
        "resilience_level": int(resilience_level),
        "error": None if ok else "Sampler or backend failed",
    }
    return res, (0 if ok else 1)


@click.command()
@click.option("--api-key", "api_key", default=None, help="IBM Quantum API Key")
@click.option("--instance", default=None, help="IBM Quantum instance name")
@click.option("--region", default=None, help="Region hint (metadata only)")
@click.option("--min-qubits", default=127, type=int, help="Minimum qubits for least-busy selection")
@click.option("--resilience-level", default=1, type=int, help="Resilience level (0..3)")
@click.option("--timeout", default=30, type=int, help="Timeout seconds for result")
def cli(api_key: Optional[str], instance: Optional[str], region: Optional[str], min_qubits: int, resilience_level: int, timeout: int) -> None:
    # Resolve from args or config module/env
    resolved_instance = instance or (getattr(_cfg, "IBM_QUANTUM_INSTANCE", None) if _cfg else None) or os.getenv("IBM_QUANTUM_INSTANCE", None)
    resolved_region = region or (getattr(_cfg, "IBM_QUANTUM_REGION", None) if _cfg else None) or os.getenv("IBM_QUANTUM_REGION", None)
    resolved_api_key = api_key or (getattr(_cfg, "IBM_QUANTUM_API_KEY", None) if _cfg else None) or os.getenv("IBM_QUANTUM_API_KEY", None)

    res, code = run_ibm_check(resolved_api_key, resolved_instance, resolved_region, min_qubits, resilience_level, timeout)
    print(_singleline(res))
    raise SystemExit(code)


if __name__ == "__main__":
    cli(standalone_mode=False)

