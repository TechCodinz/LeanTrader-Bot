# Quantum (IBM) – Quick Start

This repo supports optional IBM Quantum Runtime connectivity. Use the small CLI to validate your setup locally, and an opt‑in GitHub Actions job to check connectivity in CI.

## Local connectivity check

1) Install deps (in your venv):

```
pip install python-dotenv click qiskit qiskit-ibm-runtime
```

2) Put credentials in `.env` (or export as env vars):

```
IBM_QUANTUM_API_KEY=your_ibm_quantum_api_key
IBM_QUANTUM_INSTANCE=ibm-q/open/main  # or your hub/group/project (optional)
IBM_QUANTUM_REGION=us-east            # optional, metadata only
Q_ENABLE_QUANTUM=true                 # optional feature flag
```

3) Run the CLI:

```
python cli/test_ibm_runtime.py \
  --instance "${IBM_QUANTUM_INSTANCE}" \
  --region   "${IBM_QUANTUM_REGION}" \
  --min-qubits 127 --resilience-level 1 --timeout 30
```

- If flags are omitted, it falls back to values in `config.py` (which loads `.env`).
- JSON output includes `instance` and `region` whether from CLI flags or config.
 - PowerShell example:
   `python cli/test_ibm_runtime.py --instance $env:IBM_QUANTUM_INSTANCE --region $env:IBM_QUANTUM_REGION`

## CI connectivity job (GitHub Actions)

An opt‑in job lives at `.github/workflows/ibm_quantum_connectivity.yml`. It runs on a schedule and on manual dispatch, but is gated behind a repo variable.

Configure in GitHub → Settings → Secrets and variables → Actions:

- Secret: `IBM_QUANTUM_API_KEY` (required)
- Variables:
  - `Q_ENABLE_QUANTUM` = `true` (enables the job)
  - `IBM_QUANTUM_INSTANCE` (optional)
  - `IBM_QUANTUM_REGION` (optional)

Trigger manually from the Actions tab ("IBM Quantum Connectivity") or wait for the scheduled run. The job uploads `ibm_connectivity.json` as an artifact with the result.
