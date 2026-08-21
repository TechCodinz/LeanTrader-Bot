# LeanTrader VPS repository reconciliation bridge v1.1

This bridge exists only to recover and reconcile the known-good LeanTrader VPS source history with the canonical `TechCodinz/LeanTrader-Bot` GitHub repository.

## Preserved safety boundaries

- Paper/Testnet authority only. No tool enables live trading or exchange credentials.
- No arbitrary shell command tool.
- No emergency-halt removal tool.
- No threshold, cost-model, falsification-gate, or prospective-validation override.
- Runtime, evidence, credential, key, wallet, model, dataset, log, and private-data paths cannot be read or staged as source.
- Repository pushes are explicit, branch-scoped, canonical-origin-only, and never forced.
- `v1.34` is an annotated tag and cannot be overwritten by the bridge.

## Read operations

`leantrader_repository_read` accepts only:

- `inventory`: branch, refs, filtered status, origin, and commits not reachable from remote refs.
- `history`: bounded commit history and file statistics.
- `diff`: bounded worktree, staged, or unpushed evidence.
- `read-source`: at most 501 lines from a tracked repository source or allowlisted running-sidecar source.
- `source-inventory`: hashes and metadata for source files discovered from the Evolution Sidecar service.
- `evidence-inventory`: aggregate counts, sizes, timestamps, and manifest hashes without evidence contents.

## Write confirmations

- `RUN_REPOSITORY_TESTS`
- `CREATE_RECONCILIATION_BACKUP`
- `STAGE_REVIEWED_PATHS`
- `COMMIT_REVIEWED_BASELINE`
- `PUSH_RECONCILED_BRANCHES`
- `IMPORT_REVIEWED_VPS_SOURCE`
- `TAG_KNOWN_GOOD_V1_34`
- `PUSH_KNOWN_GOOD_V1_34_TAG`

The backup operation creates a verified all-refs Git bundle plus a locally retained evidence archive and manifests under `/opt/leantrader/backups/reconciliation`. Evidence archives are never staged or pushed.
