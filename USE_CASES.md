# Use Cases

Status ledger for use cases under `use-cases/`. Machine-maintained — the `define-use-case` skill appends rows; the dev-team orchestrator updates the `Status` and `Updated` columns as it works. Do not hand-edit those two columns unless you know why; edit the use-case file or re-run the skill instead.

Statuses:
- `pending` — saved but not yet picked up by the dev-team
- `in-progress` — the dev-team has started analysis
- `done` — implementation and tests completed
- `blocked` — the dev-team escalated (6-round cap hit, user abort, or infeasibility)

| # | File | Title | Status | Updated |
|---|------|-------|--------|---------|
| 01 | [use-cases/01-connectome-plumbing-poc.md](use-cases/01-connectome-plumbing-poc.md) | Connectome plumbing POC | pending | 2026-09-16 |
| 02 | [use-cases/02-start-gate-finish-flight.md](use-cases/02-start-gate-finish-flight.md) | Start→gate→finish flight training | pending | 2026-09-16 |
