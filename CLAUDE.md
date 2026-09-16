# drone-fly

Project-local guidance for Claude Code. `PROJECT_BRIEF.md` is the authoritative source for
architecture, stack, paths, and quality standards; this file records repo-level conventions the
dev-team needs that don't belong in the brief's structured frontmatter.

## Dev-team write authorizations

The dev-team's **developer** agent may write to production-code paths declared in
`PROJECT_BRIEF.md` (`paths.production` → `src/drone_fly/**`). In addition, the developer is
authorized to create and modify these repository-level paths, which sit outside `paths.production`
but are required for normal implementation work:

- `pyproject.toml` — project metadata, dependencies, and tool config (ruff, pytest, mypy).
- `scripts/**` — developer utility scripts (e.g. dev-time test-fixture generators). These are not
  runtime application code and are not part of `paths.production`.
- Root user-facing docs — `README.md` (and `CHANGELOG.md` if/when one exists) — for keeping
  user-visible behavior and docs in sync per the developer role's documentation rules.
- `.env.example` — template for required environment variables.

The **QA** agent's write scope remains `paths.test` (`tests/**`) as declared in the brief, plus
`.claude/allowed-commands.yaml` per the standard dev-team protocol.

No agent may write anywhere outside this repository, and none may touch the tooling workspace it
was launched from.
