# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report it privately via [GitHub Security Advisories](https://github.com/Ashok007-cmd/production-grade-rag/security/advisories/new) for this repository. This lets us assess and patch the issue before it's publicly disclosed.

Include as much detail as you can:
- Affected file(s)/endpoint(s) and version (commit SHA or tag)
- Steps to reproduce, or a proof-of-concept
- Impact assessment (what an attacker could do)

## Supported Versions

This project tracks `main` as the only actively supported branch. There are no maintained release branches — security fixes land on `main` and are included in the next tagged release / Docker image publish.

## Scope

In scope:
- The application code in `src/`, `scripts/`, and the `Dockerfile`/`docker-compose.yml`
- CI/CD workflow configuration in `.github/workflows/`

Out of scope:
- Vulnerabilities in third-party dependencies — these are tracked separately via [Dependabot](https://github.com/Ashok007-cmd/production-grade-rag/security/dependabot) and `pip-audit` (see `ANALYSIS_REPORT.md` for the current audit trail). Please report upstream to the dependency's own maintainers unless this project's usage of it introduces the vulnerability.
- Denial-of-service issues that require significant resources to exploit (e.g. large-payload resource exhaustion) — please still report if concretely exploitable with minimal effort.

## Disclosure Process

1. Report privately via GitHub Security Advisories (link above).
2. We aim to acknowledge reports within a few days and provide an initial assessment.
3. A fix is developed and released; the advisory is credited to the reporter (unless anonymity is requested) and published once a patch is available.
