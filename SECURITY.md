# Security Policy

## Supported Versions

CogSynDelta is currently in active development. Security updates are provided for the following versions:

| Version | Supported          | Notes                           |
| ------- | ------------------ | ------------------------------- |
| 0.3.x   | :white_check_mark: | Current development             |
| 0.2.x   | :white_check_mark: | Previous release                |
| 0.1.x   | :x:                | Initial prototype, deprecated   |
| < 0.1   | :x:                | Pre-release, not supported      |

## Reporting a Vulnerability

The CogSynDelta team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**DO NOT** report security vulnerabilities through public GitHub issues.

Instead, please report security vulnerabilities by:

1. **Email**: Send a detailed report to [security@averagejoes-labs.com](mailto:security@averagejoes-labs.com)
2. **GitHub Security Advisories**: Use [GitHub's private vulnerability reporting](https://github.com/tzervas/CogSynDelta/security/advisories/new) (preferred)

### What to Include

Please include the following in your report:

- **Type of vulnerability** (e.g., injection, buffer overflow, privilege escalation)
- **Affected component** (e.g., API server, memory system, quantum module)
- **Full paths** of source file(s) related to the vulnerability
- **Location** of the affected source code (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept** or exploit code (if possible)
- **Impact** of the vulnerability (what an attacker could achieve)

### Response Timeline

| Phase | Timeframe |
|-------|-----------|
| Acknowledgment | Within 48 hours |
| Initial Assessment | Within 1 week |
| Status Update | Every 7 days |
| Resolution Target | 90 days (critical: 30 days) |

### What to Expect

After you submit a vulnerability report:

1. **Acknowledgment**: We'll confirm receipt within 48 hours
2. **Triage**: We'll assess the severity and validity
3. **Investigation**: We'll investigate and develop a fix
4. **Coordination**: We'll work with you on disclosure timing
5. **Fix & Release**: We'll release a patch and publish an advisory
6. **Credit**: We'll acknowledge your contribution (unless you prefer anonymity)

## Security Measures

### Code Security

- All code is scanned with CodeQL static analysis
- Dependencies are monitored with Dependabot
- Python security checks (Bandit, Safety) run in CI
- Type checking with mypy strict mode

### Infrastructure Security

- No credentials are stored in the repository
- Secrets are managed via GitHub Secrets
- GPG-signed commits are encouraged
- Branch protection rules enforce code review

### Dependency Security

- Dependencies are regularly audited
- License compliance is tracked in `LICENSES/LICENSE_TRACKER.md`
- Known vulnerable versions are patched promptly

## Security Best Practices for Contributors

1. **Never commit secrets** - Use environment variables
2. **Validate input** - Especially for API endpoints
3. **Use parameterized queries** - Never construct SQL/commands from user input
4. **Follow principle of least privilege** - Request minimal permissions
5. **Keep dependencies updated** - Run `uv lock --upgrade` regularly

## Scope

This security policy applies to:

- The main CogSynDelta codebase
- Official CogSynDelta packages
- CogSynDelta API servers
- Associated documentation

### Out of Scope

- Third-party dependencies (report to upstream maintainers)
- User-deployed instances (unless configuration vulnerability)
- Social engineering attacks
- Denial of service attacks

## Security Contacts

- **Primary**: Tyler Zervas (@tzervas)
- **Organization**: Average Joe's Labs (AJL)
- **Email**: security@averagejoes-labs.com

## Attribution

We thank the following researchers for responsible disclosure:

*No vulnerabilities reported yet.*

---

*Last Updated: January 18, 2026*
