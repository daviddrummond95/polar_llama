# Cargo Configuration

This directory contains cargo-specific configuration files for the polar-llama project.

## Files

### `audit.toml`

Configuration for `cargo-audit` security vulnerability scanning.

This file documents security advisories that are temporarily accepted with detailed justification. Each ignored advisory includes:
- The reason for acceptance
- Risk assessment
- Mitigation plan
- Review date

**Current exceptions:**
- `RUSTSEC-2025-0020` (pyo3 buffer overflow): Blocked by pyo3-polars dependency compatibility. Risk assessed as LOW. Will be fixed when polars 0.52+ stabilizes.

See the file comments for full details on each exception.
