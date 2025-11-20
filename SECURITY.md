# Security Policy

## Supported Versions

We actively support the following versions of Polar Llama with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

We take the security of Polar Llama seriously. If you discover a security vulnerability, please follow these steps:

### Where to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities by:

1. **Email**: Send a detailed report to the maintainer at [david.drummond.ai@gmail.com]
   - Include "SECURITY" in the subject line
   - Provide a clear description of the vulnerability
   - Include steps to reproduce the issue
   - Suggest a fix if possible

2. **GitHub Security Advisories**: Use GitHub's private vulnerability reporting feature
   - Navigate to the "Security" tab in the repository
   - Click "Report a vulnerability"
   - Fill out the advisory form

### What to Include

Please include the following information in your report:

- **Type of vulnerability** (e.g., code injection, XSS, authentication bypass)
- **Full paths of source file(s)** related to the vulnerability
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions to reproduce** the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit it
- **Suggested mitigation** (if you have ideas)

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Status Updates**: We will keep you informed of our progress every 5-7 days
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days
- **Disclosure**: We follow coordinated disclosure practices

### Our Commitment

When you report a vulnerability to us, you can expect:

1. **Acknowledgment**: Prompt confirmation that we received your report
2. **Communication**: Regular updates on our progress addressing the issue
3. **Credit**: Public acknowledgment of your responsible disclosure (if desired)
4. **Timeline**: Clear timeline for when the fix will be released

### Security Update Process

Once a vulnerability is confirmed:

1. We will develop a fix in a private repository
2. We will prepare a security advisory
3. We will release a patched version
4. We will publish the security advisory with:
   - Description of the vulnerability
   - Affected versions
   - Fixed version
   - Mitigation steps
   - Credit to the reporter (if authorized)

### Scope

The following are in scope for vulnerability reports:

- **Core Library**:
  - Rust source code (`src/`)
  - Python bindings and expressions
  - API client implementations

- **Dependencies**:
  - Known vulnerabilities in dependencies
  - Dependency confusion attacks

- **Injection Vulnerabilities**:
  - Code injection
  - Command injection
  - JSON/YAML parsing issues

- **Authentication & Authorization**:
  - API key handling
  - Environment variable exposure

- **Data Security**:
  - Sensitive data exposure
  - Information disclosure

### Out of Scope

The following are generally out of scope:

- Vulnerabilities in third-party LLM APIs (OpenAI, Anthropic, etc.)
- Issues requiring physical access to the system
- Social engineering attacks
- Denial of Service (DoS) attacks
- Issues in unsupported versions

### Security Best Practices for Users

When using Polar Llama, please follow these security best practices:

1. **API Keys**: Never hardcode API keys in your code
   - Use environment variables (`.env` file)
   - Keep `.env` files out of version control
   - Rotate API keys regularly

2. **Input Validation**: Validate all user inputs before passing to LLM APIs
   - Sanitize prompts to prevent injection attacks
   - Validate taxonomy structures before use
   - Check response formats match expected schemas

3. **Dependencies**: Keep dependencies up to date
   - Run `cargo update` regularly for Rust dependencies
   - Update Python packages with `pip install --upgrade`
   - Monitor Dependabot alerts

4. **Network Security**: Use HTTPS for all API communications (default in library)
   - Verify SSL certificates are not disabled
   - Use secure network connections

5. **Least Privilege**: Use API keys with minimal required permissions
   - OpenAI: Use project-scoped keys
   - AWS Bedrock: Use IAM roles with least privilege
   - Anthropic: Use workspace-specific keys when available

## Known Security Considerations

### API Key Management

- This library requires API keys to be passed via environment variables
- API keys are never logged or persisted by this library
- Ensure your `.env` file is in `.gitignore`

### LLM Prompt Injection

- Be aware that LLM APIs may be vulnerable to prompt injection
- Validate and sanitize user inputs before including in prompts
- Consider using structured outputs to limit response formats

### Network Communication

- All API requests use HTTPS by default
- Ensure your environment has valid SSL certificates
- The library uses `rustls-tls` for secure connections

## Security Tooling

This repository uses the following security tools:

- **cargo-audit**: Scans Rust dependencies for known vulnerabilities (runs in CI)
- **Dependabot**: Automatically creates PRs for dependency updates
- **GitHub Security Advisories**: Monitors for known vulnerabilities
- **Clippy**: Rust linter that catches common security issues
- **safety**: Python dependency security scanner (runs in CI)

## Security Hall of Fame

We gratefully acknowledge the following individuals who have responsibly disclosed security vulnerabilities:

_No security issues have been reported yet._

## Questions?

If you have questions about this security policy, please open a discussion in the GitHub Discussions tab or contact the maintainers.

---

**Last Updated**: 2025-01-20
