<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Security Policy

## Supported versions

FlagGems provides security fixes for the latest release line. Older release
lines may receive fixes at the maintainers' discretion; users should upgrade
to a supported release before reporting an issue.

| Version | Supported | Notes |
|---|---|---|
| v5.4.x | Yes | Latest release line |
| v5.3.x | Best effort | Upgrade recommended |
| < v5.3 | No | Unsupported |

This table is reviewed at each release. It describes the project's practice,
not a commercial support contract.

## Reporting a vulnerability

If you believe you have found a security vulnerability in FlagGems, please
report it to us through coordinated disclosure.

Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.

Instead, please send an email to **security@flagos.io**.

If you need encrypted communication, say so in your first email and we will
share a PGP key out of band. Do not send sensitive details before we confirm
the channel.

### What to include

- Affected FlagGems version or commit.
- Backend and hardware (for example NVIDIA, Ascend, or CPU) plus driver and
  runtime versions.
- A minimal reproducer, or the exact command that triggers the issue.
- Observed impact, and whether the issue is already public.
- Your preferred credit name, or a request to remain anonymous.

### Our process

1. We aim to acknowledge receipt within **3 business days**.
2. We triage severity and confirm the affected versions within **10 business
   days**.
3. We agree on a disclosure date with the reporter. Our default is **90 days**
   after the report, or earlier if a fix is available and users are protected.
4. We credit reporters in the release notes unless they ask us not to.

These are best-effort targets for an open-source project, not a commercial
support contract or a legal commitment.

### Scope

In scope:

- FlagGems source code, build scripts, and CI configuration in this repository.
- Vulnerabilities that affect users of a supported release.

Out of scope:

- Vulnerabilities in third-party dependencies; please report them upstream.
- Misconfiguration of a user's environment or model.
- Issues that require an unsupported version or a non-default build.
- Social engineering and physical attacks.

### Relationship to the Cyber Resilience Act (CRA)

FlagGems is an upstream open-source component, not a product manufacturer.
Integrators that ship FlagGems in a product remain responsible for their own
obligations under the EU Cyber Resilience Act, including incident reporting.
We will cooperate with integrators and share the information we have, but we
do not act as their notified body or legal adviser.

### Hall of fame

We thank reporters who follow coordinated disclosure. Names are added here
with permission.

---

Please see [PyTorch's Security Policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
for more information and recommendations on how to securely interact with models.
