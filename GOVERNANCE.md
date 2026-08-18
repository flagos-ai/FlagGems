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

[中文版](./GOVERNANCE_cn.md)

# FlagGems Governance

## Overview

FlagGems is an open-source project under the [FlagOS](https://flagos.io/) ecosystem, developed and maintained by [BAAI (Beijing Academy of Artificial Intelligence)](https://www.baai.ac.cn/) and the open-source community. The project is licensed under the [Apache License 2.0](./LICENSE).

This document describes how the FlagGems project is governed — the roles, responsibilities, decision-making processes, and policies that guide its development.

Related documents:
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Contributing Guide](./CONTRIBUTING.md)
- [Security Policy](./SECURITY.md)
- [Maintainers](./MAINTAINERS.md)

## Roles & Responsibilities

FlagGems recognizes three community roles. Each role builds on the previous one.

### Contributor

Anyone who contributes to FlagGems — including code, documentation, bug reports, reviews, or community support.

- **Rights**: Submit issues and pull requests; participate in discussions.
- **Requirements**: None. All contributions are welcome.

### Committer

A trusted contributor who has demonstrated sustained, high-quality contributions and a solid understanding of the project.

- **Rights**: All Contributor rights, plus write access to the repository (push to non-protected branches, triage issues, approve PRs).
- **How to become one**: Nominated by an existing Maintainer and approved by a majority of Maintainers (see [Nomination & Approval](#nomination--approval)).
- **Expectations**: Review PRs in their area of expertise; follow the project's coding standards and contribution guidelines.

### Maintainer

A Committer who takes on overall project stewardship — setting technical direction, managing releases, and ensuring project health.

- **Rights**: All Committer rights, plus merge to protected branches, approve releases, vote on governance matters, and nominate new Committers and Maintainers.
- **How to become one**: Nominated by an existing Maintainer and approved by a supermajority (2/3) of Maintainers.
- **Expectations**: Actively participate in project decisions; mentor Contributors and Committers; ensure timely reviews and releases.

The current list of Maintainers is in [MAINTAINERS.md](./MAINTAINERS.md).

## Nomination & Approval

1. An existing Maintainer opens a nomination issue (or email thread) describing the candidate's contributions.
2. Discussion period: **7 calendar days** for other Maintainers to provide feedback.
3. Vote:
   - **Committer nomination**: approved by a **simple majority** (> 50%) of active Maintainers.
   - **Maintainer nomination**: approved by a **supermajority** (≥ 2/3) of active Maintainers.
4. If approved, the nominee is added to the relevant list and granted appropriate access.

## Emeritus & Inactive Policy

Sustained participation is valued, but the project also respects that people's availability changes over time.

- **Inactive threshold**: If a Maintainer or Committer has had no meaningful project activity (code, review, issue triage, governance participation) for **12 consecutive months**, an active Maintainer will reach out to discuss their status.
- **Emeritus status**: If the individual confirms they wish to step back — or does not respond within 30 days — they are moved to **Emeritus** status in [MAINTAINERS.md](./MAINTAINERS.md).
- **Emeritus rights**: Emeritus members retain recognition for their contributions. They lose write access and voting rights but are welcome to participate as Contributors.
- **Returning**: An Emeritus member may return to active status through the standard [Nomination & Approval](#nomination--approval) process.

## Decision Making

### Lazy Consensus

Most day-to-day decisions (merging PRs, triaging issues, minor process changes) follow **lazy consensus**: a proposal is considered approved if no Maintainer objects within a reasonable period (typically 72 hours for non-trivial changes).

### Maintainer Vote

For decisions that cannot be resolved through lazy consensus — including architectural changes, policy changes, and disputed PRs:

1. Any Maintainer may call for a formal vote by opening a GitHub issue labeled `governance/vote`.
2. Voting period: **7 calendar days**.
3. Each active Maintainer has one vote. A **simple majority** (> 50%) is required.
4. In the event of a tie, the **Project Lead** (designated in [MAINTAINERS.md](./MAINTAINERS.md)) casts the deciding vote.

### Transparency

All governance decisions and vote outcomes are recorded in GitHub issues for public reference.

## Release Process

1. A **Release Manager** (a Maintainer, rotating per release) proposes a release by opening a tracking issue with the planned scope and timeline.
2. **Minor / patch releases**: approved through lazy consensus.
3. **Major releases**: require a Maintainer majority vote.
4. The Release Manager is responsible for tagging, changelog, and release artifacts.

## Amendments

Changes to this governance document require a **supermajority (≥ 2/3)** of active Maintainers, with a discussion period of at least **14 calendar days** before the vote.

---

*This document is effective as of its merge date and supersedes any prior informal governance arrangements.*
