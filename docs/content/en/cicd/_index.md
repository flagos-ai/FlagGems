---
title: CI/CD
weight: 65
bookCollapseSection: true
---

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

# CI/CD

This section documents the GitHub Actions workflows that make up FlagGems'
continuous integration and delivery pipeline: what runs on a pull request,
what runs on a schedule, and what runs on release.

- [Pull request checks](pull-request/) — workflows triggered by opening,
  updating, or commenting on a pull request
- [Scheduled and on-demand testing](scheduled/) — daily/weekly full-suite
  tests, coverage reporting, and on-demand `/test` commands
- [Release and sync](release-sync/) — building/publishing wheels and
  mirroring the repository to other remotes
