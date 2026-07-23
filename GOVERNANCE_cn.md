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

[English](./GOVERNANCE.md)

# FlagGems 项目治理

## 概述

FlagGems 是 [FlagOS](https://flagos.io/) 生态系统下的开源项目，由 [BAAI（北京智源人工智能研究院）](https://www.baai.ac.cn/) 和开源社区共同开发与维护。项目采用 [Apache License 2.0](./LICENSE) 许可证。

本文档描述了 FlagGems 项目的治理方式——包括角色、职责、决策流程和相关政策。

相关文档：
- [行为准则](./CODE_OF_CONDUCT.md)
- [贡献指南](./CONTRIBUTING_cn.md)
- [安全策略](./SECURITY.md)
- [维护者名单](./MAINTAINERS.md)

## 角色与职责

FlagGems 社区设三个角色，每个角色在前一级的基础上递进。

### 贡献者（Contributor）

任何为 FlagGems 做出贡献的人——包括代码、文档、Bug 报告、代码审查或社区支持。

- **权利**：提交 Issue 和 Pull Request；参与讨论。
- **要求**：无。欢迎所有贡献。

### 提交者（Committer）

经过持续、高质量贡献且对项目有深入理解的受信任贡献者。

- **权利**：拥有贡献者的全部权利，加上仓库写入权限（向非保护分支推送代码、管理 Issue、审批 PR）。
- **如何成为提交者**：由现任维护者提名，经维护者多数票通过（参见[提名与审批](#提名与审批)）。
- **期望**：在专长领域审查 PR；遵循项目的编码规范和贡献指南。

### 维护者（Maintainer）

承担项目整体管理职责的提交者——负责技术方向、版本发布管理和项目健康。

- **权利**：拥有提交者的全部权利，加上向保护分支合并代码、审批发布、参与治理投票、提名新的提交者和维护者。
- **如何成为维护者**：由现任维护者提名，经维护者绝对多数（2/3）票通过。
- **期望**：积极参与项目决策；指导贡献者和提交者；确保及时审查和发布。

当前维护者名单见 [MAINTAINERS.md](./MAINTAINERS.md)。

## 提名与审批

1. 现任维护者通过 Issue（或邮件）提出候选人提名，说明其贡献情况。
2. 讨论期：**7 个自然日**，供其他维护者反馈。
3. 投票：
   - **提交者提名**：需**简单多数**（> 50%）的活跃维护者同意。
   - **维护者提名**：需**绝对多数**（≥ 2/3）的活跃维护者同意。
4. 通过后，候选人将被添加到相应名单并获得对应权限。

## 荣休与不活跃政策

项目重视持续参与，同时也尊重每个人的可用时间变化。

- **不活跃阈值**：如果维护者或提交者连续 **12 个月** 无实质性项目活动（代码、审查、Issue 管理、治理参与），活跃维护者将主动联系沟通。
- **荣休状态**：如当事人确认希望退出——或 30 天内未回复——将在 [MAINTAINERS.md](./MAINTAINERS.md) 中标记为**荣休（Emeritus）**状态。
- **荣休权利**：荣休成员的贡献将继续获得认可。其写入权限和投票权将被移除，但仍欢迎以贡献者身份参与。
- **复出**：荣休成员可通过标准的[提名与审批](#提名与审批)流程恢复活跃状态。

## 决策机制

### 懒惰共识（Lazy Consensus）

大多数日常决策（合并 PR、管理 Issue、次要流程变更）采用**懒惰共识**：如在合理期限内（非重大变更通常为 72 小时）没有维护者反对，则视为通过。

### 维护者投票

当懒惰共识无法解决分歧时——包括架构变更、策略变更和有争议的 PR：

1. 任何维护者均可通过创建带有 `governance/vote` 标签的 GitHub Issue 发起正式投票。
2. 投票期：**7 个自然日**。
3. 每位活跃维护者拥有一票。需**简单多数**（> 50%）通过。
4. 如出现平票，由**项目负责人**（见 [MAINTAINERS.md](./MAINTAINERS.md)）投出决定票。

### 透明度

所有治理决策和投票结果均记录在 GitHub Issue 中，供公开查阅。

## 发布流程

1. **发布经理**（由维护者轮流担任）通过创建跟踪 Issue 提出发布计划，说明预计范围和时间线。
2. **小版本 / 补丁版本**：通过懒惰共识批准。
3. **大版本发布**：需维护者多数投票通过。
4. 发布经理负责打标签、编写变更日志和发布制品。

## 修订

对本治理文档的修改需**绝对多数（≥ 2/3）**活跃维护者同意，投票前讨论期至少 **14 个自然日**。

---

*本文档自合并之日起生效，并取代此前任何非正式的治理安排。*
