---
title: 性能数据库后端
weight: 30
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

# 性能数据库后端配置

*FlagGems* 实现了 `LibCache` 数据持久化模块，支持将历史性能评测指标写入关系型数据库，以便于不同提交版本、不同硬件架构之间的横向性能对比与回归追踪。`LibCache` 底层基于 `SQLAlchemy` ORM 框架，可透明适配多种主流数据库后端。

连接目标数据库的连接串通过环境变量 `FLAGGEMS_DB_URL` 进行配置。本文以常用的 **SQLite3**（本地嵌入式）与 **PostgreSQL**（网络集中式）为例进行说明。

## 1. SQLite3（本地默认后端）

默认情况下，基准测试框架采用嵌入式轻量数据库 *SQLite3*。

### 1.1 文件持久化存储

若希望将测试结果持久化保存到本地特定磁盘路径，可声明文件数据库路径：

```shell
export FLAGGEMS_DB_URL=sqlite:///${DB_PATH}
```

### 1.2 纯内存数据库

若无需在磁盘保留历史数据，仅用于单次单会话临时存储，可启用内存模式（会话结束即自动释放）：

```shell
export FLAGGEMS_DB_URL=sqlite:///:memory:
```

## 2. PostgreSQL（集中式与多节点并发写入）

由于 *SQLite3* 基于文件级锁机制，不支持多并发客户端同时执行写入；而在持续集成（CI/CD）或多机分布式性能评测集群中，通常需要多个节点将基准测试数据集中上报汇总。此时推荐使用 **PostgreSQL**。

### 2.1 环境准备

请预先搭建好 PostgreSQL 服务实例，并在 Python 环境中安装对应的数据库驱动包：

```shell
pip install psycopg
```

### 2.2 配置连接串

通过环境变量配置网络连接凭证即可接入远程集中数据库：

```shell
export FLAGGEMS_DB_URL=postgresql+psycopg://${user}:${password}@${host}:${port}/${db}
```

若在本地运行免密或本地套接字认证的 PostgreSQL 实例，连接串可简写为：

```shell
export FLAGGEMS_DB_URL=postgresql+psycopg:///${db}
```
