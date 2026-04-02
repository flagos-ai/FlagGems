---
title: 性能数据库后端
weight: 30
---

<!--
# Performance Database Backends

*FlagGems* implements a `LibCache` class for persisting performance benchmark data
into a database. The `LibCache` interacts with the database backend through
`sqlalchemy`, a generic database abstraction library.

The connection to the backend database can be specified using the environment variable
`FLAGGEMS_DB_URL`.
-->
# 性能数据库后端

*FlagGems* 实现了一个 `LibCache` 类，用来将性能基准测试数据写入数据库中长期保存。
`LibCache` 通过 `sqlalchemy` （一个通用的数据库抽象库）来与具体的数据库后端交互。

与后端数据库的连接信息可以使用环境变量 `FLAGGEMS_DB_URL` 来设置。
本文档以 *SQLite3* 和 *PostgreSQL* 为例讲解数据库后端的配置，
如果需要，你也可以使用其他数据库管理系统来进行类似的配置。

<!--
## 1. SQLite3

The default backend is *SQLite3*, an embedded database.
Please make sure the library `sqlite3` has been installed before running any benchmarks.
If you want to store the database file in a specific place,
you can set the environment variable as shown below:

-->
## 1. SQLite3

默认的数据库后端是 *SQLite3*，一种嵌入式数据库。
在运行性能基准测试之前，请确保 `sqlite3` 库已经被安装。
如果你希望将数据库文件保存在一个特定位置，可以按如下方式设置环境变量：

```shell
export FLAGGEMS_DB_URL=sqlite:///${DB_PATH}
```

<!--
If you don't want to maintain or reuse the cached data in your current environment,
you can choose to use SQLite as an in-memory database. This can be achieved
by setting the environment variable `FLAGGEMS_DB_URL` as shown below:
-->
如果你不想在当前的环境中维护或者复用已经缓存的性能数据，你也可以将 SQLite
作为一种内存数据库来使用。你可以通过按下例所给的方式来设置 `FLAGGEMS_DB_URL`
环境变量来实现这一点：

```shell
export FLAGGEMS_DB_URL=sqlite:///:memory:
```

<!--
The performance data would be cached in memory during the benchmark session.
When the session ends, the database would be lost.
-->
性能数据会在基准测试进行期间缓存在内存中。测试结束之后，数据库会丢失。

<!--
## 2. PostgreSQL

As an embedded database, *SQLite3* doesn't support multi-writers at the same time.
However, having multiple writers writing performace data is a common use case.
For this reason, we also support using *PostgreSQL* as the backend database.
Different from the embedded database, *PostgreSQL* requires an additional setup
step before being used. You could refer to the
[PostgreSQL document](https://documentation.ubuntu.com/server/how-to/databases/install-postgresql/)
for setup instructions. Note that you have to install the `psycopg` Python
package before using *PostgreSQL*.
-->
## 2. PostgreSQL

作为一种嵌入式数据库，*SQLite3* 不支持同一时刻存在多个写入者的使用场景。
但是，同时有多个写者写入性能数据是一种很常见的使用情况。
出于这一原因，我们也支持类似 *PostgreSQL* 这类数据库作为后端数据库。
与嵌入式数据库不同，*PostgreSQL* 在使用之前需要一些额外的安装部署操作。
你可以参阅 [PostgreSQL 文档](https://documentation.ubuntu.com/server/how-to/databases/install-postgresql/)
了解安装指令。需要注意的是，你在使用 *PostgreSQL* 之前必须安装 `psycopg` Python 包。

<!--
With a backend database like *PostgreSQL* in place, you can use it as a remote database
to allow several *FlagGems* instances to connect to it at the same time
and share benchmark results in this way.

After having created your own database, you could use the following environment
variable make the URL available to the *FlagGems* benchmarking framework.
-->
有了类似于 *PostgreSQL* 这类后端数据库，你就可以将其作为一种远程数据库，
支持多个 *FlagGems* 实例同时与它建立连接，共享基准测试数据结果。

创建了自己的数据库之后，你可以使用下面的环境变量将数据库的 URL
告知 *FlagGems* 的基准测试框架。

```shell
export FLAGGEMS_DB_URL=postgresql+psycopg:///${user}:${password}@${host}:${port}/${db}
```

<!--
If the database runs on your local machine, and your account has direct access to it,
you can set the `FLAGGEMS_DB_URL` environment variable as shown below:
-->
如果你是在本地机器上运行 *PosgtreSQL* 数据库实例，并且你的账号具有直接访问数据库的权限，
你可以按下面的方式来设置 `FLAGGEMS_DB_URL` 环境变量：

```bash
export FLAGGEMS_DB_URL=postgresql+psycopg:///${db}
```
