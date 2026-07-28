---
title: Performance Database Backends
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


# Performance Datadase Backends

*FlagGems* implements a `LibCache` class for persisting performance benchmark data
into a database. The `LibCache` interacts with the database backend through
`sqlalchemy`, a generic database abstraction library.

The connection to the backend database can be specified using the environment variable
`FLAGGEMS_DB_URL`. This document shows the configurations for *SQLite3* and *PostgreSQL*,
but you can experiment with other DBMS in a similar way, if needed.

## 1. SQLite3

The default backend is *SQLite3*, an embedded database.
Please make sure the library `sqlite3` has been installed before running any benchmarks.
If you want to store the database file in a specific place,
you can set the environment variable as shown below:

```shell
export FLAGGEMS_DB_URL=sqlite:///${DB_PATH}
```

If you don't want to maintain or reuse the cached data in your current environment,
you can choose to use SQLLite as an in-memory database. This can be achieved
by setting the environment variable `FLAGGEMS_DB_URL` as shown below:

```shell
export FLAGGEMS_DB_URL=sqlite:///:memory:
```

The performance data would be cached in memory during the benchmark session.
When the session ends, the database would be lost.

## 2. PostgreSQL

As an embedded database, *SQLite3* doesn't support multi-writers at the same time.
However, having multiple writers writing performace data is a common use case.
For this reason, we also support using *PostgreSQL* as the backend database.
Different from the embedded database, *PostgreSQL* requires an additional setup
step before being used. You could refer to the
[PostgreSQL document](https://documentation.ubuntu.com/server/how-to/databases/install-postgresql/)
for setup instructions. Note that you have to install the `psycopg` Python
package before using *PostgreSQL*.

With a backend database like *PostgreSQL* in place, you can use it as a remote database
to allow several *FlagGems* instances to connect to it at the same time
and share benchmark results in this way.

After having created your own database, you could use the following environment
variable make the URL available to the *FlagGems* benchmarking framework.

```shell
export FLAGGEMS_DB_URL=postgresql+psycopg:///${user}:${password}@${host}:${port}/${db}
```

If the database runs on your local machine, and your account has direct access to it,
you can set the `FLAGGEMS_DB_URL` environment variable as shown below:

```bash
export FLAGGEMS_DB_URL=postgresql+psycopg:///${db}
```
