# LibCache DataBase Backend Setup

`LibCache` provides a dispatch mechanism for the database used to store benchmark results. The dependency library is `sqlalchemy`, so please make sure it's available in the working environment.

The way to select corresponding backend is to set the environment variable `FLAGGEMS_DB_URL`.

## SQLite3

The default backend available is `SQLite3`. Please make sure the relative library `sqlite3` is already installed. If you want to store the db file in an assigned place, please set the environment variable as
```bash
export FLAGGEMS_DB_URL=sqlite:///${db_path}
```
If you want use it as a memory library, or don't want to remain or reuse caches in your current environment, you could set the environment as
```bash
export FLAGGEMS_DB_URL=sqlite:///:memory:
```
The cache would be only stored in the memory, and when the connection is broken, the database would be lost.

## PostgreSQL

As an embed database, `SQLite3` doesn't support multi-writers at the same time, which could be common in some cases. So we also support the users to select `PostgreSQL` as the backend. Different from the embed database, it requires the setup it before using. You could refer to the [document](https://documentation.ubuntu.com/server/how-to/databases/install-postgresql/) on how to set it up. Or you could use a remote database to allow several `FlagGems` instances to connect to it at the same time and share benchmark results in this way.

After creating your own database, you could use the following url to set the environment variable
```bash
export FLAGGEMS_DB_URL=postgresql+psycopg:///${user}:${password}@${host}:${port}/${db}
```
If the database in on your local machine, and your current role could allow you to access it directly, you can use the following environment variable
```bash
export FLAGGEMS_DB_URL=postgresql+psycopg:///${db}
```

Before using it, please make sure the related dependency `psycopg` is installed on your machine.
