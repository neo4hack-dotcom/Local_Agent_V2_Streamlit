from __future__ import annotations

import re
from typing import Any

import requests

from .models import DatabaseProfile

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|merge)\b",
    re.IGNORECASE,
)


def _validate_read_only_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    if not cleaned:
        raise ValueError("The SQL query is empty.")
    if not cleaned.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    if _WRITE_KEYWORDS.search(cleaned):
        raise ValueError("The query contains disallowed write operations.")
    return cleaned


def _apply_limit(sql: str, limit: int) -> str:
    if re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql}\nLIMIT {limit}"


class BaseConnector:
    def __init__(self, profile: DatabaseProfile) -> None:
        self.profile = profile

    def test_connection(self) -> dict[str, Any]:
        raise NotImplementedError

    def schema_snapshot(self, allowed_tables: list[str] | None = None) -> str:
        raise NotImplementedError

    def run_query(self, sql: str, limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def run_statement(self, sql: str, limit: int = 200) -> dict[str, Any]:
        raise NotImplementedError


class ClickHouseConnector(BaseConnector):
    def _client(self):
        try:
            import clickhouse_connect
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The clickhouse-connect package is required for ClickHouse."
            ) from exc

        if not self.profile.host or not self.profile.username:
            raise ValueError("host and username are required for ClickHouse.")

        return clickhouse_connect.get_client(
            host=self.profile.host,
            port=self.profile.port or 8123,
            username=self.profile.username,
            password=self.profile.password or "",
            database=self.profile.database or "default",
            secure=self.profile.secure,
            **self.profile.options,
        )

    def test_connection(self) -> dict[str, Any]:
        client = self._client()
        result = client.query("SELECT 1 AS ok")
        return {
            "status": "ok",
            "engine": "clickhouse",
            "message": "Connection successful.",
            "sample": result.result_rows[0][0] if result.result_rows else None,
        }

    def schema_snapshot(self, allowed_tables: list[str] | None = None) -> str:
        client = self._client()
        database = self.profile.database or "default"
        result = client.query(
            """
            SELECT table, name, type
            FROM system.columns
            WHERE database = %(database)s
            ORDER BY table, position
            LIMIT 800
            """,
            parameters={"database": database},
        )

        entries: list[str] = []
        allow = set(allowed_tables or [])
        for row in result.result_rows:
            table, column, col_type = row
            if allow and table not in allow:
                continue
            entries.append(f"{table}.{column} ({col_type})")

        if not entries:
            return "No schema available."
        return "\n".join(entries)

    def run_query(self, sql: str, limit: int) -> list[dict[str, Any]]:
        safe_sql = _apply_limit(_validate_read_only_sql(sql), limit)
        client = self._client()
        result = client.query(safe_sql)
        columns = result.column_names
        return [dict(zip(columns, row, strict=True)) for row in result.result_rows]

    def run_statement(self, sql: str, limit: int = 200) -> dict[str, Any]:
        cleaned = sql.strip().rstrip(";")
        if not cleaned:
            raise ValueError("The SQL statement is empty.")

        client = self._client()
        if cleaned.lower().startswith("select"):
            result = client.query(_apply_limit(cleaned, max(1, limit)))
            columns = result.column_names
            rows = [dict(zip(columns, row, strict=True)) for row in result.result_rows]
            return {
                "statement_type": "select",
                "row_count": len(rows),
                "rows": rows,
                "message": f"SELECT executed successfully ({len(rows)} row(s)).",
            }

        command_result = client.command(cleaned)
        message = str(command_result).strip() if command_result is not None else ""
        return {
            "statement_type": "command",
            "row_count": None,
            "rows": [],
            "message": message or "Statement executed successfully.",
        }


class OracleConnector(BaseConnector):
    def _connection(self):
        try:
            import oracledb
        except ModuleNotFoundError as exc:
            raise RuntimeError("The oracledb package is required for Oracle.") from exc

        username = self.profile.username
        password = self.profile.password
        dsn = self.profile.dsn

        if not username or not password:
            raise ValueError("username and password are required for Oracle.")

        if not dsn:
            if not self.profile.host or not self.profile.port or not self.profile.database:
                raise ValueError(
                    "For Oracle, provide either dsn, or host/port/database."
                )
            dsn = f"{self.profile.host}:{self.profile.port}/{self.profile.database}"

        return oracledb.connect(user=username, password=password, dsn=dsn)

    def test_connection(self) -> dict[str, Any]:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 AS ok FROM dual")
                row = cursor.fetchone()
        return {
            "status": "ok",
            "engine": "oracle",
            "message": "Connection successful.",
            "sample": row[0] if row else None,
        }

    def schema_snapshot(self, allowed_tables: list[str] | None = None) -> str:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                owner = (self.profile.options.get("schema") or self.profile.username or "").upper()
                cursor.execute(
                    """
                    SELECT table_name, column_name, data_type
                    FROM all_tab_columns
                    WHERE owner = :owner
                    ORDER BY table_name, column_id
                    FETCH FIRST 800 ROWS ONLY
                    """,
                    {"owner": owner},
                )
                rows = cursor.fetchall()

        allow = {table.upper() for table in (allowed_tables or [])}
        entries: list[str] = []
        for table_name, column_name, data_type in rows:
            if allow and table_name.upper() not in allow:
                continue
            entries.append(f"{table_name}.{column_name} ({data_type})")

        if not entries:
            return "No schema available."
        return "\n".join(entries)

    def run_query(self, sql: str, limit: int) -> list[dict[str, Any]]:
        safe_sql = _validate_read_only_sql(sql)
        wrapped_sql = f"SELECT * FROM ({safe_sql}) WHERE ROWNUM <= :limit_rows"

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(wrapped_sql, {"limit_rows": limit})
                columns = [col[0] for col in cursor.description or []]
                rows = cursor.fetchall()

        return [dict(zip(columns, row, strict=True)) for row in rows]

    def run_statement(self, sql: str, limit: int = 200) -> dict[str, Any]:
        raise ValueError(
            "Statement execution is not supported for Oracle connector in this mode."
        )


class ElasticsearchConnector(BaseConnector):
    def _base_url(self) -> str:
        host = (self.profile.host or "").strip()
        if not host:
            raise ValueError("host is required for Elasticsearch.")

        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/")

        scheme = "https" if self.profile.secure else "http"
        port = self.profile.port or 9200
        return f"{scheme}://{host}:{port}"

    def _request_kwargs(self) -> dict[str, Any]:
        options = self.profile.options or {}
        headers: dict[str, str] = {"Content-Type": "application/json"}

        custom_headers = options.get("headers")
        if isinstance(custom_headers, dict):
            for key, value in custom_headers.items():
                headers[str(key)] = str(value)

        api_key = str(options.get("api_key", "")).strip()
        if api_key:
            headers["Authorization"] = f"ApiKey {api_key}"

        auth = None
        username = (self.profile.username or "").strip()
        password = self.profile.password or ""
        if username:
            auth = (username, password)

        verify_ssl = bool(options.get("verify_ssl", True))
        timeout_seconds = int(options.get("timeout_seconds", 10))
        return {
            "headers": headers,
            "auth": auth,
            "verify": verify_ssl,
            "timeout": timeout_seconds,
        }

    def test_connection(self) -> dict[str, Any]:
        url = f"{self._base_url()}/_cluster/health"
        response = requests.get(url, **self._request_kwargs())
        response.raise_for_status()
        body = response.json()
        return {
            "status": "ok",
            "engine": "elasticsearch",
            "message": "Connection successful.",
            "cluster_name": body.get("cluster_name"),
            "cluster_status": body.get("status"),
        }

    def schema_snapshot(self, allowed_tables: list[str] | None = None) -> str:
        return (
            "Elasticsearch does not expose relational SQL schema. "
            "Use the Elasticsearch Retriever agent instead."
        )

    def run_query(self, sql: str, limit: int) -> list[dict[str, Any]]:
        raise ValueError(
            "SQL execution is not supported on Elasticsearch profiles. "
            "Use a ClickHouse/Oracle profile for SQL agents."
        )

    def run_statement(self, sql: str, limit: int = 200) -> dict[str, Any]:
        raise ValueError(
            "Statement execution is not supported on Elasticsearch profiles."
        )


def connector_for(profile: DatabaseProfile) -> BaseConnector:
    if profile.engine == "clickhouse":
        return ClickHouseConnector(profile)
    if profile.engine == "oracle":
        return OracleConnector(profile)
    if profile.engine == "elasticsearch":
        return ElasticsearchConnector(profile)
    raise ValueError(f"Unsupported engine: {profile.engine}")
