"""
Warehouse package for OLYMPIA.
Provides query and access utilities for the SQLite Star Schema.
"""

from .warehouse import (
    get_db_connection,
    get_table_counts,
    get_schema_info,
    get_denormalized_medals,
    execute_query
)

__all__ = [
    'get_db_connection',
    'get_table_counts',
    'get_schema_info',
    'get_denormalized_medals',
    'execute_query'
]
