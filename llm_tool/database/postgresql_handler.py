#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
postgresql_handler.py

MAIN OBJECTIVE:
---------------
This script provides comprehensive PostgreSQL database handling for
LLM annotations including connection management, table operations,
and batch updates.

Dependencies:
-------------
- sys
- logging
- typing
- pandas
- sqlalchemy
- json

MAIN FEATURES:
--------------
1) PostgreSQL connection management
2) Table creation and column management
3) Batch insert and update operations
4) JSONB column support for annotations
5) Transaction management
6) Connection pooling

Author:
-------
Antoine Lemor
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager

import pandas as pd

try:
    from sqlalchemy import (
        create_engine,
        text,
        MetaData,
        Table,
        Column,
        Integer,
        String,
        Float,
        DateTime,
        JSON,
        bindparam,
        inspect,
        pool
    )
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.dialects.postgresql import JSONB
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    logging.warning("SQLAlchemy not installed. PostgreSQL support will be limited.")


class PostgreSQLHandler:
    """PostgreSQL database handler for LLM annotations"""

    def __init__(self, connection_params: Dict[str, Any]):
        """
        Initialize PostgreSQL handler.
        
        Parameters
        ----------
        connection_params : dict
            Database connection parameters:
            - host: Database host
            - port: Database port (default 5432)
            - database: Database name
            - user: Username
            - password: Password
            - pool_size: Connection pool size (default 5)
            - max_overflow: Max overflow connections (default 10)
        """
        if not HAS_SQLALCHEMY:
            raise ImportError(
                "SQLAlchemy is required for PostgreSQL support. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            )
        
        self.logger = logging.getLogger(__name__)
        self.connection_params = connection_params
        self.engine = None
        self.metadata = MetaData()
        
        # Initialize connection
        self._init_connection()

    def _init_connection(self):
        """Initialize database connection with pooling"""
        try:
            # Build connection string
            conn_string = (
                f"postgresql://{self.connection_params['user']}:"
                f"{self.connection_params['password']}@"
                f"{self.connection_params.get('host', 'localhost')}:"
                f"{self.connection_params.get('port', 5432)}/"
                f"{self.connection_params['database']}"
            )
            
            # Create engine with connection pooling
            self.engine = create_engine(
                conn_string,
                pool_size=self.connection_params.get('pool_size', 5),
                max_overflow=self.connection_params.get('max_overflow', 10),
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=self.connection_params.get('echo', False)
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.fetchone()[0] == 1:
                    self.logger.info("Successfully connected to PostgreSQL")
                    
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def get_transaction(self):
        """Context manager for database transactions"""
        with self.engine.begin() as conn:
            yield conn

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    def create_annotation_table(
        self,
        table_name: str,
        text_columns: List[str],
        identifier_column: str = 'id',
        drop_if_exists: bool = False
    ):
        """
        Create table for annotations if it doesn't exist.
        
        Parameters
        ----------
        table_name : str
            Name of the table to create
        text_columns : list
            List of text column names
        identifier_column : str
            Name of the identifier column
        drop_if_exists : bool
            Whether to drop table if it exists
        """
        try:
            with self.get_transaction() as conn:
                # Drop table if requested
                if drop_if_exists and self.table_exists(table_name):
                    conn.execute(text(f"DROP TABLE {table_name} CASCADE"))
                    self.logger.info(f"Dropped existing table {table_name}")
                
                # Create table
                create_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {identifier_column} BIGSERIAL PRIMARY KEY
                """
                
                # Add text columns
                for col in text_columns:
                    create_query += f",\n    {col} TEXT"
                
                # Add annotation columns
                create_query += f""",
                    annotation JSONB,
                    annotation_inference_time DOUBLE PRECISION,
                    annotation_raw_per_prompt JSONB,
                    annotation_cleaned_per_prompt JSONB,
                    annotation_status_per_prompt JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                
                conn.execute(text(create_query))
                self.logger.info(f"Created table {table_name}")
                
                # Create indexes for better performance
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_annotation 
                    ON {table_name} USING GIN (annotation)
                """))
                
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_status 
                    ON {table_name} ((annotation IS NOT NULL))
                """))
                
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating table: {e}")
            raise

    def add_column_if_not_exists(
        self,
        table_name: str,
        column_name: str,
        column_type: str = 'JSONB'
    ):
        """
        Add column to table if it doesn't exist.
        
        Parameters
        ----------
        table_name : str
            Table name
        column_name : str
            Column name to add
        column_type : str
            PostgreSQL column type
        """
        try:
            with self.get_transaction() as conn:
                conn.execute(text(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                """))
                self.logger.info(f"Added column {column_name} to {table_name}")
        except SQLAlchemyError as e:
            self.logger.error(f"Error adding column: {e}")
            raise

    def load_data(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None,
        random_sample: bool = False
    ) -> pd.DataFrame:
        """
        Load data from PostgreSQL table.
        
        Parameters
        ----------
        table_name : str
            Table name
        columns : list, optional
            Columns to select (None for all)
        where_clause : str, optional
            WHERE clause for filtering
        limit : int, optional
            Number of rows to return
        random_sample : bool
            Whether to randomly sample rows
        
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        try:
            # Build SELECT query
            if columns:
                columns_str = ", ".join(columns)
            else:
                columns_str = "*"
            
            query = f"SELECT {columns_str} FROM {table_name}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            
            if random_sample and limit:
                query += f" ORDER BY RANDOM() LIMIT {limit}"
            elif limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn)
                self.logger.info(f"Loaded {len(df)} rows from {table_name}")
                return df
                
        except SQLAlchemyError as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def update_annotation(
        self,
        table_name: str,
        identifier_column: str,
        identifier_value: Any,
        annotation_column: str,
        annotation_data: Union[str, dict],
        inference_time: Optional[float] = None
    ):
        """
        Update single annotation in database.
        
        Parameters
        ----------
        table_name : str
            Table name
        identifier_column : str
            Identifier column name
        identifier_value : Any
            Identifier value
        annotation_column : str
            Annotation column name
        annotation_data : str or dict
            Annotation data (JSON string or dict)
        inference_time : float, optional
            Inference time in seconds
        """
        try:
            with self.get_transaction() as conn:
                # Parse annotation if string
                if isinstance(annotation_data, str):
                    annotation_json = json.loads(annotation_data) if annotation_data else None
                else:
                    annotation_json = annotation_data
                
                # Build UPDATE query
                update_query = f"""
                UPDATE {table_name}
                SET {annotation_column} = :annotation,
                    updated_at = CURRENT_TIMESTAMP
                """
                
                params = {
                    'annotation': annotation_json,
                    'identifier': identifier_value
                }
                
                # Add inference time if provided
                if inference_time is not None:
                    update_query += f",\n    {annotation_column}_inference_time = :inference_time"
                    params['inference_time'] = inference_time
                
                update_query += f"\nWHERE {identifier_column} = :identifier"
                
                # Execute update
                query = text(update_query).bindparams(
                    bindparam('annotation', type_=JSON)
                )
                result = conn.execute(query, params)
                
                if result.rowcount > 0:
                    self.logger.debug(f"Updated annotation for {identifier_column}={identifier_value}")
                else:
                    self.logger.warning(f"No row found for {identifier_column}={identifier_value}")
                    
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating annotation: {e}")
            raise

    def batch_update_annotations(
        self,
        table_name: str,
        identifier_column: str,
        updates: List[Dict[str, Any]]
    ):
        """
        Batch update annotations in database.
        
        Parameters
        ----------
        table_name : str
            Table name
        identifier_column : str
            Identifier column name
        updates : list
            List of update dictionaries with keys:
            - identifier: Identifier value
            - annotation: Annotation data
            - inference_time: Optional inference time
        """
        if not updates:
            return
        
        try:
            with self.get_transaction() as conn:
                for update in updates:
                    self.update_annotation(
                        table_name=table_name,
                        identifier_column=identifier_column,
                        identifier_value=update['identifier'],
                        annotation_column='annotation',
                        annotation_data=update['annotation'],
                        inference_time=update.get('inference_time')
                    )
                
                self.logger.info(f"Batch updated {len(updates)} annotations")
                
        except SQLAlchemyError as e:
            self.logger.error(f"Error in batch update: {e}")
            raise

    def insert_data(
        self,
        table_name: str,
        data: Union[pd.DataFrame, List[Dict]]
    ):
        """
        Insert data into PostgreSQL table.
        
        Parameters
        ----------
        table_name : str
            Table name
        data : pd.DataFrame or list of dicts
            Data to insert
        """
        try:
            if isinstance(data, list):
                data = pd.DataFrame(data)
            
            with self.get_connection() as conn:
                data.to_sql(
                    table_name,
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                self.logger.info(f"Inserted {len(data)} rows into {table_name}")
                
        except SQLAlchemyError as e:
            self.logger.error(f"Error inserting data: {e}")
            raise

    def get_unannotated_count(
        self,
        table_name: str,
        annotation_column: str = 'annotation'
    ) -> int:
        """
        Get count of unannotated rows.
        
        Parameters
        ----------
        table_name : str
            Table name
        annotation_column : str
            Annotation column name
        
        Returns
        -------
        int
            Number of unannotated rows
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) 
                    FROM {table_name} 
                    WHERE {annotation_column} IS NULL
                """))
                count = result.fetchone()[0]
                return count
                
        except SQLAlchemyError as e:
            self.logger.error(f"Error counting unannotated rows: {e}")
            raise

    def get_annotation_statistics(self, table_name: str) -> Dict[str, Any]:
        """
        Get annotation statistics for a table.
        
        Parameters
        ----------
        table_name : str
            Table name
        
        Returns
        -------
        dict
            Statistics including counts and averages
        """
        try:
            with self.get_connection() as conn:
                # Get counts
                result = conn.execute(text(f"""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(annotation) as annotated,
                        AVG(annotation_inference_time) as avg_inference_time,
                        MIN(annotation_inference_time) as min_inference_time,
                        MAX(annotation_inference_time) as max_inference_time
                    FROM {table_name}
                """))
                
                row = result.fetchone()
                
                stats = {
                    'total_rows': row[0],
                    'annotated_rows': row[1],
                    'unannotated_rows': row[0] - row[1],
                    'completion_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0,
                    'avg_inference_time': row[2],
                    'min_inference_time': row[3],
                    'max_inference_time': row[4]
                }
                
                return stats
                
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting statistics: {e}")
            raise

    def export_annotations(
        self,
        table_name: str,
        output_path: str,
        format: str = 'csv',
        include_nulls: bool = False
    ):
        """
        Export annotations to file.
        
        Parameters
        ----------
        table_name : str
            Table name
        output_path : str
            Output file path
        format : str
            Output format ('csv', 'json', 'parquet')
        include_nulls : bool
            Whether to include unannotated rows
        """
        try:
            # Load data
            where_clause = None if include_nulls else "annotation IS NOT NULL"
            df = self.load_data(table_name, where_clause=where_clause)
            
            # Export based on format
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', lines=True)
            elif format == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported {len(df)} rows to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting annotations: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Closed PostgreSQL connection")
