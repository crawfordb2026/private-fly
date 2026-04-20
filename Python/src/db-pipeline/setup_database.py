#!/usr/bin/env python3
"""
Setup script to create the PostgreSQL database and schema for the fly-ML pipeline.

This script:
1. Creates the database if it doesn't exist
2. Executes schema.sql to create all tables and indexes
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config import DB_CONFIG
import os

def create_database():
    """Create the database if it doesn't exist."""
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database='postgres'
    )
    # Set isolation level to autocommit BEFORE using the connection
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG['database'],))
            if not cur.fetchone():
                cur.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
                print(f"Created database: {DB_CONFIG['database']}")
            else:
                print(f"Database {DB_CONFIG['database']} already exists")
    finally:
        conn.close()

def create_schema():
    """Execute schema.sql to reset and recreate schema objects."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute the schema SQL
            cur.execute(schema_sql)
            
            conn.commit()
    print("Schema reset and created successfully!")

if __name__ == "__main__":
    print("Setting up database...")
    create_database()
    create_schema()
    print("Database setup complete!")

