import os
import sqlite3
from typing import List, Tuple

DB_CANDIDATES = [
    os.path.abspath("./curioscan_test.db"),
    os.path.abspath("./data/curioscan.db"),
    os.path.abspath("./curioscan.db"),
]

NEW_COLUMNS = [
    ("filename", "TEXT"),
    ("file_name", "TEXT"),
    ("file_size", "INTEGER"),
    ("mime_type", "TEXT"),
    ("progress", "REAL DEFAULT 0.0"),
    ("input_path", "TEXT"),
    ("processing_metadata", "TEXT"),
    ("render_type", "TEXT"),
    ("error_message", "TEXT"),
    ("confidence_score", "REAL"),
    ("webhook_sent", "INTEGER DEFAULT 0"),
    ("webhook_url", "TEXT"),
]


def get_existing_columns(cur: sqlite3.Cursor, table: str) -> List[str]:
    cur.execute(f"PRAGMA table_info('{table}')")
    return [row[1] for row in cur.fetchall()]


def migrate_db(db_path: str) -> Tuple[int, List[str]]:
    if not os.path.exists(db_path):
        return (0, [])

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
        if not cur.fetchone():
            print(f"[SKIP] 'jobs' table not found in {db_path}")
            return (0, [])

        # Check id column type and pk
        cur.execute("PRAGMA table_info('jobs')")
        cols_info = cur.fetchall()
        col_names = [row[1] for row in cols_info]
        existing = set(col_names)
        id_row = next((r for r in cols_info if r[1] == 'id'), None)
        # row: (cid, name, type, notnull, dflt_value, pk)
        id_is_integer_pk = id_row is not None and (id_row[2] or '').upper().strip() == 'INTEGER' and id_row[5] == 1

        added: List[str] = []

        # Add missing simple columns first
        for col, coltype in NEW_COLUMNS:
            if col not in existing:
                ddl = f"ALTER TABLE jobs ADD COLUMN {col} {coltype}"
                print(f"[MIGRATE] {db_path}: {ddl}")
                cur.execute(ddl)
                added.append(col)
            else:
                print(f"[OK] {db_path}: column '{col}' already exists")

        # If id is not proper INTEGER PRIMARY KEY, rebuild table to fix autoincrement
        if not id_is_integer_pk:
            print(f"[REBUILD] Fixing jobs.id to be INTEGER PRIMARY KEY AUTOINCREMENT in {db_path}")
            # Determine columns present in old table to copy
            cur.execute("PRAGMA table_info('jobs')")
            old_cols_info = cur.fetchall()
            old_cols = [c[1] for c in old_cols_info]

            # Create new table with desired schema
            cur.execute("BEGIN TRANSACTION")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE,
                    filename TEXT,
                    file_name TEXT,
                    file_size INTEGER,
                    mime_type TEXT,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    input_path TEXT,
                    processing_metadata TEXT,
                    render_type TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME,
                    processing_duration REAL,
                    error_message TEXT,
                    confidence_score REAL,
                    webhook_sent INTEGER DEFAULT 0,
                    webhook_url TEXT
                )
                """
            )

            # Build column list intersection (excluding id)
            copy_cols = [c for c in [
                'job_id','filename','file_name','file_size','mime_type','status','progress','input_path',
                'processing_metadata','render_type','created_at','updated_at','completed_at','processing_duration',
                'error_message','confidence_score','webhook_sent','webhook_url'
            ] if c in old_cols]
            cols_csv = ",".join(copy_cols)
            if cols_csv:
                cur.execute(f"INSERT INTO jobs_new ({cols_csv}) SELECT {cols_csv} FROM jobs")

            cur.execute("DROP TABLE jobs")
            cur.execute("ALTER TABLE jobs_new RENAME TO jobs")
            cur.execute("COMMIT")

        conn.commit()
        return (len(added), added)
    finally:
        conn.close()


def main():
    total_added = 0
    for db in DB_CANDIDATES:
        added_count, added_cols = migrate_db(db)
        total_added += added_count
    print(f"Done. Total columns added across DBs: {total_added}")


if __name__ == "__main__":
    main()

