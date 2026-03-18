"""Load extra_edges.csv into PostgreSQL edges table."""
import psycopg2
import csv
import sys

def main():
    conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_v16', user='postgres', password='bench')
    conn.autocommit = True
    cur = conn.cursor()

    edges_path = 'test16/imdb_schema_v16/overnight_20260306_221435/extra_edges.csv'

    # Check columns
    with open(edges_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        row1 = next(reader)
        print(f'Edge columns: {header}')
        print(f'Sample row: {row1}')

    # Create edges table
    cur.execute('DROP TABLE IF EXISTS edges CASCADE')
    col_defs = ', '.join([f'"{c}" TEXT' for c in header])
    cur.execute(f'CREATE TABLE edges ({col_defs})')
    print('Created edges table')

    # Load data
    print('Loading extra_edges.csv (7.3M rows, this may take a minute)...', flush=True)
    col_list = ', '.join([f'"{c}"' for c in header])
    with open(edges_path, 'r', encoding='utf-8', errors='replace') as f:
        cur.copy_expert(f"COPY edges ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')", f)

    cur.execute('SELECT COUNT(*) FROM edges')
    n = cur.fetchone()[0]
    print(f'Loaded edges: {n:,} rows')

    # Cast numeric columns
    print('Converting column types...', flush=True)
    for col in ['src_id', 'dst_id']:
        cur.execute(f'ALTER TABLE edges ALTER COLUMN "{col}" TYPE INT USING "{col}"::INT')
    for col in ['weight']:
        cur.execute(f'ALTER TABLE edges ALTER COLUMN "{col}" TYPE FLOAT USING NULLIF("{col}",\'\')::FLOAT')
    for col in ['valid_from', 'valid_to']:
        cur.execute(f'ALTER TABLE edges ALTER COLUMN "{col}" TYPE FLOAT USING NULLIF("{col}",\'\')::FLOAT')

    # Create indexes
    print('Creating indexes...', flush=True)
    cur.execute('CREATE INDEX idx_edges_src ON edges(src_id)')
    cur.execute('CREATE INDEX idx_edges_dst ON edges(dst_id)')
    cur.execute('CREATE INDEX idx_edges_type ON edges(edge_type)')
    cur.execute('CREATE INDEX idx_edges_src_dst ON edges(src_id, dst_id)')

    cur.execute('ANALYZE edges')
    print('Done loading edges!')

    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
