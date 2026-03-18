import psycopg2
conn = psycopg2.connect(host="localhost", port=5433, dbname="benchmark_v15",
                        user="postgres", password="bench")
conn.autocommit = True
cur = conn.cursor()
# Float-formatted person_id like "20030.0" -> truncate at decimal
cur.execute("""
    ALTER TABLE awards
    ALTER COLUMN person_id TYPE INT
    USING NULLIF(split_part(person_id, '.', 1), '')::INT
""")
print("OK awards.person_id")
cur.execute("ANALYZE awards")
cur.close()
conn.close()
