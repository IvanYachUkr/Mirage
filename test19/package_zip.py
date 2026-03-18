"""
Test 9 -- Package enriched CSVs into a zip file.
Includes full entity details (bios, styles, etc.) in all tables.
"""
import pandas as pd
import zipfile
import os
from pathlib import Path

BASE = Path(__file__).parent

def main():
    # Load all tables
    movie = pd.read_csv(BASE / "movie.csv")
    cast = pd.read_csv(BASE / "cast_info.csv")
    dirs = pd.read_csv(BASE / "movie_directors.csv")
    comps = pd.read_csv(BASE / "movie_companies.csv")
    kw = pd.read_csv(BASE / "movie_keyword.csv")
    persons_enriched = pd.read_csv(BASE / "persons_enriched.csv")
    # V15 FIX: prefer companies_enriched.csv (post-pipeline, has defunct_year mutations)
    # over entities/company.csv (pre-pipeline, all defunct_year=NaN)
    company_csv_path = BASE / "companies_enriched.csv"
    if not company_csv_path.exists():
        company_csv_path = BASE / "entities" / "company.csv"
    company_csv = pd.read_csv(company_csv_path)
    keyword_csv = pd.read_csv(BASE / "entities" / "keyword.csv")
    flat = pd.read_csv(BASE / "movies_flat.csv")
    
    # Enrich cast_info with person details
    person_cols = ["person_id", "name", "nationality", "gender", "bio",
                   "style_tags", "genre_affinity", "career_stage", "roles",
                   "market_fit", "pop_weight", "debut_year", "peak_start",
                   "peak_end", "retirement_year"]
    available = [c for c in person_cols if c in persons_enriched.columns]
    cast_enriched = cast.merge(
        persons_enriched[available],
        on="person_id", how="left"
    )
    
    # Enrich movie_directors with director details
    dirs_enriched = dirs.merge(
        persons_enriched[available].rename(columns={"person_id": "director_id"}),
        on="director_id", how="left"
    )
    
    # Enrich movie_companies with company details
    company_csv["company_id"] = range(1, len(company_csv) + 1)
    comp_cols = ["company_id", "name", "country", "description",
                 "specialty_genres", "tier", "pop_weight"]
    available_c = [c for c in comp_cols if c in company_csv.columns]
    comps_enriched = comps.merge(
        company_csv[available_c],
        on="company_id", how="left"
    )
    
    # Enrich movie_keyword with keyword text
    keyword_csv["keyword_id"] = range(1, len(keyword_csv) + 1)
    kw_enriched = kw.merge(
        keyword_csv[["keyword_id", "keyword", "topic_genre", "pop_weight"]],
        on="keyword_id", how="left"
    )
    
    # Edge graph
    edge_path = BASE / "graph" / "edge_graph.csv"
    if edge_path.exists():
        edges = pd.read_csv(edge_path)
    else:
        edges = pd.DataFrame()
    
    # Package
    zip_path = BASE / "test9_enriched.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        tables = {
            "movie.csv": movie,
            "cast_info_enriched.csv": cast_enriched,
            "movie_directors_enriched.csv": dirs_enriched,
            "movie_companies_enriched.csv": comps_enriched,
            "movie_keyword_enriched.csv": kw_enriched,
            "persons_enriched.csv": persons_enriched,
            "company.csv": company_csv,
            "keyword.csv": keyword_csv,
            "movies_flat.csv": flat,
        }
        if len(edges) > 0:
            tables["edge_graph.csv"] = edges
        
        for name, df in tables.items():
            csv_data = df.to_csv(index=False)
            zf.writestr(name, csv_data)
            print(f"  {name:40s} {len(df):>6,} rows x {len(df.columns):>3} cols")
    
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"\nSaved: {zip_path} ({size_mb:.1f} MB)")
    print(f"Tables: {len(tables)}")

if __name__ == "__main__":
    main()


