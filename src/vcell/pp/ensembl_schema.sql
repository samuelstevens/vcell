PRAGMA journal_mode = WAL;    -- Concurrent reads/writes
PRAGMA synchronous = NORMAL;  -- Good balance speed/safety
PRAGMA foreign_keys = ON;     -- Enforce FK constraints
PRAGMA busy_timeout = 30000;  -- Wait up to 30s before throwing timeout errors
PRAGMA strict = ON;           -- Enforce strict type checking (SQLite ≥ 3.37)
PRAGMA encoding = 'UTF-8';    -- Consistent text encoding


CREATE TABLE IF NOT EXISTS datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    gene_id_col TEXT
) STRICT;


CREATE TABLE IF NOT EXISTS ensembl_genes (
    ensembl_id INTEGER PRIMARY KEY AUTOINCREMENT,
    gene_id TEXT NOT NULL,  -- e.g., "ENSG00000187634" (with no ".14" version suffix)
    version TEXT,
    display_name TEXT
) STRICT;


CREATE TABLE IF NOT EXISTS symbols (
    symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,

    dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    included_ensembl_id INTEGER REFERENCES ensembl(ensembl_id),

    UNIQUE (dataset_id, name)  -- same symbol text unique within a dataset
) STRICT;


CREATE TABLE IF NOT EXISTS symbol_ensembl_map (
    symbol_id        INTEGER NOT NULL REFERENCES symbols(symbol_id) ON DELETE CASCADE,
    ensembl_gene_id  INTEGER NOT NULL REFERENCES ensembl_genes(ensembl_id) ON DELETE CASCADE,
    source           TEXT NOT NULL,  -- e.g., "ensembl_xrefs", "mygene", "gtf"
    PRIMARY KEY (symbol_id, ensembl_gene_id, source)
) STRICT;
