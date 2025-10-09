PRAGMA journal_mode = WAL;    -- Concurrent reads/writes
PRAGMA synchronous = NORMAL;  -- Good balance speed/safety
PRAGMA foreign_keys = ON;     -- Enforce FK constraints
PRAGMA busy_timeout = 30000;  -- Wait up to 30s before throwing timeout errors
PRAGMA strict = ON;           -- Enforce strict type checking (SQLite ≥ 3.37)
PRAGMA encoding = 'UTF-8';    -- Consistent text encoding


-- Tracks h5ad/h5mu datasets we've processed for gene canonicalization.
-- Purpose: Prevent reprocessing the same dataset and track which gene ID column was used.
-- Key columns: dataset_id (auto-generated unique ID), name (file path)
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,  -- Full path to the h5ad/h5mu file
    gene_id_col TEXT,  -- Column in adata.var containing Ensembl IDs
) STRICT;


-- Master list of all Ensembl gene IDs we've encountered.
-- Purpose: Central repository of valid Ensembl IDs to ensure referential integrity.
-- Key column: ensembl_gene_id (includes version suffix like .14)
CREATE TABLE IF NOT EXISTS ensembl_genes (
    ensembl_gene_id TEXT PRIMARY KEY,  -- e.g., "ENSG00000187634.14" (include suffix)
    name TEXT  -- Gene description from Ensembl
) STRICT;


-- All unique gene symbols (like "TP53", "BRCA1") we've seen across all datasets.
-- Purpose: Track unique gene symbols across all datasets.
-- Key column: symbol_id (the gene symbol)
CREATE TABLE IF NOT EXISTS gene_symbols (
    symbol_id TEXT PRIMARY KEY  -- Gene symbol like "TP53"
) STRICT;


-- Many-to-many mapping between symbols and Ensembl IDs (one symbol can map to multiple IDs).
-- Purpose: Handle ambiguous gene symbols that map to multiple Ensembl IDs.
-- Key: Composite of (symbol_id, ensembl_gene_id, source) to track where mapping came from
CREATE TABLE IF NOT EXISTS symbol_ensembl_map (
    symbol_id        TEXT NOT NULL REFERENCES gene_symbols(symbol_id) ON DELETE CASCADE,
    ensembl_gene_id  TEXT NOT NULL REFERENCES ensembl_genes(ensembl_gene_id) ON DELETE CASCADE,
    source           TEXT NOT NULL,  -- e.g., "ensembl_xrefs", "mygene", "gtf"
    PRIMARY KEY (symbol_id, ensembl_gene_id, source)
) STRICT;


-- Links gene symbols to the datasets they appear in.
-- Purpose: Track which symbols are in which datasets and preserve original gene IDs.
-- Key: Composite of (dataset_id, symbol_id) - each symbol appears once per dataset
CREATE TABLE IF NOT EXISTS dataset_symbols (
    symbol_id TEXT NOT NULL REFERENCES gene_symbols(symbol_id) ON DELETE CASCADE,
    dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    original_gene_id TEXT REFERENCES ensembl_genes(ensembl_gene_id),  -- Original ID from dataset
    PRIMARY KEY (dataset_id, symbol_id)
) STRICT;
