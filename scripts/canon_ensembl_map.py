# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:09:46 2025

@author: alexa
"""

import polars as pl
import scanpy as sc


def load_vcc(vcc_path: str) -> tuple[pl.DataFrame, dict, set]:
    """Load VCC genes into Polars DataFrame and mapping dict."""
    adata_vcc = sc.read_h5ad(vcc_path, backed='r')
    vcc_mgene_df = pl.from_pandas(
        adata_vcc.var.reset_index().rename(
            columns={'index': 'symbol_id', 'gene_id': 'ensembl_gene_id'}
        )
    ).with_columns(pl.lit('vcc').alias('source'))
    
    vcc_dict = {
        **dict(zip(vcc_mgene_df['ensembl_gene_id'], vcc_mgene_df['ensembl_gene_id'])),
        **dict(zip(vcc_mgene_df['symbol_id'], vcc_mgene_df['ensembl_gene_id']))
    }
    return vcc_mgene_df, vcc_dict, set(vcc_dict.keys())


def get_linked_genes(symbol: str, df: pl.DataFrame) -> tuple[list[str], int]:
    """Expand a seed symbol into its connected symbol/ensembl set.
    Returns (genes, number of iterations).
    """
    seen, frontier, is_symbol, counter = set(), {symbol}, True, 0
    
    while frontier:
        counter += 1
        if is_symbol:
            matches = df.filter(pl.col('symbol_id').is_in(frontier))['ensembl_gene_id'].to_list()
        else:
            matches = df.filter(pl.col('ensembl_gene_id').is_in(frontier))['symbol_id'].to_list()
        next_frontier = set(matches) - seen
        seen |= next_frontier
        frontier = next_frontier
        is_symbol = not is_symbol
    
    return list(seen | {symbol}), counter


def assign_canon_id(genes: list[str], vcc_mgene_df: pl.DataFrame, vcc_dict: dict) -> tuple[str, bool, int]:
    """Assign canonical ensembl ID for a set of genes.
    Returns (canon_id, is_multi_vcc, n_vcc_in_set).
    """
    ens_ids = [g for g in genes if g.startswith('ENSG')]
    vcc_filtered = vcc_mgene_df.filter(pl.col('ensembl_gene_id').is_in(genes))
    n_vcc_in_set = vcc_filtered.height
    
    # if set contains 1 vcc gene, override ensembl id to match vcc
    if n_vcc_in_set == 1:
        return vcc_filtered['ensembl_gene_id'][0], False, 1
    # if set contains multiple vcc genes, separate them from the set and re-assign canon ensembl id for remaining genes
    elif n_vcc_in_set > 1:
        mapped = [vcc_dict.get(g) for g in genes if g in vcc_dict]
        canon = mapped[0] if mapped else (max(ens_ids) if ens_ids else None)
        return canon, True, n_vcc_in_set
    else:
        canon = max(ens_ids) if ens_ids else None
        return canon, False, 0


def main():
    vcc_path = r'vcc_adata_Training.h5ad'
    uri = 'sqlite:///ensembl.sqlite'
    
    vcc_mgene_df, vcc_dict, vcc_gene_set = load_vcc(vcc_path)
    df = pl.concat([
        pl.read_database_uri("SELECT * FROM symbol_ensembl_map", uri=uri),
        vcc_mgene_df
    ])
    
    results, processed = [], set()
    symbol_list = df['symbol_id'].unique().to_list()
    multi_set_count, set_index = 0, 1
    
    print(f'finding unique sets and assigning canonical ensembl id for {len(symbol_list)} gene symbols')
    
    for i, symbol in enumerate(symbol_list, start=1):
        print(f'finding set for symbol {i} of {len(symbol_list)}, {symbol}')
        
        # skip if gene already included
        if symbol in processed:
            continue
        
        genes, n_iters = get_linked_genes(symbol, df)
        canon_id, is_multi, n_vcc = assign_canon_id(genes, vcc_mgene_df, vcc_dict)
        if is_multi:
            multi_set_count += 1
        
        processed |= set(genes)
        n_genes = len(genes)
        
        tmp_df = pl.DataFrame({
            'gene': genes,
            'canon_ensembl_id': [canon_id] * n_genes,
            'set_index': [set_index] * n_genes,
            'total_set_iterations': [n_iters] * n_genes,
            'n_genes': [n_genes] * n_genes,
            'seed_symbol': [symbol] * n_genes,
            'is_vcc_gene': [g in vcc_gene_set for g in genes],
            'is_set_multi_vcc': [is_multi] * n_genes,
            'n_vcc_in_set': [n_vcc] * n_genes,
        })
        results.append(tmp_df)
        set_index += 1
    
    canon_df = pl.concat(results, how='vertical')
    
    print(f'there were {multi_set_count} sets with multiple vcc genes, which have been singluated')
    canon_df.write_csv('canon_ensembl_map.csv')


if __name__ == '__main__':
    main()