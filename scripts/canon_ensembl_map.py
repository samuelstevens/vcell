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


def assign_canon_id(gene_df: pl.DataFrame, vcc_mgene_df: pl.DataFrame, vcc_dict: dict) -> pl.DataFrame:
    """
    Assign canonical ensembl IDs:
      - If exactly one VCC gene in set: all genes take that ensembl ID.
      - If multiple VCC genes: each VCC gene gets its own ID, and all non-VCC genes
        get the global max ENSG as their canon.
      - If no VCC genes: canon is max ENSG in set (or None).
    
    Returns a dataframe with [gene, canon_ensembl_id, is_multi, n_vcc_in_set].
    """
    # Which genes in this set are VCC?
    vcc_filtered = vcc_mgene_df.filter(
        vcc_mgene_df['ensembl_gene_id'].is_in(gene_df['gene'])
    )
    n_vcc_in_set = vcc_filtered.height

    # Default canon = max ENSG in set
    ens_ids = gene_df.filter(pl.col('gene').str.contains('ENSG'))['gene']
    canon_default = ens_ids.max() if ens_ids.len() > 0 else None

    if n_vcc_in_set == 1:
        # one vcc gene in set → everything mapped to its ensembl_id
        vcc_canon = vcc_filtered['ensembl_gene_id'][0]
        out = gene_df.select([
            pl.col('gene'),
            pl.lit(vcc_canon).alias('canon_ensembl_id')
        ])

    elif n_vcc_in_set > 1:
        # multiple vcc genes → each vcc gene mapped individually, others get max ENSG
        out = (
            gene_df
            .with_columns(
                pl.col('gene')
                .map_elements(lambda g: vcc_dict.get(g, None), return_dtype=pl.Utf8)
                .alias('mapped_id')
            )
            .with_columns(
                pl.when(pl.col('mapped_id').is_not_null())
                .then(pl.col('mapped_id'))
                .otherwise(pl.lit(canon_default))
                .alias('canon_ensembl_id')
            )
            .select(['gene', 'canon_ensembl_id'])
        )

    else:
        # no vcc genes → use max ENSG (or None)
        out = gene_df.select([
            pl.col('gene'),
            pl.lit(canon_default).alias('canon_ensembl_id')
        ])

    # add flags
    out = out.with_columns([
        pl.lit(n_vcc_in_set).alias('n_vcc_in_set').cast(pl.Int32),
        pl.lit(n_vcc_in_set > 1).alias('is_set_multi_vcc').cast(pl.Boolean)
    ])

    return out


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
        
        processed |= set(genes)
        n_genes = len(genes)
        
        tmp_df = pl.DataFrame({
            'gene': genes,
            'set_index': [set_index] * n_genes,
            'total_set_iterations': [n_iters] * n_genes,
            'n_genes': [n_genes] * n_genes,
            'seed_symbol': [symbol] * n_genes,
            'is_vcc_gene': [g in vcc_gene_set for g in genes]
        })
        
        canon_assignments = assign_canon_id(tmp_df, vcc_mgene_df, vcc_dict)
        tmp_df = tmp_df.join(canon_assignments, on='gene', how='left')
        
        if tmp_df['is_set_multi_vcc'][0]:
            multi_set_count += 1
        
        results.append(tmp_df)
        set_index += 1
    
    canon_df = pl.concat(results, how='vertical')
    
    print(f'there were {multi_set_count} sets with multiple vcc genes, which have been singluated')
    canon_df.write_csv('canon_ensembl_map.csv')


if __name__ == '__main__':
    main()