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


def assign_canon_id(gene_df: pl.DataFrame, vcc_mgene_df: pl.DataFrame, vcc_dict: dict, global_max_ensg: str = None) -> pl.DataFrame:
    """
    Returns a DataFrame with columns:
      - gene
      - canon_ensembl_id
      - n_vcc_in_set
      - is_set_multi_vcc

    Behavior:
      - If exactly one VCC gene in set: all genes -> that VCC's ENSG.
      - If multiple VCC genes:
          * VCC genes -> their own ensembl id
          * Non-VCC genes:
              - If unmapped ENSGs exist: assign max ENSG among them
              - Else: assign their own gene symbol
      - If no VCC genes: canon = max ENSG in the set (or None).
    """
    gene_list = gene_df['gene'].to_list()
    # Which genes in this set are VCC?
    vcc_genes_in_set = vcc_mgene_df.filter(
       vcc_mgene_df['ensembl_gene_id'].is_in(gene_df['gene'])
       )
    n_vcc_in_set = vcc_genes_in_set.height
    is_multi = n_vcc_in_set > 1

    # helper: max ENSG in iterable
    def max_ensg_in(iterable):
        ens = [x for x in iterable if isinstance(x, str) and x.startswith('ENSG')]
        return max(ens) if ens else None

    canon_default = max_ensg_in(gene_list)

    if n_vcc_in_set == 1:
        vcc_canon = vcc_dict[vcc_genes_in_set['ensembl_gene_id'][0]]
        canon_list = [vcc_canon] * len(gene_list)

    elif n_vcc_in_set > 1:
        mapped_ids = [vcc_dict.get(g) for g in gene_list]

        # unmapped ENSGs in this set
        unmapped_ensgs = [g for g, m in zip(gene_list, mapped_ids)
                          if m is None and isinstance(g, str) and g.startswith('ENSG')]

        if unmapped_ensgs:
            per_set_fallback = max_ensg_in(unmapped_ensgs)
            canon_list = [m if m is not None else per_set_fallback for m in mapped_ids]
        else:
            # assign own symbol for non-VCC genes
            canon_list = [m if m is not None else g for g, m in zip(gene_list, mapped_ids)]

    else:
        canon_list = [canon_default] * len(gene_list)

    out = pl.DataFrame({
        'gene': pl.Series(gene_list, dtype=pl.Utf8),
        'canon_ensembl_id': pl.Series(canon_list, dtype=pl.Utf8)
    }).with_columns([
        pl.lit(n_vcc_in_set).alias('n_vcc_in_set').cast(pl.Int32),
        pl.lit(is_multi).alias('is_set_multi_vcc').cast(pl.Boolean)
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