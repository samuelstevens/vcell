# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:09:46 2025

@author: alexa
"""

import polars as pl
import numpy as np

uri = "sqlite:///C:/Users/alexa/Documents/VirtualCell/ensembl.sqlite"

df = pl.read_database_uri(query="SELECT * FROM symbol_ensembl_map", uri=uri)

# dataframe scheme
canon_df_schema = {
    "canon_ensembl_id": pl.String,
    "gene": pl.String
}

canon_df = pl.DataFrame(schema=canon_df_schema)
symbol_list = df['symbol_id'].unique()


print(f"finding unique sets and assigning canonical ensembl id for {len(symbol_list)} gene symbols")
for symbol in symbol_list:
    #if gene symbol is already included, skip
    if symbol in canon_df["gene"]:
        continue
    
    counter = 0
    is_symbol = True
    tmp_gene_list = np.array([symbol], dtype=object)
    gene_list = np.array([], dtype=object)
    
    #if symbol is actually ensembl_id, search for associated ensembl ids, then run first iteration of while loop on ensembl_id list WITH symbol
    if "ENSG" in symbol:
        tmp_gene_list2 = np.unique(np.array(df.filter(pl.col("symbol_id") == symbol).select("ensembl_gene_id")["ensembl_gene_id"]))
        tmp_gene_list = np.concatenate((tmp_gene_list, tmp_gene_list2))
        is_symbol = False
    
    #get gene symbol from ensembl id and vice versa until no new elements are generated (reached the complete gene set associated with a given symbol)
    while np.all(np.in1d(tmp_gene_list, gene_list)) == False:
        new_gene_list = np.setdiff1d(tmp_gene_list, gene_list)
        gene_list = np.unique(np.concatenate((gene_list, new_gene_list)))
        
        if is_symbol:
            tmp_gene_list = np.array([], dtype=object)
            for sym in new_gene_list:
                #print(f"symbol: {sym}")
                tmp_gene_list2 = np.unique(np.array(df.filter(pl.col("symbol_id") == sym).select("ensembl_gene_id")["ensembl_gene_id"]))
                tmp_gene_list = np.concatenate((tmp_gene_list, tmp_gene_list2))
                    
        else:
            tmp_gene_list = np.array([], dtype=object)
            for ens in new_gene_list:
                #print(f"ensembl id: {ens}")
                tmp_gene_list2 = np.unique(np.array(df.filter(pl.col("ensembl_gene_id") == ens).select("symbol_id")["symbol_id"]))
                tmp_gene_list = np.concatenate((tmp_gene_list, tmp_gene_list2))
        is_symbol = not is_symbol  # toggle between gene symbol and ensembl id
        counter += 1
    
    #print(f"Iterations to complete set for {symbol}: {counter}")            
    canon_ens_id = max([gene_id for gene_id in gene_list if "ENSG" in gene_id])
    tmp_df =  pl.DataFrame({"canon_ensembl_id": canon_ens_id, "gene": np.unique(np.array(gene_list))}).unique()
    canon_df = pl.concat([canon_df, tmp_df], how="vertical")
    
    
    #print(f"canon ensembl id: {canon_ens_id}")
    #print(f"gene list: {set(gene_list)}")
    print(".")
    

canon_df = canon_df.unique()
print(canon_df)

#check if there are any genes corresponding to multiple canon_ensembl_ids
counts_canon = canon_df["gene"].value_counts(name="n_gene_occurrences")
counts_of_counts_canon = counts_canon["n_gene_occurrences"].value_counts().sort("n_gene_occurrences")

print(counts_of_counts_canon)

canon_df_wcounts = counts_canon.join(canon_df, on="gene").sort("n_gene_occurrences")
canon_df_wcounts.write_csv("C:/Users/alexa/Documents/VirtualCell/data_inves/canon_ensembl_map.csv")
