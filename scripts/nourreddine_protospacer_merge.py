# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 14:40:10 2025

@author: alexa
"""
import scanpy as sc
import pandas as pd


def gene_String(string):
    gene_list = string.split(sep='|')

    gene_list_edit = [gene.rsplit(sep='-', maxsplit=1)[0] for gene in gene_list]

    gene_string = '|'.join(gene_list_edit)
    
    return gene_string


def main():
    
    file_path = r'C:\Users\alexa\Documents\VirtualCell\data_integration'
    kolf_file = 'nourreddine2025_sampled_dataset.h5ad'
    pert_file = 'protospacer_calls_sample.csv'
    
    adata = sc.read_h5ad(file_path + '\\' + kolf_file)
    
    protospacer_calls = pd.read_csv(file_path + '\\' + pert_file, index_col=0)
    protospacer_calls['batch'] = protospacer_calls['cell_barcode'].apply(lambda x: x.rsplit(sep='-', maxsplit=1)[1])
    protospacer_calls['target_gene'] = protospacer_calls['feature_call'].apply(lambda x: gene_String(x))
    protospacer_calls = protospacer_calls.set_index('cell_barcode')
    
    # might need to subset if n cells in protospacer calls != n cells in adata
    """
    subset_mask = adata.obs_names.isin(protospacer_calls.index)
    
    adata_subset = adata[subset_mask, :].copy()
    protospacer_calls = protospacer_calls[protospacer_calls.index.isin(adata_subset.obs_names)]
    
    adata_subset.obs = adata_subset.obs.merge(
        protospacer_calls,
        left_index=True,
        right_index=True,
        how='left'
    )
    """
    
    adata.obs = adata.obs.merge(
        protospacer_calls,
        left_index=True,
        right_index=True,
        how='left'
    )
    
    print(adata.obs[adata.obs['num_features'] ==1].head(5))
    
    adata.write(file_path + '\\nourreddine2025_KOFL_Pan_Genome_Aggreggate_full.h5ad')

    
if __name__ == "__main__":
	main()