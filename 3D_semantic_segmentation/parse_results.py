import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest, chisquare, cumfreq, friedmanchisquare
from numpy.random import poisson, uniform, normal




fig,ax = plt.subplots(2,4, figsize=(12,12))

files = [
        'semantic_segmentation_results-S3DIS.csv',
        'semantic_segmentation_results-Semantic3D.csv',
        'semantic_segmentation_results-ScanNet.csv',
        'semantic_segmentation_results-SemanticKITTI.csv',
        ]


results = {}

for i,file in enumerate(files):
    dataset = file.split('-')[1].split('.')[0]
    print(dataset)
    r = pd.read_csv(file)
    r['rank'] = r.index+1
    results[dataset] = r
    counts = r[['family', 'miou']].groupby(['family']).count()['miou'].to_numpy()

    print(f'Counts from {dataset} are uniformly distributed with probability:\
 {chisquare(counts, f_exp=[counts.sum()/counts.shape[0] for _ in range(counts.shape[0])])[1]:.4f}'
)

    
    per_family = pd.DataFrame({
        'mean_miou':r[['family', 'miou']].groupby(['family']).mean()['miou'],
        'mean_rank':r[['family', 'rank']].groupby(['family']).mean()['rank'],
        'counts':r[['family', 'method']].groupby(['family']).count()['method'],
        })
    per_family = per_family.reindex(['mlp', 'cnn','gnn'])

    per_family['colors']=[
                '#F6511D',
                '#FFB400',
                '#083D77',
                ]

    print([f"{t[0]} ({t[1]})" for t in zip(per_family.index.to_list(), per_family['counts'].to_list())])
    print(per_family)
    if dataset == 'S3DIS':
        S3DIS_per_family = per_family

    row=0
    col=i
    ax[row,col].bar(
        [f"{t[0]} ({t[1]})" for t in zip(per_family.index.to_list(), per_family['counts'].to_list())],
        per_family['mean_rank'].to_numpy(),
        color=per_family['colors'],
        )
    ax[row,col].set_title(dataset+' mean rank')


    row=1
    col=i
    ax[row,col].bar(
        [f"{t[0]} ({t[1]})" for t in zip(per_family.index.to_list(), per_family['counts'].to_list())],
        per_family['mean_miou'].to_numpy(),
        color=per_family['colors'],
        )
    ax[row,col].set_title(dataset+' mean mIoU')


plt.show()