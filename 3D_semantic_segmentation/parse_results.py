import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest, chisquare, cumfreq, friedmanchisquare
from numpy.random import poisson, uniform, normal

color_lookup = {
                'mlp':'#F6511D',
                'cnn':'#FFB400',
                'gnn':'#083D77'
                }



assign_color = lambda family: color_lookup[family]


# colors = [assign_color(family) for family in families]








s3dis_results = pd.read_csv('semantic_segmentation_results-S3DIS.csv')
semantic3d_results = pd.read_csv('semantic_segmentation_results-Semantic3D.csv')

scannet_results = pd.read_csv('semantic_segmentation_results-ScanNet.csv')
semantickitti_results = pd.read_csv('semantic_segmentation_results-SemanticKITTI.csv')






# # s3dis_grouped = s3dis_results[['family', 'S3DIS mIoU']].groupby(['family']).mean()
# # semantic3d_grouped = semantic3d_results[['family', 'Semantic3D mIoU']].groupby(['family']).mean()
# s3dis_results[['family', 'S3DIS mIoU']].groupby(['family']).count()
# # print(semantic3d_results[['family', 'Semantic3D mIoU']].groupby(['family']).count())


# s3dis_counts = s3dis_results[['family', 'miou']].groupby(['family']).count()['miou'].to_numpy()
# semantic3d_counts = semantic3d_results[['family', 'miou']].groupby(['family']).count()['miou'].to_numpy()
# scannet_counts = scannet_results[['family', 'miou']].groupby(['family']).count()['miou'].to_numpy()
# semantickitti_counts = semantickitti_results[['family', 'miou']].groupby(['family']).count()['miou'].to_numpy()





files = [
        'semantic_segmentation_results-S3DIS.csv',
        'semantic_segmentation_results-Semantic3D.csv',
        'semantic_segmentation_results-ScanNet.csv',
        'semantic_segmentation_results-SemanticKITTI.csv',
        ]


results = {}
l = []
for file in files:
    dataset = file.split('-')[1].split('.')[0]
    r = pd.read_csv(file)
    r['rank'] = r.index+1

    # print(r)

    results[dataset] = r
    counts = r[['family', 'miou']].groupby(['family']).count()['miou'].to_numpy()

    # print(f'Counts from {dataset} are uniformly distributed with probability:\
    #         {chisquare(counts, f_exp=[counts.sum()/counts.shape[0] for _ in range(counts.shape[0])])[1]:.4f}'
    #     )




    # _grouped = r[['family', 'miou']].groupby(['family']).mean()
    
    per_family = pd.DataFrame({
        'mean_miou':r[['family', 'miou']].groupby(['family']).mean()['miou'],
        'mean_rank':r[['family', 'rank']].groupby(['family']).mean()['rank'],
        'counts':r[['family', 'method']].groupby(['family']).count()['method'],
        })
    # print(per_family)

    l.append(per_family['mean_rank'].to_numpy())


print(l)

print(
    friedmanchisquare(l)
    )
    # results[]

    # print(r[['family', 'miou']].groupby(['family']).count())
    # _grouped = r[['family', 'method']].groupby(['family']).count()

    # print(_grouped)

    # print(dataset)
    # results[dataset] = s3dis_results[['family', 'miou']].groupby(['family']).count()['miou'].to_numpy()



# for results in [s3dis_results,semantic3d_results,scannet_results,semantickitti_results,]:
#     counts = results[['family', 'miou']].groupby(['family']).count()['miou'].to_numpy()
#     # print(counts)
#     print(
#         chisquare(counts, f_exp=[counts.sum()/counts.shape[0] for _ in range(counts.shape[0])])
#         )





# s3dis_probs = s3dis_counts / s3dis_counts.sum()

# semantic3d_counts = semantic3d_results[['family', 'Semantic3D mIoU']].groupby(['family']).count()['Semantic3D mIoU'].to_numpy()
# semantic3d_probs = semantic3d_counts / semantic3d_counts.sum()








# semantic3d_grouped.plot.barh()
# s3dis_grouped.plot.barh()

# plt.show()


# np.random.seed(0)
# data = uniform(low=0.0, high=19.0, size=3)
# data = uniform(low=0.0, high=1.0, size=3)

# data = normal(0,1, size=100)
# plt.plot(np.sort(s3dis_probs))

# plt.scatter(np.sort(s3dis_probs), np.linspace(0, 1, len(s3dis_probs), endpoint=False), s=5, label='1980-2009')
# plt.step(np.sort(s3dis_probs), np.linspace(0, 1, len(s3dis_probs), endpoint=False))
# plt.step(np.sort(data), np.linspace(0, 1, len(data), endpoint=False))
# plt.show()




# plt.hist(data)
# print(kstest(data, 'norm'))

# plt.show()
# print(kstest(s3dis_probs, 'uniform'))
# print(kstest(semantic3d_probs, 'uniform'))



# print(
#     chisquare(s3dis_counts, f_exp=[s3dis_counts.sum()/s3dis_counts.shape[0] for _ in range(s3dis_counts.shape[0])])
#     )




# print(
#     chisquare(semantic3d_counts, f_exp=[semantic3d_counts.sum()/semantic3d_counts.shape[0] for _ in range(semantic3d_counts.shape[0])])
#     )










# plt.plot(S3DIS mIoU['family', 'me'])


# df_dict = {
#             'method':list(results.keys()),
#             'family':families,
#             'date': publication_dates,
#             'colors': colors
#             }

# for i, column in enumerate(columns[1:]):
#     l = []
#     for key in results.keys():
#         l.append(results[key][i])

#     df_dict[column] = l

# df = pd.DataFrame(df_dict)


# non_zero_df = df.replace(0, np.NaN)

# non_zero_df['mean_performance'] = non_zero_df.mean(axis=1)


# df = non_zero_df
