import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

color_lookup = {
                'mlp':'#F6511D',
                'cnn':'#FFB400',
                'gnn':'#083D77'
                }



assign_color = lambda family: color_lookup[family]


# colors = [assign_color(family) for family in families]





s3dis_results = pd.read_csv('semantic_segmentation_results-S3DIS.csv')
semantic3d_results = pd.read_csv('semantic_segmentation_results-Semantic3D.csv')


# s3dis_grouped = s3dis_results[['family', 'S3DIS mIoU']].groupby(['family']).mean()
# semantic3d_grouped = semantic3d_results[['family', 'Semantic3D mIoU']].groupby(['family']).mean()
print(s3dis_results[['family', 'S3DIS mIoU']].groupby(['family']).count())
print(semantic3d_results[['family', 'Semantic3D mIoU']].groupby(['family']).count())



# semantic3d_grouped.plot.barh()
# s3dis_grouped.plot.barh()

# plt.show()
from scipy.stats import kstest
from numpy.random import poisson, uniform

# data = poisson(5, 5)
data = uniform(low=0.0, high=1.0, size=500)
print(data)
print(kstest(data, 'uniform'))
print(kstest(data, 'uniform')[1])













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
