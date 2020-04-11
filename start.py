import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Enter here TVSeries ID from IMDb https://www.imdb.com/title/tt0108778/. Each season and episode should have own number
# TODO replace '\\N' in season and episode numbers for unique letters chronologically
series_id = 'tt0773262'  # https://www.imdb.com/title/tt0056758/?ref_=fn_al_tt_1
series_title = 'Dexter'
# tt0096697 Simpsons, tt0108778 Friends, tt2364582 Agents of SHIELD, tt0460681 Supernatural, tt0944947 GoT, tt0068098 MASH, tt0386676 The Office, tt0121955 South Park

# Files from https://datasets.imdbws.com/
episode_path = 'data/title.episode.tsv.gz'
ratings_path = 'data/title.ratings.tsv.gz'

print("Opening Episodes File...")
cols_episodes = ['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber']
df_episodes = pd.read_csv(episode_path, index_col=None, header=1, names=cols_episodes, compression='gzip', sep='\t+',
                          engine='python')
df_episodes = df_episodes[df_episodes['parentTconst'] == series_id]

print("Opening Ratings File...")
cols_ratings = ['tconst', 'averageRating', 'numVotes']
df_ratings = pd.read_csv(ratings_path, header=1, index_col=None, names=cols_ratings, compression='gzip', sep='\t+',
                         engine='python')

# this is faster than merge but gives warnings, look at df.loc #TODO need to care of warnings
df_episodes['averageRating'] = df_episodes.tconst.map(df_ratings.set_index('tconst')['averageRating'].to_dict())
# df = pd.merge(df_selected_episodes, df_ratings, how='left')
df_episodes = df_episodes[['seasonNumber', 'episodeNumber', 'averageRating']]

# replace non-numeric values and then drop rows to make pivot later
df_episodes.seasonNumber.replace(to_replace=['\\N'], value=[np.NaN], inplace=True)
df_episodes.episodeNumber.replace(to_replace=['\\N'], value=[np.NaN], inplace=True)
# df_episodes.averageRating.replace(to_replace=['\\N'], value=[np.nan], inplace=True)
df_episodes.dropna(inplace=True)

df_episodes.seasonNumber = df_episodes.seasonNumber.astype(int)
df_episodes.episodeNumber = df_episodes.episodeNumber.astype(int)
df_episodes.averageRating = df_episodes.averageRating.astype(float)

df_episodes = df_episodes.sort_values(by=['seasonNumber', 'episodeNumber'])
df_episodes = df_episodes.pivot('episodeNumber', 'seasonNumber', 'averageRating')

# Make colored map (table)
norm = plt.Normalize(2, 10)
colours = plt.cm.RdYlGn(norm(df_episodes.values))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

the_table = plt.table(cellText=df_episodes.values, rowLabels=df_episodes.index, colLabels=df_episodes.columns,
                      colWidths=[0.036] * df_episodes.values.shape[1], cellLoc='center', loc='center',
                      cellColours=colours).scale(1, 1.2)

plt.title(series_title)
plt.savefig('res_img/' + series_title + '.png', dpi=300)
plt.show()
