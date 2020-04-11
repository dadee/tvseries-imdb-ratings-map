from pandas import DataFrame, np

df_episodes = DataFrame({'season':[1,1,1,2,2,2],'episode':[1,2,3,1,2,3],'rating':[3,4,5,6,7,8]})
df_episodes.season[3] = np.nan
print(df_episodes)
df_episodes.dropna(inplace=True)
print(df_episodes)