#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sys
import csv

games = []
with open(sys.argv[1], 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        if(i == 0):
            i += 1
            continue
        games += [[int(row[2]), row[3], row[4], int(row[6]), int(row[7]), int(row[8][1])]]


# In[18]:



init_season = games[0][0]
season = 0
gamesPerSeason = [[]]
for g in games:
    if(g[0] == init_season and g[5] == 1):
        gamesPerSeason[season] += [g]
    elif(g[5] == 1):
        season += 1
        init_season = g[0]
        gamesPerSeason += [[]]
        gamesPerSeason[season] += [g]

    


# In[19]:


for season in gamesPerSeason:
    with open(str(season[0][0]) + '.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for g in season:
            spamwriter.writerow([g[1], g[2], g[3], g[4]])


# In[ ]:




