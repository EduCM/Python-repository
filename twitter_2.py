# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:23:32 2017

@author: Hector
"""


import numpy as np
import pandas as pd
import tweepy
from datetime import datetime

#%% Cuentas de acceso
Consumer_Key = 'mfXeGc2wOSaeVlKtDmEY4PbOY'
Consumer_Secret = 'L3rqtmGJnwlEtWqGit7H33Vc0PWurMGwY9lat38pxQpbxVaZWt'
Access_Token = '851652483847999490-9d7Jg5BN3TdY8lDYWcJPyFcErly4N3i'
Access_Token_Secret = 'El3TsIJq6k4mf1DcHxiwUFYaePiqPxiZOf3AF8IXMK2uZ'

#%% Abrir conexion de la API
auth = tweepy.OAuthHandler(Consumer_Key, Consumer_Secret)
auth.set_access_token(Access_Token, Access_Token_Secret)
api = tweepy.API(auth)

#%% Elegir usuario a seguir 
person     = 'econokafka'#
new_tweets = api.user_timeline(screen_name = person, count = 200)

#%%Descargar Muchos tweets
max_tweets = 2000
alltweets = []
alltweets.extend(new_tweets)

n_alltweets = len(alltweets)
n_alltweets_old = 0

oldesttweet = alltweets[-1].id-1

while (len(alltweets) < max_tweets) and (n_alltweets_old < n_alltweets):
    print('Getting tweets before %s' %(oldesttweet))
    n_alltweets_old = n_alltweets
    
    new_tweets = api.user_timeline(screen_name = person, count = 200, max_id=oldesttweet)
    alltweets.extend(new_tweets)
    
    n_alltweets = len(alltweets)
    
    oldesttweet = alltweets[-1].id-1
    print('...%s tweets download so far' %(n_alltweets))

#%%  dar formato a los tweets descargados
out_alltweets = [[tweet.id_str,
                  tweet.in_reply_to_status_id,
                  tweet.user.screen_name,
                  tweet.in_reply_to_screen_name,
                  tweet.in_reply_to_user_id,
                  tweet.retweet_count,
                  tweet.created_at,
                  tweet.text] for tweet in alltweets]
pd_alltweets = pd.DataFrame(out_alltweets)
pd_alltweets.columns = ['id',
                        'rp_status_id',
                        'posted_by',
                        'in_rp_to_screen_name',
                        'rp_user_id',
                        'retweet_count',
                        'created_at',
                        'text']
                        
#%%filtrar los tweets en ciertas fechas

#limite de fechas
fecha_inicio = datetime(2017,4,10)
fecha_final = datetime(2017,4,27)

index = (pd_alltweets['created_at']>=fecha_inicio) & (pd_alltweets['created_at']<=fecha_final)
pd_tweets = pd_alltweets[index]
pd_tweets = pd_tweets.drop_duplicates()

#%% seleccionar el último tweet de la persona de interes

tweet_to_analyze = pd_tweets['id'].iloc[-1]
since_id = tweet_to_analyze

#%%obtencion de las menciones de la persona de interes
tweet_count = 0
max_tweets = 2000
max_id = -1

allmentions =[]

while tweet_count < max_tweets:
    try:
        if (max_id<=0):
            new_tweets = api.search(q=person, count=100, since_id=since_id)
        else:
            new_tweets = api.search(q=person, count=100, since_id=since_id, max_id=str(max_id-1))
        if not new_tweets:
            print('No more tweets found')
            break
        
        allmentions.extend(new_tweets)
        tweet_count = len(allmentions)
        print('Download %s tweets' % (tweet_count))
        max_id = new_tweets[-1].id
        
    except tweepy.TweepError as e:
        print('some error: ' +str(e))
        break

#%%  dar formato a los tweets descargados
out_alltweets = [[tweet.id_str,
                  tweet.in_reply_to_status_id,
                  tweet.user.screen_name,
                  tweet.in_reply_to_screen_name,
                  tweet.in_reply_to_user_id,
                  tweet.retweet_count,
                  tweet.created_at,
                  tweet.text] for tweet in allmentions]
pd_allmentions = pd.DataFrame(out_alltweets)
pd_allmentions.columns = ['id',
                        'rp_status_id',
                        'posted_by',
                        'in_rp_to_screen_name',
                        'rp_user_id',
                        'retweet_count',
                        'created_at',
                        'text']
                    
#%%
index = (pd_allmentions['rp_status_id']>0)
pd_somementions = pd_allmentions[index]

#%% crear grapho de las conversaciones
#teoria de grafos 
import networkx as nx
grapho = nx.DiGraph()

#Agregar las menciones de los tweets
for k in np.arange(len(pd_somementions)):
    tweet = pd_somementions.iloc[k,:]
    grapho.add_node(int(tweet.id),
                    tweet_v = tweet.text,
                    author = tweet.posted_by,
                    created_at_v = str(tweet.created_at))

nx.draw(grapho)

#%% agregar grapho del susodicho
#Agregar las menciones de los tweets
for k in np.arange(len(pd_tweets)):
    tweet = pd_tweets.iloc[k,:]
    grapho.add_node(int(tweet.id),
                    tweet_v = tweet.text,
                    author = tweet.posted_by,
                    created_at_v = str(tweet.created_at))

nx.draw(grapho)

#%%Agregar las Flechas a las conversaciones
for k in np.arange(len(pd_somementions)):
    tweet = pd_somementions.iloc[k,:]
    grapho.add_edge(int(tweet.rp_status_id),int(tweet.id))
nx.draw(grapho)

#%%grapho info
from operator import itemgetter
print(nx.info(grapho))
sorted_replied = sorted(grapho.degree_iter(),
                        key = itemgetter(1),
                        reverse =True)
most_replied_id,replies=sorted_replied[0]
longest_path = nx.dag_longest_path(grapho)

node = grapho.node[most_replied_id]
node['tweet_v']