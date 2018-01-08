# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:50:39 2017

@author: if692068
"""

import numpy as np
import tweepy
import pandas as pd
from tweepy.auth import OAuthHandler

#%% Cuentas de acceso
Consumer_Key = 'mfXeGc2wOSaeVlKtDmEY4PbOY'
Consumer_Secret = 'L3rqtmGJnwlEtWqGit7H33Vc0PWurMGwY9lat38pxQpbxVaZWt'
Access_Token = '851652483847999490-9d7Jg5BN3TdY8lDYWcJPyFcErly4N3i'
Access_Token_Secret = 'El3TsIJq6k4mf1DcHxiwUFYaePiqPxiZOf3AF8IXMK2uZ'

#%%abrir la conexion de la API
auth = tweepy.OAuthHandler(Consumer_Key,Consumer_Secret)
auth.set_access_token(Access_Token,Access_Token_Secret)
api = tweepy.API(auth)

#%% elegir el usuario a seguir
person='@BMVMercados'
new_tweets=api.user_timeline(screen_name=person)

#%% Extraer información de los tweets

out_tweets=[[tweet.id.str,
             tweet.in_reply_to_status_id,
             tweet.user.screen_name,
             tweet.in_reply_to_screen_name,
             tweet.in_reply_to_user_id,
             tweet.retweet_count,
             tweet.created_at,
             tweet.text]for tweet in new_tweets]
pd_tweets=pd.DataFrame(out_tweets)
pd_tweets.columns=['id_srt','in_reply_to_status_id','user.screen_name','in_reply_to_screen_name','in_reply_to_user_id','retweet_count','created_at','text']

#%% tweets con mas retweets

n_max=pd_tweets['retweet_count'].max()
index=pd_tweets['retweet_count']==n_max
pd_tweets[index]

#%% menciones de las personas de interes
new_tweets=api.search(q=person,count=200)

