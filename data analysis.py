from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re, os
import statsmodels.api as sm
from scipy.fft import fft, ifft
from statsmodels.tsa.seasonal import seasonal_decompose

%matplotlib inline
sns.set(style="whitegrid")
#pd.set_option('display.max_colwidth', None)

# For figure aesthetics
plt.rcParams['mathtext.fontset'] = 'custom'  
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'  
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'  
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'  
plt.rcParams['font.size'] = 22
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['font.family'] = 'STIXGeneral'

# number of messages by type of id ratio
# messages sent by tutor/student ratio
# messages sent by tutor/tutor ratio
# messages sent by student/student ratio
# messages sent by student/tutor ratio

def message_id_ratio(df, ids_profile = 'tutor', message_profile = 'PROFESSOR' ):
  meios = df['meio'].unique()
  msg_for_id_ratio = []

  if ids_profile == 'tutor':
    receiver = 'ALUNO'
    sender = 'PROFESSOR'
  else:
    receiver = 'PROFESSOR'
    sender = 'ALUNO' 

  for meio in meios:
    id_receivers = set(df[(df['profile_sender']==receiver) & (df['meio']==meio)]['id_receiver'].unique())
    id_senders = set(df[(df['profile_sender']==sender) & (df['meio']==meio)]['id_sender'].unique())
    ids = id_receivers.union(id_senders)
    n_id = len(ids)
    n_messages = len(df[(df['profile_sender']==message_profile) & (df['meio']==meio)])
    msg_for_id_ratio.append(n_messages/n_id)
  means_series = pd.Series(index = meios, data = msg_for_id_ratio)
  return means_series


# create vertical barchart with annotation
def annotate_barchart(values, labels, title, size = (8,5), col = None, rotate_xticks = False):
  plt.figure(figsize = size)
  plt.title(title)
  if rotate_xticks:
    plt.xticks(values, labels, rotation='vertical')
  if type(col) == str or col == None:
    g = sns.barplot(x=labels, y=values, color = col)
  elif type(col) == list:
    g = sns.barplot(x=labels, y=values, palette = col)
    
  for p in g.patches:
      g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., 
                                                p.get_height()), ha = 'center', 
                va = 'center', xytext = (0, 5), textcoords = 'offset points')
  plt.show()

#clrs = ['grey' if (x < max(values)) else 'red' for x in values ]
#sb.barplot(x=idx, y=values, palette=clrs) # color=clrs)

# create donnut chart
def donnut(values, labels, size = (8,8), col = None, pct = 0.5):
  plt.figure(figsize=size)
  my_circle=plt.Circle( (0,0), 0.6, color='white')
  plt.pie(values, labels=labels, autopct='%1.1f%%',startangle=90, pctdistance=pct, colors = col)
  p=plt.gcf()
  p.gca().add_artist(my_circle)
  plt.show()

  #load all files
files_path = '/content/drive/MyDrive/Mineracao/Logs'
files_list = os.listdir(files_path)
dfs = []
for log in files_list:
  print(log)
  file_path = files_path + '/' + log
  df = pd.read_csv(file_path, error_bad_lines=False, sep=',')
  dfs.append(df)
df = pd.concat(dfs)

# ordem cronológica
df['data_hora_mensagem'] = pd.to_datetime(df['data_hora_mensagem'])
df = df.sort_values(by='data_hora_mensagem')
print()
print('Logs do STUART')
df.reset_index(inplace=True,drop=True)
df.head(3)

# tutores
tutor_csv = '/content/drive/MyDrive/Mineracao/Logs_tutores/tutors_updated.csv'
df_tutors = pd.read_csv(tutor_csv, error_bad_lines=False, sep=',')
df_tutors['data_hora'] = pd.to_datetime(df_tutors['data_hora'], dayfirst= True)

# igualar a quantidade de semanas para antes e depois do STUART
#df_tutors = df_tutors[df_tutors['data_hora'] >= '2020-11-17']


df_tutors = df_tutors[df_tutors['meio']=='chat']
df_tutors

df_tutors_before = df_tutors[df_tutors['data_hora'] < '2021-01-06']
df_tutors_after = df_tutors[df_tutors['data_hora'] >= '2021-01-06']

def get_set_courses(df, column = 'cursos'):
  cursos = df[column].unique()
  set_cursos = []
  for c in cursos:
    if type(c) != str:
      c = ''
    list_cursos = c.split(', ')
    set_cursos = set_cursos + list_cursos
  set_cursos = set(set_cursos)
  return set_cursos

set_cursos_antes = get_set_courses(df_tutors_before,column = 'cursos')
print("cursos dos alunos no chat antes do stuart")
print(set_cursos_antes)

set_cursos_depois = get_set_courses(df_tutors_after,column = 'cursos')
print("\ncursos dos alunos no chat após o stuart")
print(set_cursos_depois)

set_cursos_stuart = get_set_courses(df[(df['autor_da_mensagem']!='STUART')],column = 'cursos_usuario')
print("\ncursos dos alunos no chat do stuart")
print(set_cursos_stuart)
print('diferença dos cursos antes e do stuart')
set_cursos_antes.difference(set_cursos_stuart)