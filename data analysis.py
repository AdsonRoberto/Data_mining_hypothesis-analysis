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

df.head()

#Dados básicos
print('quantidade de alunos diferentes em cada meio que interagiram')
df_tutors[df_tutors['profile_sender']=='ALUNO'].groupby('meio')['id_sender'].nunique()

df.head(1)
df_tutors_before = df_tutors[df_tutors['data_hora'] < '2021-01-06']
df_tutors_after = df_tutors[df_tutors['data_hora'] >= '2021-01-06']

students_before = set(df_tutors_before[df_tutors_before['profile_sender']=='ALUNO']['id_sender'].unique())
#students_before = students_before.union(set(df_tutors_before[df_tutors_before['profile_sender']=='PROFESSOR']['id_receiver'].unique()))
students_stuart = df[df['autor_da_mensagem']!='STUART']['remetente'].nunique()
students_after = set(df_tutors_after[df_tutors_after['profile_sender']=='ALUNO']['id_sender'].unique())
#students_after = students_after.union(set(df_tutors_after[df_tutors_after['profile_sender']=='PROFESSOR']['id_receiver'].unique()))
print('quantidade de estudantes que interagiram com STUART: {a}'.format(a=students_stuart))
print('quantidade de estudantes antes do STUART: {a}'.format(a=len(students_before)))
print('quantidade de estudantes depois do STUART: {a}'.format(a=len(students_after)))
print('interseção entre os dois períodos: {a}'.format(a=len(students_before.intersection(students_after))))
print('total: {a}'.format(a=len(students_before.union(students_after))))

students_tutor_chat = set(df_tutors[(df_tutors['profile_sender']=='ALUNO') & 
          (df_tutors['meio']=='chat')]['id_sender'].unique())

students_stuart_chat = set(df[(df['autor_da_mensagem']!='STUART')]['remetente'].unique())

print('total de estudantes no experimento: ',len(students_tutor_chat.union(students_stuart_chat)))

# mensagens enviadas por alunos e tutores
print('mensagens enviadas por alunos a tutores antes do stuart: ', len(df_tutors_before[df_tutors_before['profile_sender']=='ALUNO']))
print('mensagens enviadas por alunos a tutores depois do stuart: ', len(df_tutors_after[df_tutors_after['profile_sender']=='ALUNO']))
print()
print('mensagens enviadas por tutores a alunos antes do stuart: ', len(df_tutors_before[df_tutors_before['profile_sender']!='ALUNO']))
print('mensagens enviadas por tutores a alunos depois do stuart: ', len(df_tutors_after[df_tutors_after['profile_sender']!='ALUNO']))

tutors_before = set(df_tutors_before[df_tutors_before['profile_sender']=='PROFESSOR']['id_sender'].unique())
#tutors_before = tutors_before.union(set(df_tutors_before[df_tutors_before['profile_sender']=='ALUNO']['id_receiver'].unique()))

tutors_after = set(df_tutors_after[df_tutors_after['profile_sender']=='PROFESSOR']['id_sender'].unique())
#tutors_after = tutors_after.union(set(df_tutors_after[df_tutors_after['profile_sender']=='ALUNO']['id_receiver'].unique()))

print('quantidade de tutores antes do STUART: {a}'.format(a=len(tutors_before)))
print('quantidade de tutores depois do STUART: {a}'.format(a=len(tutors_after)))
print('interseção entre os dois períodos: {a}'.format(a=len(tutors_before.intersection(tutors_after))))
print('total: {a}'.format(a=len(tutors_before.union(tutors_after))))

#Quantidades de vezes que o participante consulta o tutor humano e o chatbot
# remove null
print('Chat com STUART')
df = df.dropna(subset=['mensagem'])
print('Total de mensagens enviados:',len(df))
#

df['timestamp'] = pd.to_datetime(df['data_hora_mensagem']) #, format='%d/%m/%y %H:%M')

janela = 'dia'

if janela == 'semana':
  frame = '168H' # one week
elif janela == 'dia':
  frame = '24H'
elif janela == 'hora':
  frame = '1H'
elif janela == 'mês':
  frame = '30D'

timeseries = df.groupby('timestamp').count()['remetente']
timeseries = timeseries.resample(frame).sum()
print('Total de {a}s: {b}'.format(a=janela,b=len(timeseries)))
print('Média de mensagens trocadas no chat do STUART por {a}: {b:.1f}'.format(a = janela, b = timeseries.mean()))


#serie temporal
frame = '24H'

timeseries_users = df[df['autor_da_mensagem'] !='STUART'].groupby('timestamp').count()['remetente'].resample(frame).sum().to_frame()
timeseries_users.reset_index(inplace=True)
timeseries_users.columns = ['data_hora','mensagens']
timeseries_users['destinatário'] = len(timeseries_users) * ['STUART']
timeseries_users['MA'] = timeseries_users['mensagens'].rolling(window=14).mean()

df_std2tut = df_tutors[(df_tutors['profile_sender']=='ALUNO') & (df_tutors['meio']=='chat')]
timeseries_std2tut = df_std2tut.groupby('data_hora').count()['id_sender'].resample(frame).sum().to_frame()
timeseries_std2tut.reset_index(inplace=True)
timeseries_std2tut
timeseries_std2tut.columns = ['data_hora','mensagens']
timeseries_std2tut['destinatário'] = len(timeseries_std2tut)*['TUTORES']
timeseries_std2tut['MA'] = timeseries_std2tut['mensagens'].rolling(window=14).mean()

timeseries_std = pd.concat([timeseries_users,timeseries_std2tut])
#timeseries_std = timeseries_std[timeseries_std['destinatário']=='TUTORES']

fig = px.line(timeseries_std, x='data_hora', y='mensagens',
              color="destinatário")

fig.update_layout(legend=dict(
    yanchor="bottom",
    y=1.0,
    xanchor="left",
    x=0.01
))
fig.show()
