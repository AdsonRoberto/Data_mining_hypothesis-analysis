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

series = timeseries_std2tut[['data_hora', 'mensagens']].set_index('data_hora')
series = series.asfreq('D')

result_a = seasonal_decompose(series, model='additive')
result_m = seasonal_decompose(series, model='multiplicative')

fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)

sns.lineplot(data=result_a.observed, x='data_hora', y='mensagens', ax=axes[0, 0])
sns.lineplot(data=result_a.seasonal, x='data_hora', y='mensagens', ax=axes[0, 1])
sns.lineplot(data=result_a.trend, x='data_hora', y='mensagens', ax=axes[0, 2])
sns.lineplot(data=result_a.resid, x='data_hora', y='mensagens', ax=axes[0, 3])

sns.lineplot(data=result_m.observed, x='data_hora', y='mensagens', ax=axes[1, 0])
sns.lineplot(data=result_m.seasonal, x='data_hora', y='mensagens', ax=axes[1, 1])
sns.lineplot(data=result_m.trend, x='data_hora', y='mensagens', ax=axes[1, 2])
sns.lineplot(data=result_m.resid, x='data_hora', y='mensagens', ax=axes[1, 3])

plt.tight_layout()

sns.lineplot(data=result_a.trend + result_a.seasonal, x='data_hora', y='mensagens')
sns.lineplot(data=result_a.observed, x='data_hora', y='mensagens')

trend_tutores = result_a.trend
seasonal_tutores = result_a.seasonal
observed_tutores = result_a.observed

series = timeseries_users[['data_hora', 'mensagens']].set_index('data_hora')
series = series.asfreq('D')

result_a = seasonal_decompose(series, model='additive')
result_m = seasonal_decompose(series, model='multiplicative')

fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)

sns.lineplot(data=result_a.observed, x='data_hora', y='mensagens', ax=axes[0, 0])
sns.lineplot(data=result_a.seasonal, x='data_hora', y='mensagens', ax=axes[0, 1])
sns.lineplot(data=result_a.trend, x='data_hora', y='mensagens', ax=axes[0, 2])
sns.lineplot(data=result_a.resid, x='data_hora', y='mensagens', ax=axes[0, 3])

sns.lineplot(data=result_m.observed, x='data_hora', y='mensagens', ax=axes[1, 0])
sns.lineplot(data=result_m.seasonal, x='data_hora', y='mensagens', ax=axes[1, 1])
sns.lineplot(data=result_m.trend, x='data_hora', y='mensagens', ax=axes[1, 2])
sns.lineplot(data=result_m.resid, x='data_hora', y='mensagens', ax=axes[1, 3])

plt.tight_layout()

sns.lineplot(data=result_a.trend + result_a.seasonal, x='data_hora', y='mensagens')
sns.lineplot(data=result_a.observed, x='data_hora', y='mensagens')

trend_stuart = result_a.trend
seasonal_stuart = result_a.seasonal
observed_stuart = result_a.observed

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.lineplot(data=trend_tutores, x='data_hora', y='mensagens', color='red', ax=ax, label="Tendência")
sns.lineplot(data=observed_tutores, x='data_hora', y='mensagens', alpha=0.5, color='k', ax=ax, label="Observação")
ax.set_xticklabels(['Jan/2021', 'Fev/2021', 'Mar/2021', 'Abr/2021', 'Maio/2021'])
ax.set_xlabel('')
ax.set_ylabel('Número de Mensagens')

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.lineplot(data=trend_stuart, x='data_hora', y='mensagens', color='red', ax=ax, label="Tendência")
sns.lineplot(data=observed_stuart, x='data_hora', y='mensagens', alpha=0.5, color='k', ax=ax, label="Observação")
ax.set_xticklabels(['Jan/2021', 'Fev/2021', 'Mar/2021', 'Abr/2021', 'Maio/2021'])
ax.set_xlabel('')
ax.set_ylabel('Número de Mensagens')

fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

sns.lineplot(data=trend_tutores + seasonal_tutores, x='data_hora', y='mensagens', ax=axes[0])
sns.lineplot(data=timeseries_std2tut, x='data_hora', y='mensagens', ax=axes[0])

sns.lineplot(data=trend_stuart + seasonal_stuart, x='data_hora', y='mensagens', ax=axes[1])
sns.lineplot(data=timeseries_users, x='data_hora', y='mensagens', ax=axes[1])

fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

sns.lineplot(data=observed_tutores, x='data_hora', y='mensagens', ax=axes[0], alpha=0.5, color='k')
sns.lineplot(data=trend_tutores, x='data_hora', y='mensagens', ax=axes[0], color='r')

sns.lineplot(data=observed_stuart, x='data_hora', y='mensagens', ax=axes[1], alpha=0.5, color='k')
sns.lineplot(data=trend_stuart, x='data_hora', y='mensagens', ax=axes[1], color='r')

#axes[1].set_xticklabels(['Jan/2021', 'Fev/2021', 'Mar/2021', 'Abr/2021', 'Maio/2021', 'Jun/2021', 'Jul/2021'])
axes[1].set_xticklabels(['Set/2020', 'Out/2020', 'Nov/2020', 'Dez/2020', 'Jan/2021', 'Fev/2021', 'Mar/2021', 'Abr/2021', 'Maio/2021'])
axes[1].set_xlabel('')
axes[1].set_ylabel('Número de Mensagens')

fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True)

sns.lineplot(data=observed_tutores, x='data_hora', y='mensagens', ax=ax, alpha=0.5, color='k', label='Observado (Tutors)')
sns.lineplot(data=trend_tutores, x='data_hora', y='mensagens', ax=ax, color='r', label='Tendência (Tutors)')

sns.lineplot(data=observed_stuart, x='data_hora', y='mensagens', ax=ax, alpha=0.5, color='k', linestyle="--", label="Observado (Stuart)")
sns.lineplot(data=trend_stuart, x='data_hora', y='mensagens', ax=ax, color='b', label="Tendência (Stuart)")
plt.legend()

ticks_locations = ax.get_xticks()
ax.set_xticklabels(['Set/2020', 'Out/2020', 'Nov/2020', 'Dez/2020', 'Jan/2021', 'Fev/2021', 'Mar/2021', 'Abr/2021', 'Mai/2021', 'Jun/2021', 'Jul/2021'], rotation=45)
ax.set_xlabel('')
ax.set_ylabel('Número de Mensagens')
plt.tight_layout()

ticks_locations

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

sns.lineplot(data=trend_tutores, x='data_hora', y='mensagens', ax=ax, color='r', label='Tutors')
sns.lineplot(data=trend_stuart, x='data_hora', y='mensagens', ax=ax, color='b', label='Stuart')
ax.set_xticklabels(['Oct/2020', 'Nov/2020', 'Dec/2020', 'Jan/2021', 'Feb/2021', 'Mar/2021', 'Apr/2021', 'May/2021'], rotation=45)
ax.set_xlabel('')
ax.set_ylabel('Number of Messages')
plt.legend(title='Trend')

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

sns.lineplot(data=seasonal_tutores, x='data_hora', y='mensagens', ax=ax, color='r', label='Tutors')
sns.lineplot(data=seasonal_stuart, x='data_hora', y='mensagens', ax=ax, color='b', label='Stuart')
ax.set_xticklabels(['Sep/2020', 'Oct/2020', 'Nov/2020', 'Dec/2020', 'Jan/2021', 'Feb/2021', 'Mar/2021', 'Apr/2021', 'May/2021'], rotation=45)
ax.set_xlabel('')
ax.set_ylabel('Number of Messages')
plt.legend(title='Seasonality')

seasonal_stuart.index

t = seasonal_stuart.index
x = seasonal_stuart['mensagens'].values
ind = np.where(np.isclose(x, np.min(x)))[0]
period = np.diff(ind)[0]
print(period)

t = seasonal_tutores.index
x = seasonal_tutores['mensagens'].values
ind = np.where(np.isclose(x, np.min(x)))[0]
period = np.diff(ind)[0]
print(period)

X = fft(timeseries_std2tut['mensagens'])
plt.plot(np.abs(X)[1:])

X = fft(timeseries_std2tut['mensagens'])
amplitudes = np.abs(X)[1:]
# Filter low amplitude frequencies
threshold = 500
X[np.where(amplitudes < threshold)[0]] = 0
print('filtering {} frequencies'.format(len(np.where(amplitudes < threshold)[0])))

filtered_series = ifft(X)

plt.plot(timeseries_std2tut['mensagens'])
plt.plot(np.real(filtered_series))

last_time = timeseries_std2tut['data_hora'].max()
next_step_time = last_time + pd.Timedelta(days=14)
print(last_time, next_step_time)

model = sm.tsa.SARIMAX(timeseries_std2tut['MA'].values, order=(1, 1, 1), trend='ct')
result = model.fit()

print(result.summary())

forecasted = result.forecast(steps=7)
timestamps = [
    last_time + pd.Timedelta(days=14 * i) for i in range(len(forecasted))
]

df_forecasted = pd.DataFrame()
df_forecasted['data_hora'] = timestamps
df_forecasted['forecast'] = forecasted

df_forecasted

fig = px.line(timeseries_std, x='data_hora', y='MA',
              color="destinatário")

fig.update_layout(legend=dict(
    yanchor="bottom",
    y=1.0,
    xanchor="left",
    x=0.01
))
fig.show()

fig, ax = plt.subplots(figsize=(15, 5))
sns.lineplot(data=timeseries_std, x='data_hora', y='MA',
             hue="destinatário", ax=ax)
sns.lineplot(data=df_forecasted, x='data_hora', y='forecast', ax=ax, color='k', linestyle='--')

# stuart and students
timeseries_stuart = df[df['autor_da_mensagem'] =='STUART'].groupby('timestamp').count()['remetente'].resample(frame).sum().to_frame()
timeseries_users = df[df['autor_da_mensagem'] !='STUART'].groupby('timestamp').count()['remetente'].resample(frame).sum().to_frame()
df_timeseries = pd.concat([timeseries_stuart,timeseries_users])
df_timeseries['autor'] = ['STUART']*len(timeseries_stuart) + ['USUÁRIOS']*len(timeseries_users)
df_timeseries.reset_index(level=0, inplace=True)


mean_stuart = timeseries_users.mean().values[0]

print('Média de mensagens enviadas por alunos por {a} no chat para o STUART: {b:.1f}'.format(a=janela,b=mean_stuart))
print()
df_std2tut = df_tutors[(df_tutors['profile_sender']=='ALUNO') & (df_tutors['meio']=='chat')]
timeseries_std2tut =  df_std2tut.groupby('data_hora').count()['id_sender'].resample(frame).sum().to_frame()
#print('Média de mensagens enviadas por alunos por dia no chat para o tutores (geral):',timeseries_students_tutors.mean().values[0])

timeseries_std2tut_after = timeseries_std2tut[timeseries_std2tut.index >= '2021-01-06']
timeseries_std2tut_before = timeseries_std2tut[timeseries_std2tut.index < '2021-01-06']

print('Mensagens no chat de alunos para tutores')
timeseries_df = timeseries_std2tut_before.describe()
timeseries_df['com STUART'] = timeseries_std2tut_after.describe().values
timeseries_df.columns = ['sem STUART', 'com STUART']
timeseries_df.round(1)

# stuart and students
frame = '7D'
timeseries_stuart = df[df['autor_da_mensagem'] =='STUART'].groupby('data_hora_mensagem').count()['remetente'].resample(frame).sum().to_frame()
timeseries_users = df[df['autor_da_mensagem'] !='STUART'].groupby('data_hora_mensagem').count()['remetente'].resample(frame).sum().to_frame()
df_timeseries = pd.concat([timeseries_stuart,timeseries_users])
df_timeseries['autor'] = ['STUART']*len(timeseries_stuart) + ['USUÁRIOS']*len(timeseries_users)
df_timeseries.reset_index(level=0, inplace=True)

mean_stuart = timeseries_users.mean().values[0]

print('Média de mensagens enviadas por alunos por {a} no chat para o STUART: {b:.1f}'.format(a=janela,b=mean_stuart))
print()
df_std2tut = df_tutors[(df_tutors['profile_sender']=='ALUNO') & (df_tutors['meio']=='chat')]
timeseries_std2tut =  df_std2tut.groupby('data_hora').count()['id_sender'].resample(frame).sum().to_frame()
#print('Média de mensagens enviadas por alunos por dia no chat para o tutores (geral):',timeseries_students_tutors.mean().values[0])

timeseries_std2tut_after = timeseries_std2tut[timeseries_std2tut.index >= '2021-01-06']
timeseries_std2tut_before = timeseries_std2tut[timeseries_std2tut.index < '2021-01-06']

print('Mensagens no chat de alunos para tutores')
timeseries_df = timeseries_std2tut_before.describe()
timeseries_df['com STUART'] = timeseries_users.describe().values
timeseries_df.columns = ['mensagens para tutores', 'mensagens para STUART']
timeseries_df.round(1)

sns.set(style="darkgrid")
antes = ['Sem STUART']*len(timeseries_std2tut_before)
apos = ['Com STUART']*len(timeseries_std2tut_after)
df_time_distributions = pd.concat([timeseries_std2tut_before.reset_index(),timeseries_std2tut_after.reset_index()])
df_time_distributions['Presença do STUART'] = antes+apos
df_time_distributions.columns = ['data_hora','mensagens', 'presença do STUART']
plt.figure(figsize=(8,8))
#plt.title('Distribuições de mensagens por intervalo de tempo de 1 {a}'.format(a=janela))
sns.boxplot(x="presença do STUART", y="mensagens", palette=['C0','C3'], data=df_time_distributions) #['C2','C5']
plt.show()

before_mean = timeseries_std2tut_before.mean().values[0]
after_mean = timeseries_std2tut_after.mean().values[0]

print('Total de {a}s antes do STUART: {b}'.format(a=janela,b=len(timeseries_std2tut_before)))
print('Média de mensagens enviadas por alunos por {a} no chat para o tutores (antes do STUART): {b:.1f}'.format(a=janela,b=before_mean))

print()

print('Total de {a}s depois do STUART: {b}'.format(a=janela,b=len(timeseries_std2tut_after)))
print('Média de mensagens enviadas por alunos por {a} no chat para o tutores (depois do STUART): {b:.1f}'.format(a=janela,b=after_mean))
print()
variation = 100*(after_mean - before_mean)/before_mean
print('Variação antes/depois: {a:.1f}%'.format(a=variation))

print()
variation_stuart = 100*(mean_stuart - after_mean)/after_mean
print('Alunos enviaram em média {a:.1f}% a mais de mensagens para o STUART do que para tutores'.format(a=variation_stuart))

#Teste de hipótese

from scipy import stats
# teste de normalidade.
# Abaixo de 0.05, significativo. Acima de 0.05, não significativo.
# No caso de uma valor significativo para a estatística do teste, isso indica falta de normalidade para a variável aleatória analisada
# p > 0.05 = NÃO-SIGNIFICATIVO = NORMAL.
s, p = stats.shapiro(timeseries_std2tut_before)
print('antes',s,p)
if p > 0.05:
  print('provavelmente normal')
else:
  print('provavelmente não é normal')
  
print()

s,p = stats.shapiro(timeseries_std2tut_after)
print('depois',s,p)
if p > 0.05:
  print('provavelmente normal')
else:
  print('provavelmente não é normal')

  stat, p = stats.ttest_ind(timeseries_std2tut_after, timeseries_std2tut_before)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('provavelmente a mesma média')
else:
	print('provavelmente médias diferentes')

#redução da carga de trabalho de tutores

before_stuart = message_id_ratio(df_tutors[df_tutors['data_hora'] < '2021-01-06' ],ids_profile='aluno')
after_stuart = message_id_ratio(df_tutors[df_tutors['data_hora'] >= '2021-01-06' ],ids_profile='aluno')

print('Mensagens de tutor/aluno')
df_message_by_student = pd.DataFrame({'Antes do STUART' : before_stuart, 'Depois do STUART': after_stuart})
df_message_by_student['variação (%)'] = 100*(df_message_by_student['Depois do STUART']-df_message_by_student['Antes do STUART'])/df_message_by_student['Antes do STUART']
df_message_by_student.round(1)

# média de mensagens por tutor
before_stuart = message_id_ratio(df_tutors[df_tutors['data_hora'] < '2021-01-06' ],ids_profile='tutor')
after_stuart = message_id_ratio(df_tutors[df_tutors['data_hora'] >= '2021-01-06' ],ids_profile='tutor')

print('Média de mensagens por tutor')
df_message_by_student = pd.DataFrame({'Antes do STUART' : before_stuart, 'Depois do STUART': after_stuart})
df_message_by_student['variação (%)'] = 100*(df_message_by_student['Depois do STUART']-df_message_by_student['Antes do STUART'])/df_message_by_student['Antes do STUART']
df_message_by_student.round(1)


