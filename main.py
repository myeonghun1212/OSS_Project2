import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time

start = time.time()

ratings_data = pd.read_csv('ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'],engine='python')

user_item_matrix = ratings_data.pivot_table('rating', index='user_id', columns='movie_id')

ids = list(range(1, 3953))

user_item_matrix = user_item_matrix.reindex(columns=ids)

user_item_matrix = user_item_matrix.fillna(0)

# kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=3, n_init=10, random_state=0))
kmeans.fit(user_item_matrix)

labels = kmeans.predict(user_item_matrix)

# 각 유저의 클러스터 레이블 확인
user_groups = pd.DataFrame({'user_id': ratings_data['user_id'].unique(), 'group': labels})

user_item_matrix = user_item_matrix.astype(np.uint8)

def get_user_item_matrix_by_group(user_item_matrix, user_groups, group_id):
    group_users = user_groups[user_groups['group'] == group_id]['user_id']
    group_matrix = user_item_matrix.loc[group_users]
    return group_matrix

def AU(group_matrix):
    return group_matrix.sum(axis=0)

def Avg(group_matrix):
    return group_matrix.mean(axis=0)

def SC(group_matrix):
    return group_matrix.notnull().sum(axis=0) 

def AV(group_matrix):
    return (group_matrix >= 4).sum(axis=0)

def BC(group_matrix):
    res = pd.DataFrame(columns = group_matrix.columns)
    for i in group_matrix.index:
        temp = group_matrix.loc[i]
        temp = temp.rank(method='average') - 1
        res.loc[i] = temp
    return res.sum(axis = 0)

    
    
# 벡터화로 한번에 계산하기
def BC_Optimized(group_matrix):
    return group_matrix.rank(method='average', axis=1).sum(axis=0) - group_matrix.shape[0]

# 전통적인 깡 구현
def CR(group_matrix):
    n = 100
    res = pd.DataFrame(index=group_matrix.columns[:n], columns=group_matrix.columns[:n], dtype=float)
    for i in group_matrix.columns[:n]:
        for j in group_matrix.columns[:n]:
            if i == j:
                continue
            g = (group_matrix[i] > group_matrix[j]).value_counts().get(True, 0)
            l = (group_matrix[i] < group_matrix[j]).value_counts().get(True, 0)
            if g > l:
                res.at[i,j] = 1
            elif g == l:
                res.at[i,j] = 0
            else:
                res.at[i,j] = -1
    return res.sum(axis=0)

# 벡터화를 통해 유저별로 어느 영화가 평이 좋은지 한번에 계산해버리자
def CR_Optimized(group_matrix):
    n = len(group_matrix.columns)
    # n = 1000
    nparr = group_matrix.T.values
    res = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1, n):
            res[i,j] = (nparr[i] > nparr[j]).sum()
            res[j,i] = (nparr[i] < nparr[j]).sum()
    
    return pd.Series(  np.sign(res - res.T).sum(axis = 0), index = group_matrix.columns[:n])

# 삼각형 인덱스를 반환하는 triu_indices를 써서 벡터화로 전부 한번에 계산해버리자
def CR_Optimized2(group_matrix):
    n = len(group_matrix.columns)
    # n = 1000
    nparr = group_matrix.T.values
    res = np.zeros((n,n))
    idx_upper = np.triu_indices(n, k=1)
    res[idx_upper] = (nparr[idx_upper[0]] > nparr[idx_upper[1]]).sum(axis = 1)
    res.T[idx_upper] = (nparr[idx_upper[0]] < nparr[idx_upper[1]]).sum(axis = 1)
    result = pd.Series(  np.sign(res - res.T).sum(axis = 0), index = group_matrix.columns[:n])
    del nparr, res, idx_upper
    return result

# @profile
def CR_Optimized2_Sub(group_matrix, res, idx_upper):
    res[idx_upper] += (group_matrix.values[idx_upper[0]] > group_matrix.values[idx_upper[1]]).sum(axis = 1)
    res.T[idx_upper] += (group_matrix.values[idx_upper[0]] < group_matrix.values[idx_upper[1]]).sum(axis = 1)

# @profile
# 전부 올리니까 메모리 문제가 생김 -> 유저 1~50까지의 평가 CR 유저 51~100까지의 평가 CR ... 로 쪼개서 시행 후 합산하자
def CR_Optimized2_Chunk(group_matrix):
    n = len(group_matrix.columns)
    idx_upper = np.triu_indices(n, k=1)
    res = np.zeros((n,n), dtype = np.int16)
    total = len(group_matrix.index)
    chunk_size = 1000 # colab tpu 같이 용량 큰데서 이정도면 돌아감
    # chunk_size = 100 # 데스크탑 램에서 돌아갈 사이즈
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        subgroup = group_matrix.iloc[start:end, :]
        CR_Optimized2_Sub(subgroup.T, res, idx_upper)
        del subgroup

    result = pd.Series(  np.sign(res - res.T).sum(axis = 0), index = group_matrix.columns)
    del res
    return result

def printList(series):
    for i, v in enumerate(series):
        print(f'{i+1}. {v}')

for groupnumber in range(3):
  group = get_user_item_matrix_by_group(user_item_matrix, user_groups, groupnumber)
  additiveUtilitarian = AU(group)
  average = Avg(group)
  simpleCount = SC(group)
  approvalVoting = AV(group)
  bordaCount = BC_Optimized(group)
  copelandRule = CR_Optimized2_Chunk(group)
  results_df = pd.DataFrame({
      'AU': additiveUtilitarian.nlargest(10).index,
      'Avg': average.nlargest(10).index,
      'SC': simpleCount.nlargest(10).index,
      'AV': approvalVoting.nlargest(10).index,
      'BC': bordaCount.nlargest(10).index,
      'CR': copelandRule.nlargest(10).index
  })

  print(f"Group {groupnumber} top 10 results:")
  print(results_df)
  print()
end = time.time()
print(f"{end - start:.5f} sec")