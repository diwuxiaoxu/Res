import tensorflow as tf

# ========================================= 1、embedding =========================================
"""
embedding = tf.constant(
    [[0.21, 0.41, 0.51, 0.11],
    [0.22, 0.42, 0.52, 0.12],
    [0.23, 0.43, 0.53, 0.13],
    [0.24, 0.44, 0.54, 0.14]],dtype=tf.float32)

feature_batch = tf.constant([2,3,1,0])

get_embedding1 = tf.nn.embedding_lookup(embedding,feature_batch)
# 根据feature_batch指定的索引寻找，返回embedding中对应的行

feature_batch_one_hot = tf.one_hot(feature_batch,depth=4)
get_embedding2 = tf.matmul(feature_batch_one_hot,embedding)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embedding1,embedding2 = sess.run([get_embedding1,get_embedding2])
    print(embedding1)
    print(embedding2)
"""
# ========================================= 2、数据处理 =========================================
import pandas as pd

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# 连续变量列
NUMERIC_COLS = [
    "ps_reg_01","ps_reg_02","ps_reg_03",
    "ps_car_12","ps_car_13","ps_car_14","ps_car_15"
]

# 不考虑的变量列
IGNORE_COLS = [
    "id", "target",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]

dfTrain = pd.read_csv(TRAIN_FILE)  # 10000*59
dfTest = pd.read_csv(TEST_FILE)  # 2000*58
print(dfTrain.head(10))


# 除了连续变量、不考虑变量，剩下的就是离散变量列。
# 获取一个feature-map，定义如何将变量的不同取值转换为其对应的特征索引feature-index

df = pd.concat([dfTrain, dfTest])  # 训练集和测试集的id是交叉不重复的，可以填充 。。。 12000*59

# 变量取值到特征索引的对应关系
feature_dict = {}
# 总的特征数量
total_feature = 0
for col in df.columns:
    if col in IGNORE_COLS:
        continue
    elif col in NUMERIC_COLS:
        # 对于连续变量，直接是变量名到索引的映射
        feature_dict[col] = total_feature
        total_feature += 1
    else:
        # 离散变量
        unique_val = df[col].unique()  # unique 去重
        # zip 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回这些元组组成的列表对象。
        # 对离散变量,内部会嵌套一个二级map,定义了该离散变量的不同取值到索引的映射
        feature_dict[col] = dict(zip(unique_val,range(total_feature,len(unique_val) + total_feature)))
        total_feature += len(unique_val)
print(total_feature)  # 254
print(feature_dict)

"""
{'ps_car_01_cat': {10: 0, 11: 1, 7: 2, 6: 3, 9: 4, 5: 5, 4: 6, 8: 7, 3: 8, 0: 9, 2: 10, 1: 11, -1: 12}, 
'ps_car_02_cat': {1: 13, 0: 14}, 
'ps_car_03_cat': {-1: 15, 0: 16, 1: 17}, 
'ps_car_04_cat': {0: 18, 1: 19, 8: 20, 9: 21, 2: 22, 6: 23, 3: 24, 7: 25, 4: 26, 5: 27}, 
'ps_car_05_cat': {1: 28, -1: 29, 0: 30}, 
'ps_car_06_cat': {4: 31, 11: 32, 14: 33, 13: 34, 6: 35, 15: 36, 3: 37, 0: 38, 1: 39, 10: 40, 12: 41, 9: 42, 17: 43, 7: 44, 8: 45, 5: 46, 2: 47, 16: 48}, 
'ps_car_07_cat': {1: 49, -1: 50, 0: 51}, 'ps_car_08_cat': {0: 52, 1: 53}, 
'ps_car_09_cat': {0: 54, 2: 55, 3: 56, 1: 57, -1: 58, 4: 59}, 
'ps_car_10_cat': {1: 60, 0: 61, 2: 62}, 
'ps_car_11': {2: 63, 3: 64, 1: 65, 0: 66},
 'ps_car_11_cat': {12: 67, 19: 68, 60: 69, 104: 70, 82: 71, 99: 72, 30: 73, 68: 74, 20: 75, 36: 76, 101: 77, 103: 78, 41: 79, 59: 80, 43: 81, 64: 82, 29: 83, 95: 84, 24: 85, 5: 86, 28: 87, 87: 88, 66: 89, 10: 90, 26: 91, 54: 92, 32: 93, 38: 94, 83: 95, 89: 96, 49: 97, 93: 98, 1: 99, 22: 100, 85: 101, 78: 102, 31: 103, 34: 104, 7: 105, 8: 106, 3: 107, 46: 108, 27: 109, 25: 110, 61: 111, 16: 112, 69: 113, 40: 114, 76: 115, 39: 116, 88: 117, 42: 118, 75: 119, 91: 120, 23: 121, 2: 122, 71: 123, 90: 124, 80: 125, 44: 126, 92: 127, 72: 128, 96: 129, 86: 130, 62: 131, 33: 132, 67: 133, 73: 134, 77: 135, 18: 136, 21: 137, 74: 138, 37: 139, 48: 140, 70: 141, 13: 142, 15: 143, 102: 144, 53: 145, 65: 146, 100: 147, 51: 148, 79: 149, 52: 150, 63: 151, 94: 152, 6: 153, 57: 154, 35: 155, 98: 156, 56: 157, 97: 158, 55: 159, 84: 160, 50: 161, 4: 162, 58: 163, 9: 164, 17: 165, 11: 166, 45: 167, 14: 168, 81: 169, 47: 170}, 
 'ps_car_12': 171, 
 'ps_car_13': 172, 
 'ps_car_14': 173, 
 'ps_car_15': 174, 
 'ps_ind_01': {2: 175, 1: 176, 5: 177, 0: 178, 4: 179, 3: 180, 6: 181, 7: 182}, 
 'ps_ind_02_cat': {2: 183, 1: 184, 4: 185, 3: 186, -1: 187}, 
 'ps_ind_03': {5: 188, 7: 189, 9: 190, 2: 191, 0: 192, 4: 193, 3: 194, 1: 195, 11: 196, 6: 197, 8: 198, 10: 199}, 
 'ps_ind_04_cat': {1: 200, 0: 201, -1: 202}, 
 'ps_ind_05_cat': {0: 203, 1: 204, 4: 205, 3: 206, 6: 207, 5: 208, -1: 209, 2: 210}, 
 'ps_ind_06_bin': {0: 211, 1: 212}, 
 'ps_ind_07_bin': {1: 213, 0: 214},
  'ps_ind_08_bin': {0: 215, 1: 216}, 
  'ps_ind_09_bin': {0: 217, 1: 218}, 
  'ps_ind_10_bin': {0: 219, 1: 220},
   'ps_ind_11_bin': {0: 221, 1: 222}, 'ps_ind_12_bin': {0: 223, 1: 224}, 'ps_ind_13_bin': {0: 225, 1: 226}, 'ps_ind_14': {0: 227, 1: 228, 2: 229, 3: 230}, 'ps_ind_15': {11: 231, 3: 232, 12: 233, 8: 234, 9: 235, 6: 236, 13: 237, 4: 238, 10: 239, 5: 240, 7: 241, 2: 242, 0: 243, 1: 244}, 'ps_ind_16_bin': {0: 245, 1: 246}, 'ps_ind_17_bin': {1: 247, 0: 248}, 'ps_ind_18_bin': {0: 249, 1: 250}, 'ps_reg_01': 251, 'ps_reg_02': 252, 'ps_reg_03': 253}
   ps_ind_01  ps_ind_02_cat  ps_ind_03  ...  ps_car_13  ps_car_14  ps_car_15
"""
# 对训练集进行转化
# print(dfTrain.columns)

train_y = dfTrain[['target']].values.tolist()
# drop 函数删除行/列，inplace参数选择是否直接替换原数组
dfTrain.drop(['target','id'],axis=1,inplace=True)

train_feature_index = dfTrain.copy()
train_feature_value = dfTrain.copy()

for col in train_feature_index.columns:
    if col in IGNORE_COLS:
        train_feature_index.drop(col,axis=1,inplace=True)
        train_feature_value.drop(col,axis=1,inplace=True)
        continue
    elif col in NUMERIC_COLS:
        train_feature_index[col] = feature_dict[col]  # 如果当前为连续型变量，则根据feature_dict中的索引值，放入当前连续性数据中
    else:
        train_feature_index[col] = train_feature_index[col].map(feature_dict[col])
        train_feature_value[col] = 1             # 如果当前为离散型变量，则将其对应的索引值设置为1


# 对测试集进行转化
test_tds = dfTest['id'].values.tolist()
dfTest.drop(['id'],axis=1,inplace=True)
test_feature_index = dfTest.copy()
test_feature_value = dfTest.copy()

for col in test_feature_index.columns:
    if col in IGNORE_COLS:
        test_feature_index.drop(col,axis=1,inplace=True)
        test_feature_value.drop(col,axis=1,inplace=True)
        continue
    elif col in NUMERIC_COLS:
        test_feature_index[col] = test_feature_index[col]
    else:
        test_feature_index[col] = test_feature_index[col].map(feature_dict[col])
        test_feature_value[col] = 1

print(train_feature_index.head(3))
print(train_feature_value.head(3))
"""
   ps_ind_01  ps_ind_02_cat  ps_ind_03  ...  ps_car_13  ps_car_14  ps_car_15
0        175            183        188  ...        172        173        174
1        176            184        189  ...        172        173        174
2        177            185        190  ...        172        173        174


   ps_ind_01  ps_ind_02_cat  ps_ind_03  ...  ps_car_13  ps_car_14  ps_car_15
0          1              1          1  ...   0.883679   0.370810   3.605551
1          1              1          1  ...   0.618817   0.388716   2.449490
2          1              1          1  ...   0.641586   0.347275   3.316625

"""


# ========================================= 3、定义模型参数及输入 =========================================
import numpy as np
# 模型参数
dfm_params = {
    "use_fm":True,
    "use_deep":True,
    "embedding_size":8,
    # "embedding_size":16,
    "dropout_fm":[1.0,1.0],
    "deep_layers":[32,32],
    # "deep_layers":[64,64],
    "dropout_deep":[0.5,0.5,0.5],
    "deep_layer_activation":tf.nn.relu,
    "epoch":30,
    "batch_size":1024,
    "learning_rate":0.001,
    "optimizer":"adam",
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "l2_reg":0.01,
    "verbose":True,
    "eval_metric":'gini_norm',
    "random_seed":3
}

dfm_params['feature_size'] = total_feature
dfm_params['field_size'] = len(train_feature_index.columns)

# 开始建立模型
# 训练模型的输入，3个，特征索引、特征值、label
feat_index = tf.placeholder(tf.int32,shape=[None,None],name='feat_index')
feat_value = tf.placeholder(tf.float32,shape=[None,None],name='feat_value')
label = tf.placeholder(tf.float32,shape=[None,1],name='label')

# 定义模型中的Weights
weights = dict()

# embeddings
# 每个特征对应的embedding，大小为 feature_size * embedding_size
weights['feature_embeddings'] = tf.Variable(
    tf.random_normal([dfm_params['feature_size'],dfm_params['embedding_size']],0.0,0.01),name='feature_embeddings'
)
# FM部分计算时用到的一次项的权重参数,相当于embedding_size为1的embedding table，大小为 feature_size * 1
weights['feature_bias'] = tf.Variable(tf.random_normal(shape=[dfm_params['feature_size'],1],mean=0.0,stddev=1.0),name='feature_bias')

# deep layers
num_layers = len(dfm_params['deep_layers'])
input_size = dfm_params['field_size'] * dfm_params['embedding_size']
# 正态分布初始化方法参数
glorot = np.sqrt(2.0/(input_size + dfm_params['deep_layers'][0]))

weights['layer_0'] = tf.Variable(
    np.random.normal(loc=0,scale=glorot,size=(input_size,dfm_params['deep_layers'][0])),dtype=np.float32
)
weights['bias_0'] = tf.Variable(
    np.random.normal(loc=0,scale=glorot,size=(1,dfm_params['deep_layers'][0])),dtype=np.float32
)

for i in range(1,num_layers):
    glorot = np.sqrt(2.0 / (dfm_params['deep_layers'][i-1] + dfm_params['deep_layers'][i]))
    weights["layer_%d" % i] = tf.Variable(
        # layers[i-1] * layers[i]
        np.random.normal(loc=0,scale=glorot,size=(dfm_params['deep_layers'][i-1],dfm_params['deep_layers'][i])),dtype=np.float32
    )
    weights["bias_%d" % i] = tf.Variable(
        # 1 * layers[i]
        np.random.normal(loc=0,scale=glorot,size=(1,dfm_params['deep_layers'][i])),dtype=np.float32
    )

# final concat project layer 最后的连接投影层
if dfm_params['use_fm'] and dfm_params['use_deep']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size'] + dfm_params['deep_layers'][-1]
elif dfm_params['use_fm']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size']
elif dfm_params['use_deep']:
    input_size = dfm_params['deep_layers'][-1]

glorot = np.sqrt(2.0/(input_size + 1))
weights['concat_project'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32)
weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32)

# ========================================= 4、嵌入层 =========================================
# 根据特征索引得到对应特征的embedding
embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'],feat_index)
reshaped_feat_value = tf.reshape(feat_value,shape=[-1,dfm_params['field_size'],1])
# 根据FM的公式，得到对应的embedding后，乘上对应的特征值。此时的embeddings就是v*x
embeddings = tf.multiply(embeddings,reshaped_feat_value)

# ========================================= 5、FM部分 =========================================
# 根据FM公式的转化，<vi,vj>xixj = 1/2 * ∑ ( (∑vx)^2 - v^2 * x^2 )
fm_first_order = tf.nn.embedding_lookup(weights['feature_bias'],feat_index)
fm_first_order = tf.reduce_sum(tf.multiply(fm_first_order,reshaped_feat_value),2)

summed_feature_emb = tf.reduce_sum(embeddings,1)
summed_feature_emb_square = tf.square(summed_feature_emb)
squared_features_emb = tf.square(embeddings)
squared_sum_features_emb = tf.reduce_sum(squared_features_emb,1)
# = 1/2 * ((∑vx)^2 - ∑(vx)^2)
fm_second_order = 0.5 * tf.subtract(summed_feature_emb_square,squared_sum_features_emb)

# ========================================= 6、Deep部分 =========================================
y_deep = tf.reshape(embeddings,shape=[-1,dfm_params['field_size'] * dfm_params['embedding_size']])
# 几层全连接的神经网络
for i in range(0,len(dfm_params['deep_layers'])):
    y_deep = tf.add(tf.matmul(y_deep,weights["layer_%d" %i]),weights["bias_%d" %i])
    y_deep = tf.nn.relu(y_deep)

# ========================================= 7、输出部分 =========================================
if dfm_params['use_fm'] and dfm_params['use_deep']:
    concat_input = tf.concat([fm_first_order,fm_second_order,y_deep],axis=1)
elif dfm_params['use_fm']:
    concat_input = tf.concat([fm_first_order,fm_second_order],axis=1)
elif dfm_params['use_deep']:
    concat_input = y_deep
# 估计的输出label
out = tf.nn.sigmoid(tf.add(tf.matmul(concat_input,weights['concat_project']) , weights['concat_bias']))

# ========================================= 8、loss和optimizer =========================================
# log_loss 交叉熵 二分类问题首选【手动先求sigmoid】
# 此处应该能直接用tf.nn.sigmoid_cross_entropy_with_logits 先sigmoid再求交叉熵，可以省去上面sigmoid计算行
loss = tf.losses.log_loss(labels=tf.reshape(label,(-1,1)),predictions=out)
optimizer = tf.train.AdamOptimizer(learning_rate=dfm_params['learning_rate'],beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(loss)


# ========================================= 9、训练测试 =========================================
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        epoch_loss,_ = sess.run([loss,optimizer],feed_dict={feat_index:train_feature_index,
                                                            feat_value:train_feature_value,label:train_y})
        print("epoch %s,loss is %s " % (str(i),str(epoch_loss)))