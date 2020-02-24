#-*- coding:utf-8 -*-

#训练wide&deep模型
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os


def get_feature_column():
    '''
    划分特征，有两个返回
    :return:
    '''
    #将连续特征本身的数值存放在deep侧，离散化之后的值放在wide侧
    age=tf.feature_column.numeric_column('age')
    education_num=tf.feature_column.numeric_column('education-num')
    capital_gain=tf.feature_column.numeric_column('capital-gain')
    capital_loss=tf.feature_column.numeric_column('capital-loss')
    hours_per_week=tf.feature_column.numeric_column('hours-per-week')

    #将离散特征首先进行hash放入wide侧，再进行embedding放入deep侧
    work_class=tf.feature_column.categorical_column_with_hash_bucket('workclass',hash_bucket_size=512)
    education=tf.feature_column.categorical_column_with_hash_bucket('education',hash_bucket_size=512)
    marital_status=tf.feature_column.categorical_column_with_hash_bucket('marital-status',hash_bucket_size=512)
    occupation=tf.feature_column.categorical_column_with_hash_bucket('occupation',hash_bucket_size=512)
    relationship=tf.feature_column.categorical_column_with_hash_bucket('relationship',hash_bucket_size=512)

    #对连续特征进行离散化
    age_bucket=tf.feature_column.bucketized_column(age,boundaries=[18,25,30,35,40,45,50,55,60,65])
    gain_bucket=tf.feature_column.bucketized_column(capital_gain,boundaries=[0,1000,2000,3000,10000])
    loss_bucket=tf.feature_column.bucketized_column(capital_loss,boundaries=[0,1000,2000,3000,5000])

    #处理交叉特征
    cross_columns=[
        tf.feature_column.crossed_column([age_bucket,gain_bucket],hash_bucket_size=36),
        tf.feature_column.crossed_column([gain_bucket,loss_bucket],hash_bucket_size=16)
    ]

    base_columns=[work_class,education,marital_status,occupation,relationship,age_bucket,gain_bucket,loss_bucket]
    #wide一侧的特征包含所有的离散特征以及交叉特征
    wide_columns=base_columns+cross_columns
    #deep一侧的特征包含所有的连续特征以及离散化后的特征的embedding特征
    deep_columns=[age,education_num,capital_gain,capital_loss,hours_per_week,
                  tf.feature_column.embedding_column(work_class,9),
                  tf.feature_column.embedding_column(education, 9),
                  tf.feature_column.embedding_column(marital_status, 9),
                  tf.feature_column.embedding_column(occupation, 9),
                  tf.feature_column.embedding_column(relationship, 9),
                  ]
    return wide_columns,deep_columns


def build_model_estimator(wide_columns,deep_columns,model_folder):
    '''
    构建wide&deep模型
    :param wide_columns:
    :param deep_columns:
    :param model_folder: 模型输出的文件夹
    :return: 模型的实例和serving_input_fn
    '''
    model_es=tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_folder,
        linear_feature_columns=wide_columns,
        linear_optimizer=tf.train.FtrlOptimizer(0.1,l2_regularization_strength=1.0),
        dnn_feature_columns=deep_columns,
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,l1_regularization_strength=0.001,l2_regularization_strength=0.001),
        dnn_hidden_units=[128,64,32,16]
    )
    feature_columns=wide_columns+deep_columns
    feature_spec=tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_fn=(tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    return model_es,serving_input_fn


def input_fn(data_file,re_time,shuffle,batch_num,predict):
    '''

    :param data_file: 训练文件或者测试文件
    :param re_time: 根据模型参数估计需要多少样本，为了防止过拟合，会重复采样
    :param shuffle:
    :param batch_num:
    :param predict:
    :return: train_feature,train_label//test_feature
    '''

    _CSV_COLUMN_DEFAULTS=[[0],[''],[0],[''],[0],[''],[''],[''],[''],[''],[0],[0],[0],[''],['']]
    _CSV_COLUMNS=[
            'age','workclass','fnlwgt','education','education-num',
            'marital-status','occupation','relationship','race','gender',
            'capital-gain','capital-loss','hours-per-week','native-country',
            'label'
        ]

    def parse_csv(value):
        columns=tf.decode_csv(value,record_defaults=_CSV_COLUMN_DEFAULTS)
        #这里返回的特征是一个字典
        features=dict(zip(_CSV_COLUMNS,columns))
        labels=features.pop('label')
        classes=tf.equal(labels,'>50K')
        return features,classes

    def parse_csv_predict(value):
        columns=tf.decode_csv(value,record_defaults=_CSV_COLUMN_DEFAULTS)
        features=dict(zip(_CSV_COLUMNS,columns))
        labels=features.pop('label')
        return features

    #读入文件，过滤掉第一行的特征名称,并且过滤掉带有问号的行
    data_set=tf.data.TextLineDataset(data_file).skip(1).filter(lambda line:tf.not_equal(tf.strings.regex_full_match(line,".*\?.*"),True))
    if shuffle:
        data_set=data_set.shuffle(buffer_size=30000)
    if predict:
        data_set=data_set.map(parse_csv_predict,num_parallel_calls=5)
    else:
        data_set=data_set.map(parse_csv,num_parallel_calls=5)

    #重复采样
    data_set=data_set.repeat(re_time)
    data_set=data_set.batch(batch_num)
    return data_set


def train_wd_model(model_es,train_file,test_file,model_export_folder,serving_input_fn):
    '''

    :param model_es: 模型对象
    :param train_file:
    :param test_file:
    :param model_export_folder: 模型导出的文件夹
    :param serving_input_fn: 辅助模型导出的函数
    :return:
    '''
    model_es.train(input_fn=lambda :input_fn(train_file,3,True,100,False))
    model_es.evaluate(input_fn=lambda :input_fn(test_file,1,False,100,False))
    model_es.export_savedmodel(model_export_folder,serving_input_fn)


def get_test_label(test_file):
    if not os.path.exists(test_file):
        return []
    test_label_list=[]
    fp=open(test_file)
    linenum=0
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        if "?" in line.strip():
            continue
        item=line.strip().split(',')
        label_str=item[-1]
        if label_str=='>50K':
            test_label_list.append(1)
        elif label_str=='<=50K':
            test_label_list.append(0)
        else:
            print("Error")
    fp.close()
    return test_label_list


def get_auc(predict_list,test_label):
    '''

    :param predict_list:
    :param test_label:
    :return:       auc=(sum(pos_index)-pos_num(pos_num+1)/2)/pos_num*neg_num
    '''
    total_list=[]
    for index in range(len(predict_list)):
        predict_score=predict_list[index]
        label=test_label[index]
        total_list.append((label,predict_score))
    sorted_total_list=sorted(total_list,key=lambda ele:ele[1])
    neg_num=0
    pos_num=0
    count=1
    total_pos_index=0
    for instance in sorted_total_list:
        label,predict_score=instance
        if label==0:
            neg_num+=1
        else:
            pos_num+=1
            total_pos_index+=count
        count+=1
    auc_score=(total_pos_index-(pos_num)*(pos_num+1)/2)/(pos_num*neg_num)
    print("auc_score is:" ,auc_score)

def test_model_performance(model_es,test_file):
    '''
    测试模型表现
    :param model_es:
    :param test_file:
    :return:
    '''
    test_label=get_test_label(test_file)
    result=model_es.predict(input_fn=lambda :input_fn(test_file,1,False,100,True))
    predict_list=[]
    for one_res in result:
        if "probabilities" in one_res:
            predict_list.append(one_res['probabilities'][1])
    get_auc(predict_list,test_label)


def run_main(train_file,test_file,model_folder,model_export_folder):
    '''

    :param train_file:
    :param test_file:
    :param model_folder:初始的存放训练模型
    :param model_export_folder:
    :return:
    '''

    #需要决定哪些特征放在wide侧，哪些放在deep侧
    wide_column,deep_column=get_feature_column()
    model_es,serving_input_fn=build_model_estimator(wide_column,deep_column,model_folder)
    train_wd_model(model_es,train_file,test_file,model_export_folder,serving_input_fn)
    test_model_performance(model_es,test_file)


if __name__=='__main__':
    run_main("../data/train1.txt","../data/test1.txt","../data/wd","../data/wd_export")