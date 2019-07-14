#-*- coding:utf-8 -*-
import xgboost as xgb
import numpy as np
import sys
from sklearn.linear_model import LogisticRegressionCV as LRCV
from scipy.sparse import coo_matrix
sys.path.append("../util")
import util.get_feature_num as GF

def get_train_data(train_file,feature_num_file):
	"""
	得到训练数据和标签
	:param train_file:
	:param feature_num_file:
	get train data and label for training
	"""
	total_feature_num = int(GF.get_feature_num(feature_num_file))    #共有103维特征
	train_label = np.genfromtxt(train_file,dtype=np.int32,delimiter=",",usecols= -1)
	feature_list = range(total_feature_num)
	train_feature = np.genfromtxt(train_file,dtype=np.int32,delimiter=",",usecols=feature_list)
	return train_feature,train_label

def train_tree_model_core(train_mat,tree_depth,tree_num,learning_rate):
	"""
	:param train_mat: train data and label
	:param tree_depth:
	:param tree_num: total tree num
	:param learning_rate: step_size
	:return:
		Booster
	"""
	para_dict = {"max_depth":tree_depth,"eta":learning_rate,"objective":"reg:linear","silent":1}
	bst = xgb.train(para_dict,train_mat,tree_num)
	return bst


def choose_parameter():
	"""
	生成参数列表
	:return:
		a list as [(tree_depth,tree_num,learning_rate),......]
	"""
	result_list = []
	tree_depth_list = [4,5,6]   #树的深度
	tree_num_list = [10,50,100] #树的总的数目
	learning_rate_list = [0.3,0.5,0.7]  #步长
	for ele_tree_depth in tree_depth_list:
		for ele_tree_num in tree_num_list:
			for ele_learning_rate in learning_rate_list:
				result_list.append((ele_tree_depth,ele_tree_num,ele_learning_rate))
	return result_list


def grid_search(train_mat):
	"""
	选取最优参数
	"""
	para_list = choose_parameter()
	for ele in para_list:
		(tree_depth,tree_num,learning_rate) = ele
		para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
		res = xgb.cv(para_dict, train_mat, tree_num, nfold=5,metrics='auc')
		auc_score = res.loc[tree_num-1,['test-auc-mean']].values[0]  #最后一棵树的结果即最终要比较的结果
		print("tree_depth: %s, tree_num: %s, learning_rate: %s, auc: %f" %(tree_depth,tree_num,learning_rate,auc_score))


def train_tree_model(train_file,feature_num_file,tree_model_file):
	"""
	:param train_file:
	:param feature_num_file: 得到特征的数目
	:param tree_model_file:
	"""
	train_feature,train_label = get_train_data(train_file,feature_num_file)
	train_mat = xgb.DMatrix(train_feature,train_label)
	#grid_search(train_mat)             #选取最优参数，只执行一次，发现当树深4，树的数目100，步长为0.3时 AUC最佳
	#sys.exit()
	tree_num = 10
	tree_depth = 4
	learning_rate = 0.3
	bst = train_tree_model_core(train_mat,tree_depth,tree_num,learning_rate)  #模型实例化输出
	bst.save_model(tree_model_file)

def get_gbdt_and_lr_feature(tree_leaf,tree_num,tree_depth):
	"""
	:param tree_leaf: prediction of the tree model
	:param tree_num: total_tree_num
	:param tree_depth: total_tree_depth
	:return:
		稀疏矩阵存取
	"""
	total_node_num = 2**(tree_depth+1) - 1   #总的节点数目
	yezi_num = 2**(tree_depth)     #叶子节点的数目
	feiyezi_num = total_node_num - yezi_num #非叶子节点的数目
	total_col_num = yezi_num * tree_num   #特征的总的维度
	total_row_num = len(tree_leaf)   #样本个数
	col = []
	row = []
	data = []
	base_row_index = 0
	for one_result in tree_leaf:
		base_col_index = 0
		for fix_index in one_result:
			yezi_index = fix_index - feiyezi_num
			yezi_index = yezi_index if yezi_index >= 0 else 0
			col.append(base_col_index + yezi_index)
			row.append(base_row_index)
			data.append(1)
			base_col_index += yezi_num
		base_row_index += 1
	total_feature_list = coo_matrix((data,(row,col)),shape=(total_row_num,total_col_num))
	return total_feature_list


def get_mix_model_tree_info():
    #最优参数
	tree_depth =4
	tree_num = 10
	learning_rate = 0.3
	result = (tree_depth,tree_num,learning_rate)
	return result

def train_tree_and_lr_model(train_file,feature_num_file,mix_tree_model_file,mix_lr_model_file):
	"""
	GBDT和LR混合模型
	:param train_file: file to training model
	:param feature_num_file: 特征数目
	:param mix_tree_model_file: 树模型
	:param mix_lr_model_file: lr模型
	"""
	train_feature,train_label = get_train_data(train_file,feature_num_file)
	train_mat = xgb.DMatrix(train_feature,train_label)
	(tree_num,tree_depth,learning_rate) = 10,4,0.3 #获取最优参数
	bst = train_tree_model_core(train_mat,tree_depth,tree_num,learning_rate)
	bst.save_model(mix_tree_model_file)    #将树的模型保存
	tree_leaf = bst.predict(train_mat,pred_leaf=True); print(tree_leaf[0])  #样本最后落在哪个节点上
    #提取特征
	total_feature_list = get_gbdt_and_lr_feature(tree_leaf,tree_num,tree_depth)
	#LR模型
	lr_clf = LRCV(Cs=[1.0],penalty='l2',dual=False,tol=0.0001,max_iter=500,cv=5).fit(total_feature_list,train_label)
	scores = list(lr_clf.scores_.values())[0]
	print("diff: %s" % (",".join([str(ele) for ele in scores.mean(axis=0)])))
	print("Accuracy: %s (+-%0.2f)" % (scores.mean(), scores.std() * 2))
	lr_cf = LRCV(Cs=[1], penalty="l2", tol=0.0001, max_iter=500, cv=5,
	             solver="liblinear", scoring="roc_auc").fit(train_feature, train_label)
	scores = list(lr_cf.scores_.values())[0]
	#print(scores)
	print("diff: %s" % (",".join([str(ele) for ele in scores.mean(axis=0)])))
	print("AUC: %s (+-%0.2f)" % (scores.mean(), scores.std() * 2))
	fw = open(mix_lr_model_file,"w+")
	coef = lr_clf.coef_[0]
	fw.write(",".join([str(ele) for ele in coef]))
	fw.close()




if __name__ == "__main__":
	#train_tree_model("../data/train_preprocess.txt","../data/feature_num.txt","../data/tree_model.txt")
	train_tree_and_lr_model("../data/train_preprocess.txt","../data/feature_num.txt","../data/xgb_mix_model","../data/lr_coef_mix_model")