# 基于Datwhale数据挖掘第7期组队学习
## 这份数据集是金融数据（非原始数据，已经处理过了），我们要做的是预测贷款用户是否会逾期。表格中 "status" 是结果标签：0表示未逾期，1表示逾期。
## 任务一：对数据进行探索和分析。包括但不限于数据类型的分析、无关特征删除、数据类型转换、缺失值处理。
#### 在做任务一时我用了互信息法删除了一些特征总共剩余52个特征，用随机森林auc达到84.22%
## 任务二：特征工程：包括但不限于IV值和随机森林等进行特征选择。
### 1、分箱计算IV值，IV值用来衡量自变量预测能力，IV值越大，说明自变量的预测能力越强，对因变量的预测贡献越大，变量越重要。
### 2、随机森林进行特征选择，主要用到随机森林的属性feature_imoportances，能够查看各个特征对模型的重要性。
#### 计算IV值，选择介于0.02~0.5之间的IV值，筛选出36个特征，用随机森林取出前20个feature_imoportances重要的变量，最终auc达到81.07%。
## 任务三：模型构建。用逻辑回归、svm和决策树；随机森林和XGBoost进行模型构建，评分方式任意，如准确率等。（不需要考虑模型调参）
### 因样本是不平衡样本，此处主要考虑auc的值。XGBoost的auc0.8332,随机森林的auc0.8248，SVC的auc0.7586,决策树的auc0.7570,逻辑回归的auc0.6843.
## 任务四：记录5个模型（逻辑回归、SVM、决策树、随机森林、XGBoost）关于accuracy、precision，recall和F1-score、auc值的评分表格，并画出ROC曲线。
## 任务五：使用网格搜索法对5个模型进行调优（调参时采用五折交叉验证的方式），并进行模型评估，记得展示代码的运行结果。
### 调参一个耗费时间的过程，XGBoost的auc0.8337,随机森林的auc0.8454，SVC的auc0.7586,决策树的auc0.7744,逻辑回归的auc0.6924.事实上，随机森林只调了n_estimators，可能出现了过拟合。
## 任务五：模型融合，模型融合方式任意，并结合Task5给出你的最优结果。例如Stacking融合，用你目前评分最高的模型作为基准模型，和其他模型进行stacking融合，得到最终模型及评分结果。

 ### 1、StackingClassifier(classifiers, meta_classifier, use_probas=False, drop_last_proba=False, average_probas=False, verbose=0, use_features_in_secondary=False, store_train_meta_features=False, use_clones=True)
 ### 2、classifiers : 基分类器，数组形式，[cl1, cl2, cl3]. 每个基分类器的属性被存储在类属性 self.clfs_.
### 3、meta_classifier : 目标分类器，即将前面分类器合起来的分类器
### 4、use_probas : bool (default: False) ，如果设置为True， 那么目标分类器的输入就是前面分类输出的类别概率值而不是类别标签
### 5、average_probas : bool (default: False)，用来设置上一个参数当使用概率值输出的时候是否使用平均值。
### 6、verbose : int, optional (default=0)。用来控制使用过程中的日志输出，当 verbose = 0时，什么也不输出， verbose = 1，输出回归器的序号和名字。verbose = 2，输出详细的参数信息。verbose > 2, 自动将verbose设置为小于2的，verbose -2.
### 7、use_features_in_secondary : bool (default: False). 如果设置为True，那么最终的目标分类器就被基分类器产生的数据和最初的数据集同时训练。如果设置为False，最终的分类器只会使用基分类器产生的数据训练。
