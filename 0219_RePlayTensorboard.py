from __future__ import print_function
import tensorflow as tf 
import numpy as np 

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
	layer_name = 'layer%s' % n_layer # python 语法，https://www.cnblogs.com/wh-ff-ly520/p/9390855.html
									 # 最后这句话效果为：layer_name = layern_layer(字符串拼接)
	with tf.name_scope(layer_name):	 # tf.name_scope 理解为tensorboard当中的命名空间
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W') # W 大写是表示矩阵
			tf.summary.histogram(layer_name + '/weights',Weights)	# tf.summary.histogram(tags, values, collections=None, name=None) 
																	# https://www.cnblogs.com/lyc-seu/p/8647792.html
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
			tf.summary.histogram(layer_name + '/biases',biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases) 	# tf.matmul（）将矩阵a乘以矩阵b，生成a * b
																	# tf.multiply（）两个矩阵中对应元素各自相乘
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b,)
		tf.summary.histogram(layer_name + '/outputs',outputs)
	return outputs

# 创造真实数据
x_data = np.linspace(-1,1,300)[:,np.newaxis] # 创建等差数列 numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
											 # np.newaxis:插入新维度：[,]：前面 列扩展（一行多列）；后面 行扩展（多行一列）；
noise = np.random.normal(0,0.05,x_data.shape)# 正态分布 numpy.random.normal(loc=0.0, scale=1.0, size=None)  
											 # loc:float 概率分布均值；scale:float 标准差；size:输出的shape形状，默认为None，为一个值；
											 # np.random.randn(size) 表示标准正态分布；
y_data = np.square(x_data) - 0.5 + noise

#  创建一个 placeholder 给神经网络：用来后期进行 feed_dict 数据传入
with tf.name_scope('inputs'): # tf.name_scope是用来tensorboard显示的
	xs = tf.placeholder(tf.float32, [None,1] ,name = 'x_input')	  # tf.placeholder 是用来做占位符，以便后期进行数据传入：feed_dict()
	ys = tf.placeholder(tf.float32, [None,1], name = 'y_input')	  # tf.placeholder( dtype , shape=None , name=None )

# 增加一个隐藏层
#l1 = add_layer(xs,1,10,n_layer=1,activation_function = tf.nn.relu)
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# 增加输出层
# n_layer 只是用来最后起名字区分用的
prediction = add_layer(l1, 10, 1, n_layer=2,activation_function= None) # 小心逗号不要用中文的逗号


# loss 误差计算：
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1] )) # reduction_indices=[1]:将多列压缩为1列；[0]则表示n行压缩为1行
	tf.summary.scalar('loss',loss) # 用来显示标量信息: tf.summary.scalar(tags, values, collections=None, name=None)
								   # 具体更深入的理解，找不到相关博客说明，待多理解
# 训练train部分
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all() # tf.summaries.merge_all(key='summaries')
								# merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
writer = tf.summary.FileWriter("logs/",sess.graph) # 指定一个文件用来保存图 tf.summary.FileWritter(path,sess.graph) 
												   # 可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	sess.run(train_step, feed_dict = {xs: x_data, ys:y_data})
	if i%50 == 0:
		result = sess.run(merged,feed_dict = {xs:x_data, ys:y_data})
		writer.add_summary(result,i) # 可以调用tf.summary.FileWriter 的 add_summary（）方法将训练过程数据保存在filewriter指定的文件中




















