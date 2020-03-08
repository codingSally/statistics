import numpy as np
from math import log

class DTreeID3(object):
    
    #==========python函数必须先定义后使用=============
    
    # 创建样本数据
    def create_data_set():
        
        data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        
        return data_set, labels
    
    # 计算信息熵
    def calc_shannon_ent(data_set):
        
        num = len(data_set)
        # 为所有的分类类目创建字典
        label_counts = {}
        for feat_vec in data_set:
            current_label = feat_vec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
            label_counts[current_label] += 1
            
        # 计算信息熵
        shannon_ent = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num
            shannon_ent = shannon_ent - prob * log(prob,2)
            
        return shannon_ent
    
    # 返回特征值等于value的子数据集，且该数据集不包含列（特征）axis
    def split_data_set(data_set, axis, value):
        ret_data_set = []
        for feat_vec in data_set:
            if feat_vec[axis] == value:
                reduce_feat_vec = feat_vec[:axis]
                reduce_feat_vec.extend(feat_vec[axis + 1:])
                ret_data_set.append(reduce_feat_vec)
        return ret_data_set
    
    # 按照最大信息增益划分数据
    def choose_best_feature_to_split(data_set):
        # 1. 求特征个数
        num_feature = len(data_set[0]) - 1
        # 2. 求经验熵
        base_entropy = calc_shannon_ent(data_set)
        # 3. 子集香农熵求和
        best_info_gain = 0
        best_feature_idx = -1
        # 计算某个特征的所有值
        for feature_idx in range(num_feature):
            feature_val_list = [number[feature_idx] for number in data_set]
            # 获取无重复的属性值
            unique_feature_val_list = set(feature_val_list)
            new_entropy = 0
            for feature_val in unique_feature_val_list:
                # split_data_set为什么是返回不包含当前值的
                # 其实就是通过求减去这个值剩余的，来求得需要值  -- 因为反正最后要求和
                sub_data_set = split_data_set(data_set, feature_idx, feature_val)
                # p(t)
                prob = len(sub_data_set) / float(len(data_set)) 
                # 对各子集香农熵求和
                new_entropy += prob * calc_shannon_ent(sub_data_set) 
             
            # 计算信息增益g(D,A)=H(D)-H(D|A)
            info_gain = base_entropy - new_entropy
            # 最大信息增益
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_idx = feature_idx
                
        return best_feature_idx

    #  统计每个类别出现的次数，并按大到小排序，返回出现次数最大的类别标签
    def majority_cnt(class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
            
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reversed=True)
        
        return sorted_class_count[0][0]
    
    # 构建决策树
    def create_tree(data_set, labels):
        # 样本分类信息
        class_list = [sample[-1] for sample in data_set]
        # 类别相同，停止划分
        if class_list.count(class_list[-1]) == len(class_list):
            return class_list[-1]
        
        # 如果长度为1，返回出现次数最多的类被  
        # == 有疑问？ -- 这里就是特征为空集的那个条件：即特征为空集的时候，返回分类中，类别多的那个
        if len(class_list[0]) == 1:
            return majority_cnt((class_list))
    
        # 按照信息增益最高，选取分类属性特征
        # 1. 返回分类的特征的数组索引
        best_feature_idx = choose_best_feature_to_split(data_set) 
        # 2. 该特征的label
        best_feat_label = labels[best_feature_idx]
        # 3. 构建树的字典 -- 有疑问,代码本身有疑问{best_feat_label: {}}
        my_tree = {best_feat_label: {}}
        # 4. 从label的list中删除该label，相当于待划分的子标签集
        del (labels[best_feature_idx])
        
        feature_values = [example[best_feature_idx] for example in data_set]
        unique_feature_values = set(feature_values)
        for feature_value in unique_feature_values:
            # 子集合
            sub_labels = labels[:]
            # 构建数据的子数据集，并进行递归
            sub_data_set = split_data_set(data_set, best_feature_idx, feature_value)
            my_tree[best_feat_label][feature_value] = create_tree(sub_data_set, sub_labels)
        return my_tree
          
    # 决策树分类
    def classify(input_tree, feat_labels, test_vec):
        # 1. 获取树的第一特征属性
        first_str = list(input_tree.keys())[0]
        # 2. 树的分子，子集合dict
        second_dict = input_tree[first_str]
        # 3. 获取决策树第一层在feat_labels中的位置
        feat_index = feat_labels.index(first_str)
        
        for key in second_dict.keys():
            if test_vec[feat_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = classify(second_dict[key], feat_labels, test_vec)
                else:
                    class_label = second_dict[key]
                    
                return class_label

        
if __name__ == "__main__": #如果模块是被直接运行的，则代码块被运行，如果模块是被导入的，则代码块不被运行。
    main(DTreeID3)       
        
    
    
    
    
    
    
    
    
    
    
    
    
            
        