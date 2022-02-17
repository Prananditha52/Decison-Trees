from _csv import reader
import matplotlib.pyplot as plt
import pandas as pd


gini_in={}
info_gain={}
b_scoreG=[]
b_valueG=[]
depth_c=3
count_acc=0
def split_lr(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Gini_index calculating
def gini_impurity(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes: # calculating the GINI Impurity
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    if len(gini_in) != 0: #calculating the information gain
        gini = gini_in[b_valueG[len(b_valueG)-1]]-gini

    return gini,score


# Spliting the dataset
def spliting(dataset):
    class_count_left=0
    class_count_right=0
    class_values = list(set(row[-1] for row in dataset))
    row_index, row_value, gini_score, split_groups,score_t = 999, 999, 999, None,None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups =split_lr(index, row[index], dataset)
            gini , score= gini_impurity(groups, class_values)
            if len(gini_in)==0:
                if gini < gini_score: # for the root node
                    row_index, row_value, gini_score, split_groups,score_t  = index, row[index], gini, groups,score
            else:
                gini_score=0 #getting the values of information gain for each split and selecting the once with the higher value
                if gini>gini_score:
                    row_index, row_value, gini_score, split_groups, score_t = index, row[index], gini, groups, score

    gini_in[row_value]=gini_score
    b_scoreG.append(score_t)
    b_valueG.append(row_value)
    # if len(gini_in)==0:
    #     info_gain[]
    # print("gini",gini_in[b_scoreG[0]])
    left,right=split_groups

    # outcomes = [row[-1] for row in left]
    # class_count_left=outcomes.count()
    # outcomes = [row[-1] for row in right]
    # class_count_right = outcomes.count()

    return {'index': row_index, 'value': row_value, 'groups': split_groups,'count_left':0,'count_right':0}


#terminal node
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    count=0
    for i in outcomes:
        if i==max(set(outcomes), key=outcomes.count):
            count+=1
    return max(set(outcomes), key=outcomes.count),count


# Create child splits for a node or make terminal, pruning based on the max depth and minimum size is done in this function
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        val_left,counter=to_terminal(left + right)
        node['left'],node['count_left'] = node['right'],node['count_right'] = val_left,counter
        return
    # check for max depth
    if depth >= max_depth:
        val_left, counter_left = to_terminal(right)
        val_right,counter_right=to_terminal(right)
        node['left'],node['count_left'], node['right'],node['count_right'] = val_left, counter_left,val_right,counter_right
        return
    # process left child
    if len(left) <= min_size:
        val_left, counter_left = to_terminal(right)
        node['left'],node['count_left'] = val_left, counter_left
    else:
        node['left'] = spliting(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        val_right, counter_right = to_terminal(right)
        node['right'],node['counter_right'] =  val_right, counter_right
    else:
        node['right'] = spliting(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = spliting(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    test = float(row[node['index']])
    if  float(row[node['index']]) < float(node['value']):
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):

        print('%s[X%s < %.3f]' % ((depth * ' ', (node['index'] + 1), float(node['value']))))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))

def prun(copy_tree, depth, acc, dataset, class_val,copy_tree1,dir,acc_count):

    if (copy_tree['right']==class_val[0] or  copy_tree['right']==class_val[1] or copy_tree['right']==class_val[2]) and( copy_tree['left']==class_val[0] or  copy_tree['left']==class_val[1] or copy_tree['left']==class_val[2]):
        # copy_tree['right']='None'
        # print('----------------------------------opt')
        # print_tree(copy_tree)
        if (int(copy_tree['count_left'])>int(copy_tree['count_right'])) or int(copy_tree['count_left'])==int(copy_tree['count_right']):
            if dir==2 or dir==0:
                copy_tree1['right'] = copy_tree1['right']['left']
            else:
                copy_tree1['left'] = copy_tree1['left']['left']

        else:
            if dir == 2 or dir==0:
                copy_tree1['right'] = copy_tree1['right']['right']
            else:
                copy_tree1['left'] = copy_tree1['left']['right']
    elif (copy_tree['right']==class_val[0] or  copy_tree['right']==class_val[1] or copy_tree['right']==class_val[2]) and isinstance(copy_tree, dict):
        # print("new left------------------------------------------------------------------")
        # print_tree(copy_tree)
        copy_tree1 = copy_tree
        prun(copy_tree['left'], depth - 1, acc, dataset, class_val, copy_tree1,1,acc_count)
    elif (copy_tree['left']==class_val[0] or  copy_tree['left']==class_val[1] or copy_tree['left']==class_val[2]) and isinstance(copy_tree, dict):
        # print("new- right-----------------------------------------------------------------")
        # print_tree(copy_tree)
        copy_tree1 = copy_tree
        prun(copy_tree['right'], depth - 1, acc, dataset, class_val, copy_tree1,2,acc_count)
    elif isinstance(copy_tree, dict):
        # print("new------------------------------------------------------------------")
        # print_tree(copy_tree)
        copy_tree1=copy_tree
        prun(copy_tree['right'],depth-1,acc,dataset,class_val,copy_tree1,0,acc_count)

    accura_prun=accuracy(dataset,copy_tree)
    while accura_prun >= acc:
        if accura_prun==acc:
            acc_count+=1

        # print("new ittr")
        # print("new acc", accura_prun)
        if acc_count==2:
            break
        else:
            prun(copy_tree,depth_c,acc,dataset,class_val,copy_tree1,0,acc_count)
    return copy_tree1,accura_prun

def accuracy(dataset,tree):
    correct_preditct = 0
    wrong_preditct = 0
    accuracy = 0
    for row in dataset:
        prediction = predict(tree, row)
        if prediction == row[-1]:
            correct_preditct += 1
        else:
            wrong_preditct += 1
        # print('Expected=%s, Got=%s' % (row[-1], prediction))
    accuracy = correct_preditct / (correct_preditct + wrong_preditct)
    # print("accuracy", accuracy * 100)
    return accuracy*100

filename = 'Iris.csv'
dataset = list()
with open(filename, 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)
dataset.pop(0)

class_val= list(set(row[-1] for row in dataset))
tree = build_tree(dataset, depth_c,1 )
print_tree(tree)
accur=accuracy(dataset,tree)
print("accuracy of a full tree",accur)
Copy_tree=tree
# print(tree)
p_tree,acc_prun=prun(tree,depth_c,accur,dataset,class_val,Copy_tree,0,0)

print("----------------------------------------------prun_tree")
print_tree(p_tree)
print("accuracy of pruned tree",acc_prun)

# print(p_tree)
X=pd.DataFrame(dataset).iloc[:,:-1]
y=pd.DataFrame(dataset).iloc[:,-1]

# print("class_val",class_val)
plt.figure(figsize=(12,12))
for row in dataset:
    if (class_val[0]==row[-1]):
        plt.scatter(row[0],row[1],color="red")
    elif (class_val[1]==row[-1]):
        plt.scatter(row[0],row[1],color="blue")
    else:
        plt.scatter(row[0], row[1], color="black")
plt.show()

print("best_split")
print(gini_in)