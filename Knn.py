import numpy as np
import heapq
from matplotlib import pyplot as plt
from collections import Counter
print(heapq)

class KDNode(object):
    def __init__(self):
        self.data = None
        self.label = None
        self.left = None
        self.right = None
        self.axis = None
        self.parent = None
        self.brother = None

class KDNodeResult(object):
    def __init__(self,d,node:KDNode):
        self.d = d
        self.data = node.data
        self.label = node.label

    def __lt__(self, other):
        return self.d<other.d

    def __gt__(self, other):
        return self.d>other.d

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)


class KDTree(object):
    def __init__(self, points, labels):
        self.root = self.__build(points, labels)

    def visit(self, tree_root):
        if tree_root is None:
            return
        print(tree_root.data)
        self.visit(tree_root.left)
        self.visit(tree_root.right)

    def __build(self, points, labels, parent=None, l_or_r=None):
        if len(points) == 0:
            return None
        if parent:
            axis = (parent.axis + 1) % points.shape[1]
        else:
            axis = 0
        points_piece = points[:, axis]
        piece_sort_inds = np.argsort(points_piece)
        left_inds = piece_sort_inds[0:int(len(points_piece) / 2)]
        right_inds = piece_sort_inds[int(len(points_piece) / 2) + 1:]
        middle_ind = piece_sort_inds[int(len(points_piece) / 2)]
        left_points = points[left_inds]
        left_labels =labels[left_inds]
        right_points = points[right_inds]
        right_labels = labels[right_inds]
        # 二维数据画图展现KD树构建过程用的代码，对于大于二维的数据无效
        # if axis == 0:
        #     x = [points[middle_ind][axis], points[middle_ind][axis]]
        #     if l_or_r == 'l':
        #         y = [0, parent.data[parent.axis]]
        #     elif l_or_r == 'r':
        #         y = [parent.data[parent.axis], 10]
        #     else:
        #         y = [0, 10]
        # elif axis == 1:
        #     y = [points[middle_ind][axis], points[middle_ind][axis]]
        #     if l_or_r == 'l':
        #         x = [0, parent.data[parent.axis]]
        #     elif l_or_r == 'r':
        #         x = [parent.data[parent.axis], 10]
        #     else:
        #         x = [0, 10]
        #
        # plt.plot(x, y, 'black')
        # plt.annotate('({x},{y})'.format(x=points[middle_ind][0], y=points[middle_ind][1]),
        #              points[middle_ind])
        # plt.pause(0.1)
        node = KDNode()
        node.data = points[middle_ind]
        node.label = labels[middle_ind]
        node.axis = axis
        node.parent = parent
        node.left = self.__build(left_points, left_labels, node, 'l')
        node.right = self.__build(right_points, right_labels, node, 'r')
        if node.left:
            node.left.brother = node.right
        if node.right:
            node.right.brother = node.left
        return node

    def search_k_nearest_neighbors(self, target, k):
        k_nearest_neighbors=[KDNodeResult(float("-inf"), KDNode())]*k
        self.__search_recur(target, self.root, k_nearest_neighbors)
        # 由于堆结构是从树的角度做了排序，而从数组来看还并不是有序的，所以还需要来一次堆排序
        return heapq.nlargest(len(k_nearest_neighbors), k_nearest_neighbors)

    def __search_recur(self, target, tree_root: KDNode,  k_nearest_neighbors):
        #使用优先队列
        #因为是找出最近的k个，也就是说需要每次移除掉堆中最远的一个，而headq是小顶堆实现的
        #故使用距离的负数做为排序标准
        split_axis = tree_root.axis
        child, brother = (tree_root.left, tree_root.right) if target[split_axis] < tree_root.data[split_axis] else (tree_root.right, tree_root.left)
        if child:
            self.__search_recur(target, child, k_nearest_neighbors)
        split_axis_dist = abs(target[split_axis] - tree_root.data[split_axis])  # 距离父区域分割平面的距离
        if -split_axis_dist > k_nearest_neighbors[0].d:  # 与父区域分割平面有交点
            if -np.linalg.norm(target - tree_root.data) > k_nearest_neighbors[0].d:
                heapq.heappushpop(k_nearest_neighbors, KDNodeResult(-np.linalg.norm(target - tree_root.data), tree_root))
            if brother:
                self.__search_recur(target, brother, k_nearest_neighbors)  # 搜索兄弟节点区域



def train(train_input, train_label):
    kd_tree = KDTree(train_input, train_label)
    return kd_tree


def predict(model:KDTree, validate_input, validate_label):
    predict_label = []
    for i in range(len(validate_input)):
        neighbors_label = []
        print("======================",i,"======================")
        results = model.search_k_nearest_neighbors(validate_input[i], 100)
        for r in results:
            neighbors_label.append(r.label)
        counter = Counter(neighbors_label)
        print(counter[1]/(counter[0]+counter[1]))
        predict_label.append(counter[1]/(counter[0]+counter[1]))
    return np.asarray(predict_label, dtype=np.float32)


def call(train_input, train_label,validate_input, validate_label):
    model = train(train_input, train_label)
    return predict(model, validate_input, validate_label)




if __name__ == '__main__':

    k=3
    points = np.random.random([100000, 5]) * 10
    labels = np.random.randint(0, 2, size=[100000])
    target_point = np.asarray([4, 8, 9, 1, 4])

    kd_tree = KDTree(points, labels)
    k_nearest_neighbors = kd_tree.search_k_nearest_neighbors(target_point, k)
    print("result:", k_nearest_neighbors)
    print("--------------------------------")
    k_nearest = [(float('-inf'), None)] * k
    current_nearest_point = None
    current_min_dist = float("inf")
    for i in range(len(points)):
        d = np.linalg.norm(target_point - points[i])
        if d < current_min_dist:
            current_min_dist = d
            current_nearest_point = points[i]
        if -d > k_nearest[0][0]:
            heapq.heappushpop(k_nearest, (-d, (points[i], labels[i])))
    print("result:", heapq.nlargest(len(k_nearest), k_nearest))
    print("nearest result:", current_nearest_point, current_min_dist)
