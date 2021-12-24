
"""
Use mse as evaluation metric
"""
import pandas as pd
import pickle
import os
import random
import numpy as np
from math import exp
import time


class DataProcessing:
    def __init__(self):
        pass

    
    # 对用进行有游戏行为和无游戏行为的标记
    def get_pos_neg_item(self, file_path="data/ratings1.csv"):
        if not os.path.exists("data/lfm_items.csv"):
            self.items_dict_path = "data/lfm_items.csv"
            self.uiscores = pd.read_csv(file_path)
            self.user_ids = set(self.uiscores["UserID"].values)
            self.item_ids = set(self.uiscores["itemID"].values)
            self.items_dict = {user_id: self.get_one(user_id) for user_id in list(self.user_ids)}

            fw = open(self.items_dict_path, "wb")
            pickle.dump(self.items_dict, fw)
            fw.close()

    # 定义单个用户的正负向
    # 有无游戏行为
    def get_one(self, user_id):
        print("为用户%s准备正负向数据。。" % user_id)
        pos_item_ids = set(self.uiscores[self.uiscores["UserID"] == user_id]["itemID"])
        # 差集
        neg_item_ids = self.item_ids ^ pos_item_ids
        neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]
        item_dict = {}
        for item in pos_item_ids: item_dict[item] = 1
        for item in neg_item_ids: item_dict[item] = 0
        return item_dict


class LFM:
    def __init__(self):
        self.class_count = 5
        self.iter_count = 5
        self.lr = 0.02
        self.lam = 0.01
        self._init_model()

    def _init_model(self):
        file_path = "data/ratings1.csv"
        pos_neg_path = "data/lfm_items.csv"

        self.uisceores = pd.read_csv(file_path)
        self.user_ids = set(self.uisceores["UserID"].values)
        self.item_ids = set(self.uisceores["itemID"].values)
        self.items_dict = pickle.load(open(pos_neg_path, "rb"))

        array_p = np.random.randn(len(self.user_ids), self.class_count)
        array_q = np.random.randn(len(self.item_ids), self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))

    # 计算用户 userid 对itemid的兴趣度
    def _predict(self, user_id, item_id):
        try:
            p = np.mat(self.p.loc[user_id].values)

            q = np.mat(self.q.loc[item_id].values).T
            r = (p * q).sum()
            # 借助 sigmod函数，转化为是否感兴趣
            logit = 1.0 / (1 + exp(-r))
            return logit
        except:
            print(item_id)

            print(self.q.loc[item_id])
            return

    # 使用mse作为损失
    def _loss(self, user_id, item_id, y, step):
        e = y - self._predict(user_id, item_id)
        return e

    # 使用梯度下降算法 求解参数，同时使用L2正则化防止过拟合
    def _optimize(self, user_id, item_id, e):
        gradient_p = -e * self.q.loc[item_id].values
        l2_p = self.lam * self.p.loc[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -e * self.p.loc[user_id].values
        l2_q = self.lam * self.q.loc[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    # 训练模型，每次迭代都要降低学习率，刚开始由于离嘴有点较远，因此下降较快，当到达一定程度后，就要降低学习率
    def train(self):
        for step in range(0, self.iter_count):
            time.sleep(30)
            for user_id, item_dict in self.items_dict.items():
                print("Step: {}, user_id: {}".format(step, user_id))
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    e = self._loss(user_id, item_id, item_dict[item_id], step)
                    self._optimize(user_id, item_id, e)
            self.lr *= 0.9
        self.save()

    def save(self):
        f = open("data/lfm.model", "wb")
        pickle.dump((self.p, self.q), f)
        f.close()

    def load(self):
        f = open("data/lfm.model", "rb")
        self.p, self.q = pickle.load(f)
        f.close()

    def predict(self, user_id, top_n=10):
        self.load()
        user_item_ids = set(self.uisceores[self.uisceores["UserID"] == user_id]["itemID"])
        other_item_ids = self.item_ids ^ user_item_ids
        interest_list = [self._predict(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(other_item_ids, interest_list), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def evaluate(self):
        self.load()
        users = random.sample(self.user_ids,10)
        user_dict = {}
        for user in users:
            user_item_ids = set(self.uisceores[self.uisceores["UserID"] == user]["itemID"])
            _sum = 0.0
            for item_id in user_item_ids:
                p = np.mat(self.p.ix[user].values)
                q = np.mat(self.q.ix[item_id].values)
                _r = (p*q).sum()
                r = self.uisceores[(self.uisceores["UserID"] == user) & (self.uisceores["itemID"]==item_id)]["Ratings"].values[0]
                _sum += abs(r - _r)
            user_dict[user] = _sum / len(user_item_ids)
        print(sum(user_dict.values())/len(user_dict))



if __name__ == '__main__':
    # 数据处理
    # dp = DataProcessing()
    # dp.get_pos_neg_item()

    lfm = LFM()
    # 训练
    # lfm.train()
    # mse
    lfm.evaluate()
