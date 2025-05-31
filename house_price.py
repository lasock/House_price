import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
import xgboost as xgb



# 加载数据

train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']
test_ids = test_data["Id"]


# 数据预处理

X = X.drop(columns=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature','Id'])
test_data = test_data.drop(columns=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature','Id'])

## 分别处理数值型和类别型
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

## 填充NA值
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])
test_data[num_cols] = test_data[num_cols].fillna(test_data[num_cols].median())
test_data[cat_cols] = test_data[cat_cols].fillna(test_data[cat_cols].mode().iloc[0])
# print(X.shape,test_data.shape)



#特征工程

## 添加新特征：房屋总面积 = 1层面积 + 2层面积
X["TotalSF"] = X["1stFlrSF"] + X["2ndFlrSF"]
test_data["TotalSF"] = test_data["1stFlrSF"] + test_data["2ndFlrSF"]

## 对类别型列进行独热编码
X = pd.get_dummies(X[cat_cols])
test_data = pd.get_dummies(test_data[cat_cols])
# print(X.shape,test_data.shape)

## 对齐训练集和测试集，进行dataframe的align并运算
X, test_data = X.align(test_data, join='left', axis=1, fill_value=0)
# print(X.shape,test_data.shape)



# 模型训练（XGBoost）

## 划分新训练测试集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

## 定义XGBoost模型
model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)

## 训练模型，每 20 轮训练输出一次评估结果
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)],
    verbose=20
    )


## 评估模型
y_pred = model.predict(X_valid)
### 计算均方根误差rmse
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print(f'新测试集的均方根误差rmse为:{rmse}')



# 预测测试集并生成预测结果

os.makedirs('./Output', exist_ok=True)

## 预测测试集
test_pred = model.predict(test_data)

test_valid = pd.read_csv('Data/sample_submission.csv')['SalePrice']
rmse2 = np.sqrt(mean_squared_error(test_valid, test_pred))
print(f'最终测试集的均方根误差rmse为:{rmse2}')

## 创建提交文件
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_pred
})

submission.to_csv('./Output/submission.csv', index=False)

print('Over!')
