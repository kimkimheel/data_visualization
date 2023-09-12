
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("mysql://root:admin@localhost/anti_killer?charset=utf8")

df0 = pd.read_csv("D:\金融数据\账户交易信息\账户交易信息.csv")

# df1 = pd.read_csv("D:\金融数据\账户交易信息_1\账户交易信息.csv")  # 好像三个数据集是一样的
#
# df2 = pd.read_csv("D:\金融数据\账户交易信息_2\账户交易信息.csv")

df0.to_sql("trans_tab", index=False,  con=engine, if_exists='replace')

df1 = pd.read_csv("D:\金融数据\账户静态信息.csv")
# df1.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)  # 记一下
df1.to_sql("acct_info", index=False,  con=engine, if_exists='replace')