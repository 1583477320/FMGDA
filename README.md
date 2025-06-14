# FMGDA框架
**update 2025/6/14:** 更新了框架，目前包含两种算法（FMGDA，FMGDA_S）

通过修改run的args.method来寻找训练的算法，目前可选择算法为FMGDA，FMGDA_S，其余超参数可在options中调整
```
if __name__ == "__main__":
    args = args_parser()
    args.method = 'fmgda'
```

## 实验记录
1. 先注册一个swanlab账号，官网https://swanlab.cn
2. 加入团队zhaoFMOO
3. 运行前先在终端输入
   ```
   swanlab log
   ```
   然后输入在设置找到的api key再运行代码
