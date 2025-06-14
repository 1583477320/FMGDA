import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import swanlab
from data_load import generate_multi_mnist, split_data_to_servers
from model.ClientModel import ClientMTLModel
from model.ServiceModel import ServerSharedModel
from optim.client_optim import ClientAgg
from optim.service_optim import ServicAgg
from utils.options import args_parser, last_client_init


def train(args, server_model, client_model, client_datasets, method: str, last_shared_parameters=None,
          last_client_grads=None):
    print(f"======== batch_size {args.batch_size} ========")
    print(f"==={args.method} Federal Round {epoch}/{args.global_epochs} ===")

    # 超参设置
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)
    args.method = method

    client_models = []
    task1_loss_locals = []
    task2_loss_locals = []

    # 客户端本地训练
    client_models_gard = []
    server_model = server_model.to(args.device)
    client_model = client_model.to(args.device)
    for client_idx, dataset in client_datasets.items():
        # 加载本地数据
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        # 本地多任务训练
        # client_model = ClientMTLModel(server_model).to(args.device)
        if args.method == 'fmgda_s':
            client_local = ClientAgg(args.method, args, client_model, train_loader,
                                     last_shared_parameters=last_shared_parameters,
                                     last_client_grads=last_client_grads, )
        else:
            client_local = ClientAgg(args.method, args, client_model, train_loader, )
        client_model, client_gard, task_loss = client_local.backward()

        client_models.append(client_model)
        client_models_gard.append(client_gard)

        # 记录客户端各任务loss
        task1_loss_locals.append(task_loss[0])
        task2_loss_locals.append(task_loss[1])

    task1_loss_avg = sum(task1_loss_locals) / len(task1_loss_locals)
    task2_loss_avg = sum(task2_loss_locals) / len(task2_loss_locals)

    # loss_history['task1']["batch_size {}".format(batch_size)].append(task1_loss_avg.detach().numpy())
    # loss_history['task2']["batch_size {}".format(batch_size)].append(task2_loss_avg.detach().numpy())

    # 服务端共享层参数更新
    servicagg = ServicAgg(args, server_model, client_models_gard)
    if args.method == 'fmgda_s':
        last_shared_parameters = servicagg.get_last_model_parm()
    else:
        pass
    last_client_shared_parameters, server_model = servicagg.backward()  # last_client_shared_parameters返回的结构为向量结构

    # 更新客户端共享层模型
    client_model.shared_layer.load_state_dict(server_model.shared_parameters.state_dict())

    print(
        "task1 loss:{:.4f}".format(task1_loss_avg), "task2 loss:{:.4f}".format(task2_loss_avg))
    print("----------------------------------------------")
    swanlab.log({"train_task1_loss": task1_loss_avg.item(), "train_task2_loss": task2_loss_avg.item()})

    return client_models


def test(args, client_models, test_data):
    # 评估全局模型（以客户端0为例）
    criterion = nn.CrossEntropyLoss()
    client0_model = client_models[0].to(args.device)
    client0_model.eval()

    total_correct_task1 = 0
    total_correct_task2 = 0
    train_loader_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    with torch.no_grad():
        for data, (target1, target2) in train_loader_test:
            data, target_task1, target_task2 = data.to(args.device), target1.to(args.device), target2.to(
                args.device)

            pred_task1, pred_task2 = client0_model(data)

            # loss
            # total_loss_task1 += criterion(pred_task1, target_task1)
            # total_loss_task2 += criterion(pred_task2, target_task2)

            # correct
            pred1 = pred_task1.argmax(dim=1, keepdim=True)
            total_correct_task1 += pred1.eq(target_task1.view_as(pred1)).sum().item()

            pred2 = pred_task2.argmax(dim=1, keepdim=True)
            total_correct_task2 += pred2.eq(target_task2.view_as(pred2)).sum().item()
    accuracy_task1 = total_correct_task1 / len(train_loader_test.dataset) * 100
    accuracy_task2 = total_correct_task2 / len(train_loader_test.dataset) * 100
    print(
        'Client 0 Test - task1 correct:{:.2f}%'.format(accuracy_task1),
        'task2 correct:{:.2f}%'.format(accuracy_task2))
    swanlab.log({"test_task1_acc": accuracy_task1, "test_task2_acc": accuracy_task2})


# # 绘制损失曲线
# plt.figure(figsize=(10, 6))
# task1_loss = loss_history["task1"]
# for i in args.batch_size_list:
#     plt.plot(task1_loss["batch_size {}".format(i)], label="batch_size {}".format(i))
# plt.title("Task1 Loss")
# plt.xlabel("Global Epoch")
# plt.ylabel("Average Local Loss")
# plt.legend()
# plt.grid(False)
# # 保存图像
# plt.savefig(
#     'task1_mulit_loss_curve_method{}_num_servers{}_num_rounds{}_local_rate{}.png'.format(args.method,
#                                                                                          args.num_clients,
#                                                                                          args.global_epochs,
#                                                                                          args.local_lr),
#     dpi=300, bbox_inches='tight')
#
# plt.figure(figsize=(10, 6))
# task2_loss = loss_history["task2"]
# for i in args.batch_size_list:
#     plt.plot(task2_loss["batch_size {}".format(i)], label="batch_size {}".format(i))
# plt.title("Task2 Loss")
# plt.xlabel("Global Epoch")
# plt.ylabel("Average Local Loss")
# plt.legend()
# plt.grid(False)
#
# # 保存图像
# plt.savefig('task2_mulit_loss_curve_method{}_num_servers{}_num_rounds{}_local_rate{}.png'.format(args.method,
#                                                                                                  args.num_clients,
#                                                                                                  args.global_epochs,
#                                                                                                  args.local_lr),
#             dpi=300, bbox_inches='tight')
# plt.show()


if __name__ == "__main__":
    args = args_parser()
    args.method = 'fmgda'

    # 准备原始数据集
    # 不同分类生成一个批次
    train_dataset = generate_multi_mnist(num_samples=60000)

    # 生成测试数据
    test_dataset = generate_multi_mnist(num_samples=6000, train=False)

    sample_index = [i for i in range(6000)]  # 假设取随机6000个训练数据
    X_train = []
    y_train = []
    for i in sample_index:
        X = train_dataset[i][0]
        X_train.append(X)
        y = train_dataset[i][1]
        y_train.append(y)

    sampled_train_data = [(X, y) for X, y in zip(X_train, y_train)]  # 包装为数据对

    client_datasets = split_data_to_servers(sampled_train_data, num_servers=args.num_clients)  # 将训练集分给客户端

    sample_test_index = [i for i in range(256)]  # 假设取随机256个训练数据
    X_test = []
    y_test = []
    for i in sample_test_index:
        X = test_dataset[i][0]
        X_train.append(X)
        y = test_dataset[i][1]
        y_train.append(y)

    sampled_test_data = [(X, y) for X, y in zip(X_train, y_train)]  # 包装为数据对

    # 创建一个SwanLab项目
    swanlab.init(
        # 设置团队名
        workspace="zhaoFMOO",
        # 设置项目名
        project="FMGDA",
        # 设置实验名称
        experiment_name=f"{args.method}-{args.batch_size}",
        # 设置超参数
        config={
            "global_lr": args.global_lr,
            "local_lr": args.local_lr,
            "num_clients": args.num_clients,
            "num_tasks": args.num_tasks,
            "global_epochs": args.global_epochs,
            "local_epochs": args.local_epochs,
            "method": args.method,
        }
    )

    # 初始化模型参数
    server_model = ServerSharedModel()
    client_model = ClientMTLModel(server_model)

    # method == fmgda_s时对下列参数初始化
    # last_shared_parameters = client_model.shared_layer.state_dict()
    # last_client_grads = last_client_init(client_model)

    # 开始训练
    for epoch in range(1, args.global_epochs + 1):
        client_models = train(args, server_model, client_model, client_datasets=client_datasets, method=args.method)

        if epoch % 4 == 0:  # Test every 4 epochs
            test(args, client_models, test_data=sampled_test_data)
