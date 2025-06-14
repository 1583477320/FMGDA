import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Tuple, Union


# -----------------客户端训练逻辑---------------
class FMGDA():
    def __init__(self, args,client_model, dataset):
        self.aggregate_local_updates = None
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.client_model = client_model
        self.dataset = dataset
        self.args.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    def get_weighted_loss(
            self,
            losses,
            shared_parameters,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            if hasattr(param, 'data'):
                grad_dims.append(param.data.numel())

        grads = torch.Tensor(sum(grad_dims), self.args.num_tasks).to(self.args.device)

        '''
        如果共享层包含两个参数：3x4矩阵(12元素)和5x1向量(5元素),则grad_dims=[12, 5]，grads创建用于存储多任务梯度的张量容器,
        grad2vec将所有参数展平
        '''
        for i in range(self.args.num_tasks):
            for p in shared_parameters:
                p.grad = None
            if i < self.args.num_tasks - 1:
                '''
                隐式反向传播任务层梯度
                '''
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, int(i))
            # multi_task_model.zero_grad_shared_modules()

        self.overwrite_grad(shared_parameters, grads, grad_dims)
        return grads

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        '''
        梯度展平储存的函数
        '''
        # store the gradients
        # 初始化当前任务梯度列为0
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                # 复制当前参数的梯度
                grad_cur = grad.data.detach().clone()

                # 计算在展平向量中的位置
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])

                # 将梯度展平存储到指定位置
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        # 缩放梯度以匹配总损失
        newgrad = torch.mean(newgrad, dim=1)  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            # 计算当前参数在向量中的位置
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])

            # 提取对应位置的梯度并重塑为参数形状
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())

            # 覆盖参数的梯度
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(self,):
        # 客户端本地多任务训练
        client_model = self.client_model
        client_model.train()
        optimizer = optim.SGD(
            client_model.parameters(), lr=self.args.local_lr)

        # 记录损失
        task_loss = []
        G = 0
        client_epoch_loss = torch.empty(0, self.args.num_tasks)
        for _ in tqdm(range(self.args.local_epochs)):
            for data, targets in self.dataset:
                optimizer.zero_grad()
                data, targets[0], targets[1] = data.to(self.args.device), targets[0].to(self.args.device), targets[1].to(self.args.device)
                # total_grad = [torch.zeros_like(param) for param in client_model.feature_extractor.parameters()]
                # total_loss = 0

                # 计算损失
                outputs = client_model(data)
                losses = torch.stack([
                    self.criterion(outputs[i], targets[i])
                    for i in range(self.args.num_tasks)
                ])

                '''
                手动累积梯度（如果需要记录各任务梯度）,分别查看每个任务对共享层的梯度,retain_graph=True：在反向传播时，设置 retain_graph=True 以保留计算图，否则计算图会在第一次反向传播后被释放，无法进行第二次反向传播。
                梯度的累加：如果你不冻结共享层的梯度，梯度会自动累加。因此，在分别计算每个任务的梯度时，需要先清除共享层的梯度。
                优化器的 zero_grad() 方法：在每次反向传播之前，建议使用 optimizer.zero_grad() 或手动清除梯度，以避免梯度累积。
                '''

                # 获取梯度
                g = self.get_weighted_loss(
                    losses=losses,
                    shared_parameters=list((p for n, p in client_model.named_parameters() if "task" not in n)),
                )

                # 更新参数
                optimizer.step()

                # 记录损失
                client_batch_loss = torch.ones_like(torch.Tensor(1, self.args.num_tasks))
                for i in range(self.args.num_tasks):
                    client_batch_loss[:, i] = losses[i]
                client_epoch_loss = torch.cat((client_epoch_loss, client_batch_loss), dim=0)
            task_loss.append(torch.sum(client_epoch_loss, dim=0) / len(client_epoch_loss))

            # 梯度记录
            G += g

        # 计算每个任务的平均梯度
        avg_grad = G / self.args.local_epochs

        return client_model, avg_grad, sum(task_loss) / len(task_loss)

class FMGDA_S():
    def __init__(self, args, client_model, last_shared_parameters, last_client_grads, dataset,):
        self.aggregate_local_updates = None
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.client_model = client_model
        if self.args.method == 'fmgda_s':
            self.last_client_model = client_model
            self.last_client_model.shared_layer.load_state_dict(last_shared_parameters)
        else:
            pass
        self.last_client_grads = last_client_grads,
        self.last_shared_parameters = list((p for n, p in self.last_client_model.named_parameters() if "task" not in n))


    def get_weighted_loss(
            self,
            losses,
            shared_parameters,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        last_shared_parameters = self.last_shared_parameters[0]

        grad_dims = []
        for param in shared_parameters:
            if hasattr(param, 'data'):
                grad_dims.append(param.data.numel())

        grads = torch.Tensor(sum(grad_dims), self.args.num_tasks).to(self.args.device)
        last_grads = torch.Tensor(sum(grad_dims), self.args.num_tasks).to(self.args.device)

        '''
        如果共享层包含两个参数：3x4矩阵(12元素)和5x1向量(5元素),则grad_dims=[12, 5]，grads创建用于存储多任务梯度的张量容器,
        grad2vec将所有参数展平
        '''
        for i in range(self.args.num_tasks):
            for p in shared_parameters:
                p.grad = None
            for p in self.last_shared_parameters:
                p.grad = None
            if i < self.args.num_tasks - 1:
                '''
                隐式反向传播任务层梯度
                '''
                losses[0][i].backward(retain_graph=True)
                losses[1][i].backward(retain_graph=True)
            else:
                losses[0][i].backward(retain_graph=True)
                losses[1][i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, int(i % self.args.num_tasks))
            self.grad2vec(last_shared_parameters, last_grads, grad_dims, int(i % self.args.num_tasks))
            # multi_task_model.zero_grad_shared_modules()

        g = self.fmgda_s(grads, last_grads)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return g

    def fmgda_s(self, grads, last_grads):
        g = grads + (1 - self.args.beta) * (self.last_client_grads[0] - last_grads)
        return g

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        '''
        梯度展平储存的函数
        '''
        # store the gradients
        # 初始化当前任务梯度列为0
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                # 复制当前参数的梯度
                grad_cur = grad.data.detach().clone()

                # 计算在展平向量中的位置
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])

                # 将梯度展平存储到指定位置
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        # 缩放梯度以匹配总损失
        newgrad = torch.mean(newgrad, dim=1)  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            # 计算当前参数在向量中的位置
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])

            # 提取对应位置的梯度并重塑为参数形状
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())

            # 覆盖参数的梯度
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(self,):
        # 客户端本地多任务训练
        self.client_model.train()
        self.last_client_model.train()
        optimizer = optim.SGD(
            self.client_model.parameters(), lr=self.args.local_lr)

        # 记录损失
        task_loss = []
        G = 0
        GG = []
        client_epoch_loss = torch.empty(0, self.args.num_tasks)
        for _ in tqdm(range(self.args.local_epochs)):
            for data, targets in self.dataset:
                data, targets[0], targets[1] = data.to(self.args.device), targets[0].to(self.args.device), targets[1].to(
                    self.args.device)
                # total_grad = [torch.zeros_like(param) for param in client_model.feature_extractor.parameters()]
                # total_loss = 0

                # 计算当前模型的损失
                current_losses = [
                    self.criterion(self.client_model(data)[i], targets[i])
                    for i in range(self.args.num_tasks)
                ]

                # 计算上一轮模型的损失
                last_losses = [
                    self.criterion(self.last_client_model(data)[i], targets[i])
                    for i in range(self.args.num_tasks)
                ]

                # 堆叠为 [2, num_tasks] 的张量
                losses = torch.stack([
                    torch.stack(current_losses),
                    torch.stack(last_losses)
                ], dim=0)

                '''
                手动累积梯度（如果需要记录各任务梯度）,分别查看每个任务对共享层的梯度,retain_graph=True：在反向传播时，设置 retain_graph=True 以保留计算图，否则计算图会在第一次反向传播后被释放，无法进行第二次反向传播。
                梯度的累加：如果你不冻结共享层的梯度，梯度会自动累加。因此，在分别计算每个任务的梯度时，需要先清除共享层的梯度。
                优化器的 zero_grad() 方法：在每次反向传播之前，建议使用 optimizer.zero_grad() 或手动清除梯度，以避免梯度累积。
                '''

                # 获取梯度
                g = self.get_weighted_loss(
                    losses=losses,
                    shared_parameters=list((p for n, p in self.client_model.named_parameters() if "task" not in n)),
                )

                # 更新参数
                optimizer.step()
                optimizer.zero_grad()

                # 记录损失
                client_batch_loss = torch.ones_like(torch.Tensor(1, self.args.num_tasks))
                for i in range(self.args.num_tasks):
                    client_batch_loss[:, i] = losses[0][i]
                client_epoch_loss = torch.cat((client_epoch_loss, client_batch_loss), dim=0)
            task_loss.append(torch.sum(client_epoch_loss, dim=0) / len(client_epoch_loss))

            # 梯度记录
            G += g
            GG.append(g)

        # 计算每个任务的平均梯度
        avg_grad = G / self.args.local_epochs

        return self.client_model, avg_grad, sum(task_loss)/len(task_loss)


class ClientAgg:
    def __init__(self, method: str, args, client_model, dataset, last_shared_parameters=None, last_client_grads=None):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        if method == 'fmgda_s':
            self.method = METHODS[method](args, client_model, last_shared_parameters, last_client_grads, dataset,)
        else:
            self.method = METHODS[method](args, client_model, dataset, )

    def get_weighted_loss(self,
                          losses,
                          shared_parameters,):
        return self.method.get_weighted_loss(losses,
                                             shared_parameters,)

    def backward(
            self,
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward()

    def __ceil__(self, losses):
        return self.backward(losses)

    def parameters(self):
        return self.method.parameters()


METHODS = dict(
    # fmgda = FMGDA,
    fmgda_s=FMGDA_S,
    fmgda=FMGDA,
)
