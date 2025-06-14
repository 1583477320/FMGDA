import torch

class MultiObjectiveWeightOptimizer():
    def __init__(self):
        pass

    def optimize(self, G: torch.Tensor) -> torch.Tensor:
        """
        计算最小化 ||GG @ w||^2 且满足 sum(w)=1 的 w
        Args:
            GG: 形状 (n, m) 的矩阵
        Returns:
            w: 形状 (m,) 的最优向量
        """
        # 计算 A = G^T @ G
        G = sum(G)/len(G)
        A = G.T @ G

        # 构造全 1 向量
        ones = torch.ones(G.size(1), device=G.device, dtype=G.dtype)

        # 解线性方程组 A w = ones (使用伪逆保证数值稳定性)
        try:
            # 尝试直接求解（若 A 可逆）
            v = torch.linalg.solve(A, ones)
        except RuntimeError:  # 若 A 奇异，使用伪逆
            A_pinv = torch.linalg.pinv(A)
            v = A_pinv @ ones

        # 归一化以满足 sum(w)=1
        w = v / v.sum()
        return w


# 示例测试
if __name__ == "__main__":
    n, m = 3, 2
    GG = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weighted_optimizer = MultiObjectiveWeightOptimizer()
    w_optimal = weighted_optimizer.optimize(GG)

    print("GG =\n", GG)
    print("Optimal w =", w_optimal)
    print("Sum(w) =", w_optimal.sum().item())
    print("||GG @ w||^2 =", (GG @ w_optimal).pow(2).sum().item())