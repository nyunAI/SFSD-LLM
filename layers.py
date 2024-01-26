import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import math
import numpy as np
import gc
import psutil
    


class DecomposeLinearSVD(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank, weight, bias):
        super(DecomposeLinearSVD, self).__init__(
            in_features=in_features, out_features=out_features
        )
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight, full_matrices=False)
        if not (isinstance(rank, float) or isinstance(rank, int)):
            variance = float(rank.split(":")[-1])
            S_sum = torch.cumsum(self.S.float(), 0)
            self.rank = torch.searchsorted(S_sum, S_sum[-1] * variance).item()
            self.target_budget = self.rank / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
        else:
            self.rank = rank
        self.weight = weight
        self.weight1 = nn.Parameter(
            torch.zeros(self.rank, in_features, requires_grad=True, device="cuda")
        )
        self.weight2 = nn.Parameter(
            torch.zeros(out_features, self.rank, requires_grad=True, device="cuda")
        )
        self.weight1.data = torch.transpose(
            torch.transpose(self.Vh[: self.rank, :], 1, 0)
            @ torch.diag((self.S[: self.rank])),
            1,
            0,
        ).cuda()
        w1_bias = torch.transpose(
            torch.transpose(self.Vh[ self.rank : , :], 1, 0)
            @ torch.diag((self.S[self.rank:])),
            1,
            0,
        ).mean(axis = 0).reshape((1,self.weight1.data.shape[1]))
        # self.weight1.data = torch.cat((self.weight1.data,w1_bias),axis = 0).cuda()
        self.weight2.data = self.U[:, : self.rank].cuda()
        w2_bias = self.U[:,self.rank:].mean(axis = 1).reshape((self.weight2.data.shape[0],1))
        # self.weight2.data = torch.cat((self.weight2.data,w2_bias),axis = 1).cuda()
        print(self.weight1.data.shape, self.weight2.data.shape)

    def forward(self, input):
        print(input.device, self.weight1.device, self.weight2.device)
        return F.linear(
            F.linear(input, self.weight1, None),
            self.weight2,
            None,
        )

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = DecomposeLinearSVD(
            linear_module.in_features,
            linear_module.out_features,
            rank,
            linear_module.weight,
            linear_module.bias,
        )
        # new_linear.weight = None
        # new_linear.bias = None
        new_linear.U = None
        new_linear.S = None
        new_linear.Vh = None
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        return new_linear


class DecomposeLinearEigen(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank, weight, bias):
        super(DecomposeLinearEigen, self).__init__(
            in_features=in_features, out_features=out_features, bias = True
        )
        self.mf16 = True
        self.init = False
        self.weight = weight
        self.rank = rank
        self.V = None
        self.weight1 = nn.Parameter(
            torch.zeros(
                rank,
                in_features,
                requires_grad=True,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.weight2 = nn.Parameter(
            torch.zeros(
                out_features,
                rank,
                requires_grad=True,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
    def make_float16(self):
        self.Y_sub = self.Y_sub.half()
        self.V = self.V.half()
        self.weight.data = self.weight.data.to(torch.float16)
        self.weight2.data = self.weight2.data.to(torch.float16)
        self.weight1.data =  self.weight1.data.to(torch.float16)
        self.bias.data = self.bias.data.to(torch.float16)
        self.b1 = self.b1.to(torch.float16)
        self.mf16 = True
        gc.collect()


    def init_lowrank(self, input):
        if self.V is None:
            Y = (
                F.linear(input, self.weight, None)
                .reshape(-1, self.out_features)
                .float()
                .cpu()
            )  # BS, out 
            Y_mean = torch.mean(Y, dim=0).unsqueeze(0)
            self.Y_sub = (Y - Y_mean)
            cov = torch.cov(torch.transpose(self.Y_sub, 1, 0))  # out, out
            _, self.V = torch.linalg.eigh(cov.float())  # out, out
        self.target_budget = self.rank / (
            self.in_features
            * self.out_features
            / (self.in_features + self.out_features)
        )
        # self.get_importance(Y)
        V = self.V[:, -self.rank:].to(self.weight.dtype)  # out, rank
        self.b1 = (Y_mean - Y_mean @ self.V @ self.V.transpose(1,0)).to(self.weight.device).to(self.weight.dtype)
        V_prune = self.V[:, :-self.rank].to(self.weight.device).to(self.weight.dtype) # out, rank
        # print(self.bias.data.dtype,V_prune.dtype, Y_sub.dtype) 
        self.Y_sub = self.Y_sub.mean(dim = 0, keepdim = True).to(self.weight.device)
        self.bias.data = self.b1 + (V_prune @ V_prune.transpose(1,0) @ self.Y_sub.transpose(1,0)).transpose(1,0).to(self.weight.device)
        # del V_prune
        self.bias.data = self.bias.data.to(self.weight.device).to(self.weight.dtype)
        self.weight2.data = V.to(self.weight.device)
        self.weight1.data = (
            torch.transpose(V, 1, 0).to(self.weight.device) @ self.weight
        )
        self.init = True
    # def init_lowrank(self, input):
    #     if self.V is None:
    #         Y = (
    #             F.linear(input, self.weight, None)
    #             .reshape(-1, self.out_features)
    #             .float()
    #             .cpu()
    #         )  # BS, out
    #         cov = torch.cov(torch.transpose(Y, 1, 0))  # out, out
    #         _, V1 = torch.linalg.eigh(cov.float())  # out, out
    #     self.target_budget = self.rank / (
    #         self.in_features
    #         * self.out_features
    #         / (self.in_features + self.out_features)
    #     )
    #     V = V1[:, -self.rank:].to(self.weight.dtype)  # out, rank
    #     self.weight2.data = V.to(self.weight.device)
    #     self.weight1.data = (
    #         torch.transpose(V, 1, 0).to(self.weight.device) @ self.weight
    #     ).to(self.weight.device)
    #     self.V = None
    #     del Y,cov,V,V1
    #     self.weight = None
    #     # self.get_importance(input)
    #     self.init = True

    def get_importance(self, input):
        input_norm = torch.norm(input.reshape(-1, input.shape[-1]), p=2, dim=0)[None, :]
        imp1 = input_norm * self.weight1.abs()
        imp1 = imp1.sum(1)
        input = F.linear(input, self.weight1, None)
        input_norm = torch.norm(input.reshape(-1, input.shape[-1]), p=2, dim=0)[None, :]
        imp2 = input_norm * self.weight2.abs()
        imp2 = imp2.sum(1)
        self.scores = imp1.tolist()
        # Y = XA @ self.weight2.transpose(1,0) # BS, out
        # self.scores = []
        # for i in range(self.rank):
        #     self.scores.append((XA[:,i][:,None] * self.weight2[i,:][None,:]).abs().mean().item()) # BS, out

    def forward(self, input):
        if not self.init:
            self.init_lowrank(input)
        # print(input.device, self.weight1.device,self.weight2.device)
        out = F.linear(
            F.linear(input, self.weight1, None),
            self.weight2,
            self.bias
        )
        if not self.mf16:
            self.make_float16()
        return out

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = DecomposeLinearEigen(
            linear_module.in_features,
            linear_module.out_features,
            rank,
            linear_module.weight,
            linear_module.bias,
        )
        # new_linear.weight = None
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        return new_linear


class DecomposeLinearSVDPrune(DecomposeLinearSVD):
    def __init__(self, in_features, out_features, rank, budget, weight, bias):
        super(DecomposeLinearSVDPrune, self).__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            weight=weight,
            bias=bias,
        )
        self.zeta = nn.Parameter(torch.ones(1, rank, requires_grad=True, device="cuda"))
        self.mask = nn.Parameter(
            torch.ones(1, rank, requires_grad=False, device="cuda")
        )
        self.pruned = False
        self.target_budget = budget
        if "auto" in budget:
            variance = float(self.target_budget.split(":")[-1])
            self.S = torch.cumsum(self.S.float(), 0)
            self.active_ranks = torch.searchsorted(self.S, self.S[-1] * variance).item()
            self.target_budget = self.active_ranks / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
        else:
            self.active_ranks = int(
                (
                    self.in_features
                    * self.out_features
                    / (self.in_features + self.out_features)
                )
                * float(budget)
            )
            self.target_budget = float(self.target_budget)

    def forward(self, input):
        if self.pruned:
            return F.linear(
                F.linear(input, self.weight1, None),
                self.weight2,
                None,
            )
        return F.linear(
            F.linear(input, self.weight1, None) * self.get_mask(),
            self.weight2,
            None,
        )

    def hard_prune(self, calculate=True):
        if calculate:
            sorted, _ = torch.sort(self.zeta.abs(), 1, descending=True)
            threshold = sorted[0][self.active_ranks]
            self.mask.data = (
                (self.zeta.abs() >= threshold).to(self.zeta.device).to(self.zeta.dtype)
            )
            self.mask.requires_grad = False
        self.target_budget = (
            self.mask.sum()
            / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
        ).item()
        self.pruned = True
        self.mask_indexes = torch.nonzero(self.mask)[:, 1]
        self.weight1 = torch.nn.Parameter(self.weight1.data[self.mask_indexes, :])
        self.weight2 = torch.nn.Parameter(self.weight2.data[:, self.mask_indexes])

    def get_mask(self):
        if self.pruned:
            return self.mask
        else:
            return self.zeta

    @staticmethod
    def from_linear(linear_module, rank, budget):
        new_linear = DecomposeLinearSVDPrune(
            linear_module.in_features,
            linear_module.out_features,
            rank,
            budget,
            linear_module.weight,
            linear_module.bias,
        )
        # new_linear.weight = None
        new_linear.U = None
        new_linear.S = None
        new_linear.Vh = None
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        new_linear.zeta.requires_grad = True
        return new_linear


class DecomposeLinearEigenPrune(DecomposeLinearEigen):
    def __init__(self, in_features, out_features, rank, budget, weight, bias):
        super(DecomposeLinearEigenPrune, self).__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            weight=weight,
            bias=bias,
        )
        self.zeta = nn.Parameter(torch.ones(1, rank, requires_grad=True, device="cuda"))
        self.mask = nn.Parameter(
            torch.ones(1, rank, requires_grad=False, device="cuda")
        )
        self.pruned = False
        self.target_budget = budget

    def init_lowrank(self, input):
        Y = F.linear(input, self.weight, None).reshape(
            -1, self.weight.shape[0]
        )  # BS, out
        cov = torch.cov(torch.transpose(Y, 1, 0))  # out, out
        E, V = torch.linalg.eig(cov)  # out, out
        if "auto" in self.target_budget:
            variance = float(self.target_budget.split(":")[-1])
            E = torch.cumsum(E.float(), 0)
            self.active_ranks = torch.searchsorted(E, E[-1] * variance).item()
            self.target_budget = self.active_ranks / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
        else:
            self.active_ranks = int(
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
                * self.target_budget
            )
        V = V[:, : self.rank].float()  # out, rank
        self.weight2.data = V.cuda()
        self.weight1.data = (torch.transpose(V, 1, 0) @ self.weight).cuda()
        self.init = True

    def forward(self, input):
        if not self.init:
            self.init_lowrank(input)
        if self.pruned:
            return F.linear(
                F.linear(input, self.weight1, None),
                self.weight2,
                None,
            )
        return F.linear(
            F.linear(input, self.weight1, None) * self.get_mask(),
            self.weight2,
            None,
        )

    def hard_prune(self, calculate=True):
        if calculate:
            sorted, _ = torch.sort(self.zeta.abs(), 1, descending=True)
            threshold = sorted[0][self.active_ranks]
            self.mask.data = (
                (self.zeta.abs() >= threshold).to(self.zeta.device).to(self.zeta.dtype)
            )
            self.mask.requires_grad = False
        self.target_budget = (
            self.mask.sum()
            / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
        ).item()
        self.pruned = True
        self.mask_indexes = torch.nonzero(self.mask)[:, 1]
        self.weight1 = torch.nn.Parameter(self.weight1.data[self.mask_indexes, :])
        self.weight2 = torch.nn.Parameter(self.weight2.data[:, self.mask_indexes])

    def get_mask(self):
        if self.pruned:
            return self.mask
        else:
            return self.zeta

    @staticmethod
    def from_linear(linear_module, rank, budget):
        new_linear = DecomposeLinearEigenPrune(
            linear_module.in_features,
            linear_module.out_features,
            rank,
            budget,
            linear_module.weight,
            linear_module.bias,
        )
        # new_linear.weight = None
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        new_linear.zeta.requires_grad = True
        return new_linear


class ChannelPrune(torch.nn.Linear):
    def __init__(self, in_features, out_features, budget, weight, bias):
        super(ChannelPrune, self).__init__(
            in_features=in_features, out_features=out_features
        )
        self.weight = weight
        self.zeta = nn.Parameter(
            torch.ones(1, out_features, requires_grad=True, device="cuda")
        )
        self.pruned = False
        self.threshold = 0
        self.target_budget = budget
        self.budget = 1.0

    def forward(self, input):
        # z = self.get_zeta()
        # self.set_threshold()
        # self.set_budget()
        # self.mask = self.zeta - self.zeta.detach() + (self.zeta>self.threshold).to(self.zeta.device).to(self.zeta.dtype)
        return F.linear(input, self.weight, None) * self.get_mask()

    def set_threshold(self):
        active_channels = int(
            math.sqrt(self.target_budget) * self.out_features
        )  ## assuming for input channels are pruned in previous layer at same rate
        sorted, _ = torch.sort(self.zeta.abs(), 1, descending=True)
        self.threshold = sorted[0][active_channels]

    def set_budget(self):
        self.budget = ((self.zeta >= self.threshold).sum() / self.out_features).item()

    def get_mask(self):
        if self.pruned:
            self.set_threshold()
            self.set_budget()
            self.zeta.requires_grad = False
            return (
                (self.zeta >= self.threshold).to(self.zeta.device).to(self.zeta.dtype)
            )
        else:
            return self.zeta

    @staticmethod
    def from_linear(linear_module, budget):
        new_linear = ChannelPrune(
            linear_module.in_features,
            linear_module.out_features,
            budget,
            linear_module.weight,
            linear_module.bias,
        )
        new_linear.weight.requires_grad = True
        new_linear.zeta.requires_grad = True
        return new_linear


class ModuleInjection:
    @staticmethod
    def make_decomposable(linear_module, budget, method="eigen"):
        """
        Make a (linear) layer decomposable.
        :param linear_module: A Linear module
        :return: a linear that is decomposed
        """
        in_channels = linear_module.in_features
        out_channels = linear_module.out_features
        kappa = in_channels * out_channels / (in_channels + out_channels)
        if method == "prune-eigen":
            new_linear = DecomposeLinearEigenPrune.from_linear(
                linear_module, linear_module.out_features, budget
            )
        elif method == "prune-svd":
            new_linear = DecomposeLinearSVDPrune.from_linear(
                linear_module, min(in_channels, out_channels), budget
            )
        elif method == "prune-channel":
            new_linear = ChannelPrune.from_linear(linear_module, budget)
        elif method == "eigen":
            if isinstance(budget, int):
                rank = budget
            else:
                rank = int(kappa * float(budget))
            new_linear = DecomposeLinearEigen.from_linear(linear_module, rank)
        elif method == "svd":
            if isinstance(budget, int):
                rank = budget
            else:
                rank = int(kappa * float(budget))
            new_linear = DecomposeLinearSVD.from_linear(linear_module, rank)
        else:
            for name, param in linear_module.named_parameters():
                param.requires_grad = True
            new_linear = linear_module
        linear_module = None
        return new_linear


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, index=None, layers=None, return_outputs=False):
        super().__init__()
        self.model = model
        self.return_outputs = return_outputs
        idx = 0
        for name, l in self.model.named_modules():
            if isinstance(l, nn.Linear):
                for eligible_layer in layers:
                    if eligible_layer in name:
                        if idx == index:
                            self.model.hook = l.register_forward_hook(
                                self.save_outputs_hook()
                            )
                        idx += 1

    def save_outputs_hook(self):
        def fn(module, input, output):
            self._features = output.float()
            if not self.return_outputs:
                assert False

        return fn

    def forward(self, x):
        try:
            x = {k: x[k].to(self.model.device) for k in x}
            outputs = self.model(**x)
            return self._features, outputs["loss"]
        except Exception as E:
            return self._features
