import torch
import torch.nn as nn
import torch.nn.functional as F


class DecomposeLinearSVD(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank, weight, bias):
        super(DecomposeLinearSVD, self).__init__(
            in_features=in_features, out_features=out_features
        )
        self.weight = weight
        self.bias = bias
        self.weight1 = nn.Parameter(
            torch.zeros(rank, in_features, requires_grad=True, device="cuda")
        )
        self.weight2 = nn.Parameter(
            torch.zeros(out_features, rank, requires_grad=True, device="cuda")
        )
        self.U, self.S, self.Vh = torch.linalg.svd(
            self.weight, full_matrices=False
        )
        self.rank = rank
        self.weight1.data = torch.transpose(torch.transpose(self.Vh[: self.rank, :],1,0) @ torch.diag((self.S[: self.rank])),1,0).cuda()
        self.weight2.data = self.U[:, : self.rank].cuda()

    def forward(self, input):
        return F.linear(
            F.linear(input, self.weight1, None),
            self.weight2,
            self.bias,
        )

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = DecomposeLinearSVD(
            linear_module.in_features, linear_module.out_features, rank, linear_module.weight, linear_module.bias
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
            in_features=in_features, out_features=out_features
        )
        self.init = False
        self.weight = weight
        self.o_bias = bias
        self.bias = bias
        if self.bias is None:
            self.bias = self.weight2 = nn.Parameter(
            torch.zeros(out_features, requires_grad=True, device="cuda")
        )
        self.weight1 = nn.Parameter(
            torch.zeros(rank, in_features, requires_grad=True, device="cuda")
        )
        self.weight2 = nn.Parameter(
            torch.zeros(out_features, rank, requires_grad=True, device="cuda")
        )
        self.rank = rank

    def init_lowrank(self, input):
        Y = F.linear(input, self.weight, self.bias).reshape(-1,self.weight.shape[0]) # BS, out
        cov = torch.cov(torch.transpose(Y,1,0)) # out, out
        _, V = torch.linalg.eig(cov) # out, out
        V = V[:,:self.rank].float() # out, rank
        self.weight2.data = V.cuda()
        self.weight1.data = (torch.transpose(V,1,0) @ self.weight).cuda()
        if self.bias is not None:
            self.bias.data = (V @ torch.transpose(V,1,0) @ self.bias.data.unsqueeze(1)).squeeze(1).cuda()
        self.init = True

    def forward(self, input):
        if not self.init:
            self.init_lowrank(input)
        return F.linear(
            F.linear(input, self.weight1, None),
            self.weight2,
            self.bias,
        )

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = DecomposeLinearEigen(
            linear_module.in_features, linear_module.out_features, rank, linear_module.weight, linear_module.bias
        )
        # new_linear.weight = None
        new_linear.bias.requires_grad = True
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        return new_linear

class DecomposeLinearSVDPrune(DecomposeLinearSVD):
    def __init__(self, in_features, out_features, rank, budget, weight, bias):
        super(DecomposeLinearSVDPrune, self).__init__(
            in_features=in_features, out_features=out_features, rank=rank, weight=weight, bias=bias
        )
        self.zeta = nn.Parameter(
            torch.ones(1, rank, requires_grad=True, device='cuda')
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
        return F.linear(
            F.linear(input, self.weight1, None)*self.get_mask(),
            self.weight2,
            self.bias,
        )
    
    def set_threshold(self):
        kappa = self.in_features*self.out_features/(self.in_features+self.out_features)
        active_ranks = int(kappa*self.target_budget)
        sorted, _ = torch.sort(self.zeta, 1, descending=True)
        self.threshold = sorted[0][active_ranks]
    
    def set_budget(self):
        self.budget = ((self.zeta>self.threshold).sum()/self.rank).item()

    def get_mask(self):
        if self.pruned:
            self.set_threshold()
            self.set_budget()
            self.zeta.requires_grad = False
            return (self.zeta>self.threshold).to(self.zeta.device).to(self.zeta.dtype)
        else:
            return self.zeta
        
    @staticmethod
    def from_linear(linear_module, rank, budget):
        new_linear = DecomposeLinearSVDPrune(
            linear_module.in_features, linear_module.out_features, rank, budget, linear_module.weight, linear_module.bias
        )
        # new_linear.weight = None
        new_linear.bias.requires_grad = True
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        new_linear.zeta.requires_grad = True
        return new_linear

class DecomposeLinearEigenPrune(DecomposeLinearEigen):
    def __init__(self, in_features, out_features, rank, budget, weight, bias):
        super(DecomposeLinearEigenPrune, self).__init__(
            in_features=in_features, out_features=out_features, rank=rank, weight=weight, bias=bias
        )
        self.zeta = nn.Parameter(
            torch.ones(1, rank, requires_grad=True, device='cuda')
        )
        self.pruned = False
        self.threshold = 0
        self.target_budget = budget
        self.budget = 1.0

    def forward(self, input):
        if not self.init:
            self.init_lowrank(input)
        # z = self.get_zeta()
        # self.set_threshold()
        # self.set_budget()
        # self.mask = self.zeta - self.zeta.detach() + (self.zeta>self.threshold).to(self.zeta.device).to(self.zeta.dtype)
        return F.linear(
            F.linear(input, self.weight1, None)*self.get_mask(),
            self.weight2,
            self.bias,
        )
    
    def set_threshold(self):
        kappa = self.in_features*self.out_features/(self.in_features+self.out_features)
        active_ranks = int(kappa*self.target_budget)
        sorted, _ = torch.sort(self.zeta, 1, descending=True)
        self.threshold = sorted[0][active_ranks]
    
    def set_budget(self):
        self.budget = ((self.zeta>self.threshold).sum()/self.rank).item()

    def get_mask(self):
        if self.pruned:
            self.set_threshold()
            self.set_budget()
            self.zeta.requires_grad = False
            return (self.zeta>self.threshold).to(self.zeta.device).to(self.zeta.dtype)
        else:
            return self.zeta

    @staticmethod
    def from_linear(linear_module, rank, budget):
        new_linear = DecomposeLinearEigenPrune(
            linear_module.in_features, linear_module.out_features, rank, budget, linear_module.weight, linear_module.bias
        )
        # new_linear.weight = None
        new_linear.bias.requires_grad = True
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        new_linear.zeta.requires_grad = True
        return new_linear


class ModuleInjection:
    @staticmethod
    def make_decomposable(linear_module, budget, method='eigen'):
        """
        Make a (linear) layer decomposable.
        :param linear_module: A Linear module
        :return: a linear that is decomposed
        """
        in_channels = linear_module.in_features
        out_channels = linear_module.out_features
        kappa = in_channels*out_channels/(in_channels+out_channels)
        rank = int(kappa*budget)
        if method=='prune':
            new_linear = DecomposeLinearEigenPrune.from_linear(linear_module, linear_module.out_features, budget)
        if method=='prune-svd':
            new_linear = DecomposeLinearSVDPrune.from_linear(linear_module, linear_module.out_features, budget)
        elif method=='eigen':
            new_linear = DecomposeLinearEigen.from_linear(linear_module, rank)
        else:
            new_linear = DecomposeLinearSVD.from_linear(linear_module, rank)
        linear_module = None
        return new_linear

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, index=None, layers=None):
        super().__init__()
        self.model = model
        idx = 0
        for name, l in self.model.named_modules():
            if isinstance(l, nn.Linear):
                for eligible_layer in layers:
                    if eligible_layer in name:
                        if idx==index:
                            l.register_forward_hook(self.save_outputs_hook())
                        idx+=1

    def save_outputs_hook(self):
        def fn(module, input, output):
            self._features = output
        return fn

    def forward(self, x):
        _ = self.model(**x)
        return self._features
