import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class MeanAdd(nn.Module):
    def __init__(self, init_means_add=(0.5, 0.001, 0.001)):
        super().__init__()
        self.coef = nn.Parameter(torch.tensor(init_means_add, requires_grad=True))

    def forward(self, x, similarity_matrix, scale):
        mask = x > 0

        means_rows = x.sum(axis=1) / (mask.sum(axis=1) + 1e-5)
        means_cols = x.sum(axis=0) / (mask.sum(axis=0) + 1e-5)

        means_rows_m = mask * means_rows[:, None]
        means_rows_sum = similarity_matrix @ means_rows_m / scale
        means_rows_add = means_rows[:, None] - means_rows_sum

        means_rows_centered = means_rows - means_rows.mean()
        means_cols_centered = means_cols - means_cols.mean()

        return (means_rows_add * self.coef[0] + means_cols_centered[None, :] * self.coef[
            1] + means_rows_centered[:, None] * self.coef[2])


class Embeddings(nn.Module):
    def __init__(
            self,
            num_embeddings,
            has_zero_emb=True,
            padding_idx=None,
            weights=None,
    ):
        super().__init__()
        embedding_dim = num_embeddings
        self.embedding_dim = embedding_dim

        if has_zero_emb:
            num_embeddings += 1
            ohe_weights = torch.cat([
                torch.zeros(1, embedding_dim),
                torch.eye(embedding_dim),
            ], dim=0)
        else:
            ohe_weights = torch.eye(embedding_dim)

        self.vals = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            _weight=weights,
        )
        self.ohe = nn.Parameter(
            ohe_weights,
            requires_grad=False,
        )

    def forward(self, x):
        shape = -1, self.embedding_dim * x.shape[1]
        embed = self.vals(x).reshape(shape)
        embed_bool = self.ohe[x].reshape(shape)
        return embed, embed_bool


class Attention(nn.Module):
    def __init__(
            self,
            init_beta      = 1.0,
            init_gamma     = 0.565,
            init_means_add = (0.5, 0.001, 0.001),
            device='cuda'
    ):
        super().__init__()
        self.beta      = nn.Parameter(torch.Tensor([init_beta]), requires_grad=True)
        self.gamma     = nn.Parameter(torch.Tensor([init_gamma]), requires_grad=True)
        self.means_add = MeanAdd(init_means_add)
        self.device    = device

    def forward(self, tr_onehot, tr_onehot_bool, x):
        mask = x > 0

        sims = tr_onehot @ tr_onehot_bool.T
        sims = sims.masked_fill(torch.eye(x.shape[0]).bool().to(self.device), 0)

        temperature = torch.pow(torch.max(sims, dim=1)[0] + 1e-3, self.gamma) * self.beta

        sims_scaled       = sims / temperature
        similarity_matrix = F.softmax(sims_scaled, dim=1)

        mult = similarity_matrix @ x.float()
        scale = similarity_matrix @ mask.float() + 1e-4
        res = mult / scale

        return res + self.means_add(x, similarity_matrix, scale)


class Submodel(nn.Module):
    def __init__(
            self,
            features_dict: Dict = None,
            init_vals: list or np.ndarray = (3., 1, -1, -3, -5),
            init_zero_emb: list or float = (0, 0, 0, 0, 0),
            init_beta: float = 1.0,
            init_gamma: float = 0.565,
            init_means_add: list or np.ndarray = (0.5, 0.001, 0.001),
            device: str = 'cuda',
            seed: int = 42
    ):

        super().__init__()
        torch.manual_seed(seed)

        self.features_dict = features_dict

        x1, x2, x3, x4, x5 = init_vals
        if isinstance(init_zero_emb, (float, int)):
            init_zero_emb = [init_zero_emb for _ in range(len(init_vals))]

        weights = torch.Tensor([
            list(init_zero_emb),
            [x1, x2, x3, x4, x5],
            [x2, x1, x2, x3, x4],
            [x3, x2, x1, x2, x3],
            [x4, x3, x2, x1, x2],
            [x5, x4, x3, x2, x1]
        ])

        self.embeddings = nn.ModuleDict()
        self.embeddings['x'] = Embeddings(5, weights=weights)
        for key, values in self.features_dict.items():
            self.embeddings[key] = Embeddings(*values)

        self.main_attention = Attention(init_beta, init_gamma, init_means_add, device)
        self.activation = nn.Hardtanh(1, 5)

    def forward(self, features):

        values = {
            'values': dict(),
            'values_bool': dict(),
            'attention_res': dict(),
        }
        for key in features:
            value, value_bool = self.embeddings[key](features[key])
            values['values'][key] = value
            values['values_bool'][key] = value_bool

        values['values']['main'] = torch.cat([
            values['values'][key]
            for key in features
        ], dim=-1)

        values['values_bool']['main'] = torch.cat([
            values['values_bool'][key]
            for key in features
        ], dim=-1)

        values['attention_res']['main'] = self.main_attention(
            values['values']['main'],
            values['values_bool']['main'],
            features['x'],
        )

        result = values['attention_res']['main']
        return self.activation(result)


def MSEloss(target, pred):
    mask = target > 0
    return F.mse_loss(target[mask].float(), pred[mask])


class ARE:
    def __init__(
            self,
            user_features_dict: Dict or None = None,
            item_features_dict: Dict or None = None,
            alpha: float = 0.75,  # user-based model weight = 1 - blanding_item_weight
            user_init_vals: tuple or np.ndarray = (3., 1, -1, -3, -5),
            item_init_vals: tuple or np.ndarray = (3., 1, -1, -3, -5),
            user_init_zero_emb: tuple or float = (0, 0, 0, 0, 0),
            item_init_zero_emb: tuple or float = (0, 0, 0, 0, 0),
            user_init_beta: float = 1.0,
            item_init_beta: float = 1.0,
            user_init_gamma: float = 0.565,
            item_init_gamma: float = 0.565,
            user_init_means_add: tuple or np.ndarray = (0.5, 0.001, 0.001),
            item_init_means_add: tuple or np.ndarray = (0.5, 0.001, 0.001),
            device: str = 'cuda',
            seed: int = 42
    ):

        if user_features_dict is None:
            user_features_dict = {}
        if item_features_dict is None:
            item_features_dict = {}

        torch.manual_seed(seed)

        self.alpha = alpha
        self.user_model = Submodel(
            user_features_dict,
            user_init_vals,
            user_init_zero_emb,
            user_init_beta,
            user_init_gamma,
            user_init_means_add,
            device,
            seed
        ).to(device)

        self.item_model = Submodel(
            item_features_dict,
            item_init_vals,
            item_init_zero_emb,
            item_init_beta,
            item_init_gamma,
            item_init_means_add,
            device,
            seed
        ).to(device)

    def fit(
            self,
            train_matrix: torch.Tensor,  # should be user-oriented
            val_matrix: torch.Tensor = None,  # should be user-oriented
            user_train_input: Dict or None = None,
            item_train_input: Dict or None = None,
            which: str = 'both',  # 'both', 'user' or 'item'
            user_optimizer: Tuple[str, Dict] = None,  # Name of optimizer and kwargs for params
            item_optimizer: Tuple[str, Dict] = None,  # Name of optimizer and kwargs for params
            user_scheduler: Tuple[str, Dict] = None,  # Name of scheduler and kwargs for params or False if disable
            item_scheduler: Tuple[str, Dict] = None,  # Name of scheduler and kwargs for params or False if disable
            user_num_epochs: int = 10,
            item_num_epochs: int = 10,
            verbose: bool = True
    ):

        val_matrix_is_not_none = (val_matrix is not None)
        if which != 'item':
            # Set optimizer
            if user_optimizer is None:
                user_optimizer = optim.AdamW(self.user_model.parameters(), lr=3e-2)
            else:
                user_optimizer = getattr(optim, user_optimizer[0])(self.user_model.parameters(), **user_optimizer[1])
            # Set scheduler
            if user_scheduler is None:
                user_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=user_optimizer,
                    milestones=[3],
                    gamma=0.3,
                )
            elif user_scheduler is False:
                user_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=user_optimizer, milestones=[1], gamma=1.)
            else:
                user_scheduler = getattr(optim.lr_scheduler, user_scheduler[0])(user_optimizer, **user_scheduler[1])
            # Set train input if it is None:
            if user_train_input is None:
                user_train_input = {'x': train_matrix}

        if which != 'user':
            # Set optimizer
            if item_optimizer is None:
                item_optimizer = optim.AdamW(self.item_model.parameters(), lr=3e-2)
            else:
                item_optimizer = getattr(optim, item_optimizer[0])(self.item_model.parameters(), **item_optimizer[1])
            # Set scheduler
            if item_scheduler is None:
                item_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=item_optimizer,
                    milestones=[2, 7],
                    gamma=0.2,
                )
            elif item_scheduler is False:
                item_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=item_optimizer, milestones=[1], gamma=1.)
            else:
                item_scheduler = getattr(optim.lr_scheduler, item_scheduler[0])(item_optimizer, **item_scheduler[1])
            # Set train input if it is None:
            if item_train_input is None:
                item_train_input = {'x': train_matrix.T}

        for i, (model, train_input, optimizer, scheduler, num_epochs) in enumerate([
            [self.user_model, user_train_input, user_optimizer, user_scheduler, user_num_epochs],
            [self.item_model, item_train_input, item_optimizer, item_scheduler, item_num_epochs]
        ]):
            if (i == 0 and which != 'item') or (i == 1 and which != 'user'):
                self._fit_submodel(model, train_input, train_matrix, val_matrix, optimizer,
                                   scheduler, num_epochs, verbose)
            train_matrix = train_matrix.T  # to change it for item-based and then vice versa
            if val_matrix_is_not_none:
                val_matrix = val_matrix.T

        user_result = self.user_model(user_train_input)
        item_result = self.item_model(item_train_input).T
        blanding_result = self.alpha * item_result + (1 - self.alpha) * user_result

        with torch.no_grad():
            user_train_loss = torch.sqrt(MSEloss(train_matrix, user_result))
            item_train_loss = torch.sqrt(MSEloss(train_matrix, item_result))
            blanding_train_loss = torch.sqrt(MSEloss(train_matrix, blanding_result))
            if val_matrix_is_not_none:
                user_val_loss = torch.sqrt(MSEloss(val_matrix, user_result))
                item_val_loss = torch.sqrt(MSEloss(val_matrix, item_result))
                blanding_val_loss = torch.sqrt(MSEloss(val_matrix, blanding_result))

        print(*(f'ARE model fitted.\nUser-based: train loss: {user_train_loss:.4f},',
                f'val loss: {user_val_loss:.4f};' if val_matrix_is_not_none else '',
                f'\nItem-based: train loss: {item_train_loss:.4f}',
                f'val loss: {item_val_loss:.4f};' if val_matrix_is_not_none else '',
                f'\nARE: train loss: {blanding_train_loss:.4f}',
                f'val loss: {blanding_val_loss:.4f};' if val_matrix_is_not_none else ''))

    def predict(self, user_input, item_input):

        with torch.no_grad():
            user_out = self.user_model(user_input)
            item_out = self.item_model(item_input).T
        are_out = item_out * self.alpha + user_out * (1 - self.alpha)
        return are_out

    def _fit_submodel(
            self,
            model,  # self.user_model or self.item_model
            train_input,  # dict of torch.Tensors
            train_matrix,  # torch.Tensor
            val_matrix,  # torch.Tensor or None
            optimizer,  # instance of torch.optim
            scheduler,  # instance of torch.optim.lr_scheduler
            num_epochs,  # int
            verbose,  # bool
    ) -> None:

        model.train()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            result = model(train_input)
            train_loss = MSEloss(train_matrix, result)
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose:

                with torch.no_grad():
                    train_loss = torch.sqrt(MSEloss(train_matrix, result))
                    if val_matrix is not None:
                        val_loss = torch.sqrt(MSEloss(val_matrix, result))

                string_to_verbose = f'Epoch {epoch}; train loss: {train_loss:.4f};'
                if val_matrix is not None:
                    string_to_verbose += f' val loss: {val_loss:.4f};'
                print(string_to_verbose)

        model.eval()


