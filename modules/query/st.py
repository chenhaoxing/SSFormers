import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import modules.registry as registry
from modules.utils import batched_index_select, _l2norm
from .innerproduct_similarity import InnerproductSimilarity


@registry.Query.register("ST")
class ST(nn.Module):

    def __init__(self, in_channels, cfg):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.inner_simi = InnerproductSimilarity(cfg, metric='cosine')
        self.nbnn_topk = cfg.model.nbnn_topk
        self.temperature = cfg.model.temperature
        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg

        self.project_dim = 160
        self.feat_dim = 640
        self.key_head = nn.Conv2d(self.feat_dim, self.project_dim, 1, bias=False)
        self.query_head = nn.Conv2d(self.feat_dim, self.project_dim, 1, bias=False)
        self.value_head = nn.Conv2d(self.feat_dim, self.feat_dim, 1, bias=False)

        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                n = l.kernel_size[0] * l.kernel_size[1] * l.out_channels
                torch.nn.init.normal_(l.weight, 0, math.sqrt(2. / n))
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)

    def _MNN(self, simi_matrix, compensate_for_single):
        b, q, N, M_q, M_s = simi_matrix.shape

        simi_matrix_merged = simi_matrix.permute(0, 1, 3, 2, 4).contiguous().view(b, q, M_q, -1) #b,q,mq,nms
        query_nearest = simi_matrix_merged.max(-1)[1]#b,q,mq
        if not compensate_for_single:
            support_nearest = simi_matrix_merged.max(-2)[1]  # For old Conv4 version  b,q,nms
        else:
            class_wise_max = (simi_matrix.max(-1)[0]).max(2)[0] + 1
            class_m = torch.nn.functional.one_hot(query_nearest, self.n_way * M_s).float() * class_wise_max.unsqueeze(
                -1)
            class_m_max, support_nearest = class_m.max(-2)
        # [b, q, M_q]
        mask = batched_index_select(support_nearest.view(-1, self.n_way * M_s).unsqueeze(1), 2,
                                    query_nearest.view(-1, M_q)) #bq,1,nms  #b*q,mq
        mask = (mask == torch.arange(M_q, device=simi_matrix.device).expand_as(mask)).view(b, q, M_q)
        return mask

    def forward(self, support_xf, support_y, query_xf, query_y):
        device = support_xf.device
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]
        support_xf = support_xf.view(b * s, c, h, w).contiguous()
        query_xf = query_xf.view(b * q, c, h, w).contiguous()

        query_xf_q = self.query_head(query_xf)
        query_xf_v = self.value_head(query_xf)

        support_xf_k = self.key_head(support_xf)
        support_xf_v = self.value_head(support_xf)

        query_xf_v = query_xf_v.view(b, q, self.feat_dim, h * w).unsqueeze(2)
        support_xf_v = support_xf_v.view(b, s, c, h, w)
        support_xf_v=  support_xf_v.view(b, self.n_way, self.k_shot, c, h, w).permute(0, 1, 3, 2, 4, 5)
        support_xf_v = support_xf_v.contiguous().view(b, self.n_way, c, -1).unsqueeze(1)

        query_xf_q = query_xf_q.view(b, q, self.project_dim, h * w).contiguous()
        query_xf_q = query_xf_q.unsqueeze(2).expand(-1, -1, self.n_way, -1, -1)
        query_xf_q = torch.transpose(query_xf_q, 3, 4)

        support_xf_k = support_xf_k.view(b, s, self.project_dim, h, w).contiguous()
        support_xf_k = support_xf_k.view(b, self.n_way, self.k_shot, self.project_dim, h, w).contiguous().permute(0, 1, 3, 2, 4, 5)
        support_xf_k = support_xf_k.contiguous().view(b, self.n_way, self.project_dim, -1)
        support_xf_k = support_xf_k.unsqueeze(1).expand(-1, q, -1, -1, -1)

        simi_matrix = query_xf_q @ support_xf_k
        simi_matrix = simi_matrix / np.power(self.project_dim, 0.5)
        q_mask = self._MNN(simi_matrix, self.cfg.model.encoder != "FourLayer_64F")  # bxQxNxM_qxM_s 1 40 1 25 1
        mask = simi_matrix * q_mask.float().unsqueeze(2).unsqueeze(-1)
        att = nn.Softmax(dim=-1)(mask)
        aligned_query_support = torch.matmul(att, support_xf_v.transpose(-1, -2)) #b,q,n,hw,c

        aligned_query_support = _l2norm(aligned_query_support, dim=-1)
        query_xf_v = _l2norm(query_xf_v, dim=-2)

        simi_matrix = aligned_query_support @ query_xf_v
        similarity = torch.topk(simi_matrix, 1, -2)[0]
        similarity = similarity.mean(-2).view(b, q, self.n_way, -1).sum(-1)
        similarity = similarity.view(b * q, self.n_way)

        query_y = query_y.view(b * q)
        if self.training:
            loss = self.criterion(similarity / self.temperature, query_y)
            return {"ST_loss": loss}
        else:
            _, predict_labels = torch.max(similarity, 1)
            rewards = [1 if predict_labels[j] == query_y[j].to(predict_labels.device) else 0 for j in
                       range(len(query_y))]
            return rewards
