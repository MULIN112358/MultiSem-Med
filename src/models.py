import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution
from config import args


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device("cpu:0")):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class CrossAttention(nn.Module):

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        return F.softmax(query @ key.t(), dim=-1) @ value


class MultiSemMed(nn.Module):
    def __init__(
            self,
            vocab_size,
            ehr_adj,
            ddi_adj,
            emb_dim=64,
            nhead=8,
            device=torch.device("cuda:0"),
            ddi_in_memory=True,
            dataset='mimic-iii'
    ):
        super(MultiSemMed, self).__init__()
        K = len(vocab_size)
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K - 1)]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=8, batch_first=True, dropout=0.2)
             for _ in
             range(K - 1)]
        )
        self.query = nn.Sequential(
            nn.LayerNorm(emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.ehr_gcn = GCN(
            voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device
        )
        self.ddi_gcn = GCN(
            voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device
        )

        self.diag_med = dill.load(open(f'../data/{dataset}/output/diag_med.pkl', 'rb'))
        self.pro_med = dill.load(open(f'../data/{dataset}/output/pro_med.pkl', 'rb'))
        self.molecule = dill.load(open(f'../data/{dataset}/output/molecule.pkl', 'rb'))

        self.fusion = nn.Sequential(
            nn.LayerNorm(emb_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=emb_dim * 4, out_features=emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

        self.init_weights()

        self.attention1 = CrossAttention()
        self.attention2 = CrossAttention()

    def forward(self, input):
        # input (adm, 3, codes)
        i1_seq = []
        i2_seq = []
        i3_seq = []
        i4_seq = []

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )  # (1,1,dim)
            i2 = mean_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i3 = mean_embedding(
                self.diag_med['diagnosis_embedding'][adm[0]].unsqueeze(dim=0).to(self.device)
            )
            i3_seq.append(i3)
            i4 = mean_embedding(
                self.pro_med['procedure_embedding'][adm[1]].unsqueeze(dim=0).to(self.device)
            )
            i4_seq.append(i4)
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)
        i3_seq = torch.cat(i3_seq, dim=1)  # (1,seq,dim)
        i3_seq = F.normalize(i3_seq, p=2, dim=-1)
        i4_seq = torch.cat(i4_seq, dim=1)  # (1,seq,dim)
        i4_seq = F.normalize(i4_seq, p=2, dim=-1)

        o1 = self.encoders[0](i1_seq)  # o1:(1, seq, dim)
        o2 = self.encoders[1](i2_seq)
        patient_representations = torch.cat([i3_seq, o1, o2, i4_seq], dim=-1).squeeze(dim=0)  # (seq, dim*4)

        queries = self.query(patient_representations)  # (seq, dim)

        query = queries[-1:]  # (1,dim)
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()

        proj = nn.Linear(in_features=self.molecule.shape[0], out_features=drug_memory.shape[0], device=self.device)
        projection = nn.Linear(in_features=512, out_features=64, device=self.device)
        drug_memory = self.fusion(torch.cat((
            F.normalize(self.diag_med['medicine_embedding'].to(self.device), p=2, dim=-1),
            F.normalize(drug_memory, p=2, dim=-1),
            F.normalize(projection(proj(self.molecule.to(self.device).t()).t()), p=2, dim=-1),
            F.normalize(self.pro_med['medicine_embedding'].to(self.device), p=2, dim=-1),
        ), dim=-1))

        fact1 = self.attention1(query=query, key=drug_memory, value=drug_memory)

        if len(input) > 1:
            history_keys = queries[: (queries.size(0) - 1)]  # (seq-1, dim)
            history_values = np.zeros((len(input) - 1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input) - 1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(
                self.device
            )  # (seq-1, size)

            fact2 = self.attention2(query=query, key=history_keys, value=history_values @ drug_memory)

        else:
            fact2 = fact1

        output = self.output(torch.cat([query, fact1, fact2], dim=-1))  # (1, dim)
        result = output

        if self.training:
            neg_pred_prob = F.sigmoid(result)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()
            return result, batch_neg
        else:
            return result

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        self.inter.data.uniform_(-initrange, initrange)
