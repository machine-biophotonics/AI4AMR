"""DS-MIL: Dual-Stage Multiple Instance Learning"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True):
        super(BClassifier, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(nn.Dropout(dropout_v), nn.Linear(input_size, input_size))
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c):
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats)
        Q = self.q(feats).view(feats.shape[0], -1)
        _, m_indices = torch.sort(c, 0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats)
        A = torch.mm(Q, q_max.transpose(0, 1))
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)
        B = torch.mm(A.transpose(0, 1), V)
        B = B.view(1, B.shape[0], B.shape[1])
        C = self.fcc(B)
        C = C.view(1, -1)
        return C, A, B

class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        return classes, prediction_bag, A, B
    
    def get_instance_predictions(self, x):
        with torch.no_grad():
            feats, classes = self.i_classifier(x)
            classes = torch.sigmoid(classes)
        return classes, feats
