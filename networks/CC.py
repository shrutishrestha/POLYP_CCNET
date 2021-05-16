import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
print("CC Moudle")
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
class CC_module(nn.Module):
    def __init__(self,in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x) #([2, 8, 5, 6])
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1) #[12, 5, 8]
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1) #[10, 6, 8]

        proj_key = self.key_conv(x) #[2, 8, 5, 6]
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) #[12, 8, 5]
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) #[10, 8, 6]


        proj_value = self.value_conv(x) #[2, 64, 5, 6]
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) #[12, 64, 5]
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) #[10, 64, 6]

        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3) #[2, 5, 6, 5]
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width) #[2, 5, 6, 6]
        concate = self.softmax(torch.cat([energy_H, energy_W], 3)) #[2, 5, 6, 11]

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height) #[12, 5, 5]
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width) #[10, 6, 6]
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1) #[2, 64, 5, 6]
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3) #[2, 64, 5, 6]
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x



if __name__ == '__main__':
    model = CC_module(64)
    x = torch.randn(2, 64, 5, 6)
    out = model(x)
