from utils import ut_mask, lt_mask
import config 

from torch import nn

#   MultiHead(Qin,Kin,Vin) = Concat(head1,··· ,headh)WO
class FFN(nn.Module):
  def __init__(self,features):
    super(FFN,self).__init__()
    self.layer1 = nn.Linear(features, features)
    self.layer2 = nn.Linear(features, features)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(0.2)
    
  def forward(self, x):
    out = self.drop(self.relu(self.layer1(x)))
    out = self.layer2(out)
    return out

class MultiHeadWithFFN(nn.Module):
  def __init__(self,n_heads,n_dims,mask_type="ut",dropout=0.2):
    super(MultiHeadWithFFN,self).__init__()
    self.n_dims = n_dims
    self.mask_type = mask_type
    self.multihead_attention = nn.MultiheadAttention(embed_dim = n_dims,
                                                      num_heads = n_heads,
                                                        dropout = dropout)
    self.layer_norm1 = nn.LayerNorm(n_dims)
    self.ffn = FFN(features = n_dims)
    self.layer_norm2 = nn.LayerNorm(n_dims)


  def forward(self,q_input,kv_input):
    q_input = q_input.permute(1,0,2)
    kv_input = kv_input.permute(1,0,2)
    query_norm = self.layer_norm1(q_input)
    kv_norm = self.layer_norm1(kv_input)
    if self.mask_type=="ut":
      mask = ut_mask(q_input.size(0))
    else: 
      mask = lt_mask(q_input.size(0))
    if config.device == "cuda":
      mask = mask.cuda()
    out_atten , weights_attent = self.multihead_attention(query=query_norm,
                                                key = kv_norm,
                                                value = kv_norm,
                                                attn_mask = mask)
    out_atten +=  query_norm
    out_atten = out_atten.permute(1,0,2)
    output_norm = self.layer_norm2(out_atten)
    output = self.ffn(output_norm)
    return output + output_norm 