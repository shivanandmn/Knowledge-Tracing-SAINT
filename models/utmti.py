from multihead_ffn import MultiHeadWithFFN
from utils import pos_encode, get_clones, ut_mask
import config 

import torch
from torch import nn

class UTMTI(nn.Module):
    def __init__(self,n_encoder,n_decoder,enc_heads,dec_heads,total_ex,n_dims,total_cat,total_responses,seq_len,max_time=300+1):
        super(UTMTI,self).__init__()
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.encoder = get_clones(EncoderBlock(enc_heads,n_dims,total_ex,total_cat,seq_len,max_time),n_encoder)
        self.decoder = get_clones(DecoderBlock(dec_heads,n_dims,total_responses,total_cat,seq_len),n_decoder)
        self.fc = nn.Linear(n_dims,1)
    
    def forward(self,in_exercise,in_category,in_response,in_etime):
        first_block = True
        exercies_ids = in_exercise
        categories = in_category
        for n in range(self.n_encoder):
          if n>=1:
            first_block=False
          enc = self.encoder[n](in_exercise,in_category,in_etime,in_response,first_block=first_block)
          in_exercise = enc
          in_category = enc
          in_etime = enc

        first_block = True
        for n in range(self.n_decoder):
          if n>=1:
            first_block=False
          dec = self.decoder[n](exercies_ids,encoder_output=in_exercise,category=categories,first_block=first_block)
          in_exercise = dec
          exercies_ids = dec
          
        return torch.sigmoid(self.fc(dec))



class EncoderBlock(nn.Module):
  def __init__(self,n_heads,n_dims,total_ex,total_cat,seq_len,time_width):
    super(EncoderBlock,self).__init__()
    self.seq_len = seq_len
    self.exercise_embed = nn.Embedding(total_ex,n_dims)
    self.category_embed = nn.Embedding(total_cat,n_dims)
    self.position_embed = nn.Embedding(seq_len,n_dims)
    self.response_embed = nn.Embedding(total_ex,n_dims)
    self.elapsetime_embed = nn.Embedding(time_width,n_dims)
    self.layer_norm = nn.LayerNorm(n_dims)
    
    self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                            n_dims = n_dims)
  
  def forward(self,input_e,category,elapse_time,response,first_block=True):
    if first_block:
      _exe = self.exercise_embed(input_e)
      _cat = self.category_embed(category)
      _etime = self.elapsetime_embed(elapse_time)
      _response = self.response_embed(response)
      position_encoded = pos_encode(self.seq_len-1)
      if config.device == "cuda":
        position_encoded = position_encoded.cuda()

      _pos = self.position_embed(position_encoded)

      interaction = _cat + _exe + _etime + _response + _pos 
    else:
      interaction = input_e
    output = self.multihead(q_input=interaction,kv_input=interaction)
    return output


class DecoderBlock(nn.Module):
    def __init__(self,n_heads,n_dims,total_exercise,total_cat,seq_len):
      super(DecoderBlock,self).__init__()
      self.seq_len = seq_len
      self.exercise_embed = nn.Embedding(total_exercise,n_dims)
      self.category_embed = nn.Embedding(total_cat,n_dims)
      self.position_embed = nn.Embedding(seq_len,n_dims)
      self.layer_norm = nn.LayerNorm(n_dims)
      self.multihead_attention = nn.MultiheadAttention(embed_dim=n_dims,
                                            num_heads = n_heads,
                                            dropout = 0.2)
      self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                              n_dims = n_dims)

    def forward(self,input_e,category,encoder_output,first_block=True):
      if first_block:
        _exe = self.exercise_embed(input_e)
        _cat = self.category_embed(category)
        position_encoded = pos_encode(self.seq_len-1)
        if config.device == "cuda":
          position_encoded = position_encoded.cuda()
        _pos = self.position_embed(position_encoded)
        exercise = _exe + _cat + _pos 
      else:
        exercise = input_e      
      exercise = exercise.permute(1,0,2)    
      #assert out_embed.size(0)==n_dims, "input dimention should be (seq_len,batch_size,dims)"
      out_norm = self.layer_norm(exercise)
      mask = ut_mask(out_norm.size(0))
      if config.device == "cuda":
        mask = mask.cuda()
      out_atten , weights_attent = self.multihead_attention(query=out_norm,
                                                  key = out_norm,
                                                  value = out_norm,
                                                  attn_mask = mask)
      out_atten +=  out_norm
      out_atten = out_atten.permute(1,0,2)
      output = self.multihead(q_input=out_atten,kv_input=encoder_output)
      return output

