import torch
import torch.nn.functional as F
import continual as co

class PasstWithSlide(co.CoModule, torch.nn.Module):
    def __init__(self, net, win_param=[512, 29], continual=False):
        torch.nn.Module.__init__(self)
        co.CoModule.__init__(self)
        self.emb_len = 1000
        self.net = net
        self.out_dim = self.net.out_dim
        self.win = win_param
        self.continual = continual

        self.embedding = None
        self.accumlator = None

        self.batch_size = None
        self.device = None

    def forward(self, input:torch.Tensor):
        """
        Returns:
            (sed_out, weak_out)
        """
        device = input.device
        batch_size, _, input_len = input.shape
        scale = self.emb_len / input_len

        self.batch_size = batch_size
        self.device = device

        # decode output shape :[batch, frame, shape]
        # For 10 seconds audio, frame length is 101
        win_width, step = self.win
        res = []
        embedding = torch.zeros([batch_size, self.emb_len, self.out_dim]).to(device)
        accumlator = torch.zeros([batch_size, self.emb_len, self.out_dim]).to(device)
        
        for w_left in range(0, input_len + step - win_width, step):
            w_right = min(w_left + win_width, input_len)
            out_left = round(w_left * scale)
            out = self.encode(input[:, :, w_left:w_right])
            out_right = int(min(self.emb_len, out_left + out.shape[1]))
            embedding[:, out_left:out_right, :] += out
            accumlator[:, out_left:out_right, :] += 1

        embedding /= accumlator
        embedding[torch.isnan(embedding)] = 0
        res.append(embedding)
            
        embedding = sum(res)/len(res)
        return embedding

    def clean_state(self, batch_size=None, device=None):
        self.net.patch_transformer_sliding_window.clean_state()

        if batch_size is None:
            batch_size = self.batch_size
        if device is None:
            device = self.device

        self.embedding = torch.zeros([batch_size, self.emb_len, self.out_dim]).to(device)
        self.accumlator = torch.zeros([batch_size, self.emb_len, self.out_dim]).to(device)

        self.state_index = 0

    def forward_steps(self, input: torch.Tensor):
        """
        Returns:
            (sed_out, weak_out)
        """
        device = input.device
        batch_size, _, input_len = input.shape

        self.batch_size = batch_size
        self.device = device

        # decode output shape :[batch, frame, shape]
        # For 10 seconds audio, frame length is 101
        win_width, step = self.win
        res = []

        if self.embedding is None:
            self.clean_state(batch_size, device)

        embedding_start_out = self.state_index

        w_right = 0
        for w_left in range(0, input_len + step - win_width, step):
            w_right = min(w_left + win_width, input_len)
            # out_left = w_left
            out = self.encode(input[:, :, w_left:w_right])
            self.net.patch_transformer_sliding_window.clean_state() # We need to reset the state after we complete a chunk

            # out_right = int(min(self.emb_len, out_left + out.shape[1]))
            self.embedding[:, self.state_index:min(self.state_index+out.size(1), self.emb_len), :] += out
            self.accumlator[:, self.state_index:min(self.state_index+out.size(1), self.emb_len), :] += 1
            self.state_index = (self.state_index + step) % self.emb_len

        if w_right < input_len:
            input_win = input[:, :, w_right:]
            out = self.encode(input_win)
            if out is None:
                return None
            self.net.patch_transformer_sliding_window.clean_state()  # We need to reset the state after we complete a chunk
            self.embedding[:, self.state_index:self.state_index+out.size(1), :] += out
            self.accumlator[:, self.state_index:self.state_index+out.size(1), :] += 1
            self.state_index = (self.state_index + step) % self.emb_len

            embedding = self.embedding[:, embedding_start_out:embedding_start_out + out.size(1)] / self.accumlator[:, embedding_start_out:embedding_start_out + out.size(1)]
        else:
            embedding = self.embedding[:, embedding_start_out:min(self.state_index + win_width, self.emb_len)] / self.accumlator[:, embedding_start_out:min(self.state_index + win_width, self.emb_len)]
        embedding[torch.isnan(embedding)] = 0
        res.append(embedding)

        embedding = sum(res) / len(res)
        return embedding

    def encode(self, input: torch.Tensor):
        #patch-wise context modeling
        input = input.unsqueeze(1)
        if self.continual and self.net.patch_transformer_sliding_window is not None:
            passt_out_dict = self.net.patch_transformer_sliding_window(input)
        else:
            passt_out_dict = self.net.patch_transformer(input)
        if passt_out_dict is None:
            return None
        passt_feature = passt_out_dict["layer{}_out".format(self.net.passt_feature_layer)]
        passt_feature = passt_feature.transpose(1, 2)  # N,C,P->N,P,C
        passt_feature = self.net.out_norm(passt_feature)
        
        B_, P_, C_ = passt_feature.shape
        # print(encoder_out['f_dim'])j
        # print(encoder_out['t_dim'])
        if not self.continual:
            passt_feature = passt_feature[:, 2:, :]
        passt_feature = passt_feature.reshape(B_, passt_out_dict['f_dim'], passt_out_dict['t_dim'], C_)
        frameout = torch.mean(passt_feature, dim=1)

        assert self.net.decode_ratio != 1
        frameout = frameout.transpose(1, 2)  # B,T,C->B,C,T
        frameout = F.interpolate(frameout, scale_factor=self.net.decode_ratio, mode=self.net.interpolate_module.mode)
        frameout = frameout.transpose(1, 2)  # B,C,T->B,T,C
        return frameout
