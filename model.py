import torch

def model_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class NLB(torch.nn.Module):
    def __init__(self, in_ch, relu_a=0.01):
        super().__init__()
        self.theta_layer = torch.nn.Conv2d(in_channels=in_ch, out_channels=in_ch//2, \
                            kernel_size=1, padding=0)
        self.phi_layer   = torch.nn.Conv2d(in_channels=in_ch, out_channels=in_ch//2, \
                            kernel_size=1, padding=0)
        self.g_layer     = torch.nn.Conv2d(in_channels=in_ch, out_channels=in_ch//2, \
                            kernel_size=1, padding=0)
        self.atten_act   = torch.nn.Softmax(dim=-1)
        self.out_cnn     = torch.nn.Conv2d(in_channels=in_ch//2, out_channels=in_ch, \
                            kernel_size=1, padding=0)
        
    def forward(self, x, ret_att=False):
        mbsz, c, h, w = x.size()
        op_ch = c // 2
        
        theta = self.theta_layer(x)
        phi   = self.theta_layer(x)
        g     = self.theta_layer(x)
        
        theta_re = theta.view(mbsz, op_ch, -1)
        theta_re = torch.transpose(theta_re, 1, 2)
        
        phi_re   = phi.view(mbsz, op_ch, -1)
        
        theta_phi = torch.matmul(theta_re, phi_re)
        _attention = self.atten_act(theta_phi)
        
        g_re = g.view(mbsz, op_ch, -1)
        g_re = torch.transpose(g_re, 1, 2)
        
        _out_tmp = torch.matmul(_attention, g_re)
        _out_tmp = torch.transpose(_out_tmp, 1, 2).reshape(mbsz, op_ch, h, w)
        _out_tmp = self.out_cnn(_out_tmp)
        _out_tmp = torch.add(_out_tmp, x)
   
        if ret_att:
            return _attention, _out_tmp
        else:
            return _out_tmp

class BraggNN(torch.nn.Module):
    def __init__(self, imgsz, fcsz=(64, 32, 16, 8)):
        super().__init__()
        self.cnn_ops = []
        cnn_out_chs = (64, 32, 8)
        cnn_in_chs  = (1, ) + cnn_out_chs[:-1]
        fsz = imgsz
        for ic, oc, in zip(cnn_in_chs, cnn_out_chs):
            self.cnn_ops += [
                            torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, \
                                stride=1, padding=0),
                            torch.nn.LeakyReLU(negative_slope=0.01),
            ]
            fsz -= 2
        self.nlb = NLB(in_ch=cnn_out_chs[0])
        self.dense_ops = []
        dense_in_chs  = (fsz * fsz * cnn_out_chs[-1], ) + fcsz[:-1]
        for ic, oc in zip(dense_in_chs, fcsz):
            self.dense_ops += [
                            torch.nn.Linear(ic, oc),
                            torch.nn.LeakyReLU(negative_slope=0.01),
            ]
        # output layer
        self.dense_ops += [torch.nn.Linear(fcsz[-1], 2), ]
        
        self.cnn_layers   = torch.nn.Sequential(*self.cnn_ops)
        self.dense_layers = torch.nn.Sequential(*self.dense_ops)
        
    def forward(self, x):
        _out = x
        for layer in self.cnn_layers[:1]:
            _out = layer(_out)

        _out = self.nlb(_out)

        for layer in self.cnn_layers[1:]:
            _out = layer(_out)
            
        _out = _out.reshape(_out.size()[0], -1)
        for layer in self.dense_layers:
            _out = layer(_out)
            
        return _out