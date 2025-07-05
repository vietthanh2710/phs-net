import torch
import torch.nn as nn

from utils.transformer import Block, PatchMerging2D, InputLayer
from utils.convmixer import ConvMixerBlock
from utils.skip_connection import SelectiveFusion, CBAM

class EncoderBlock_C(nn.Module):
    """Convolutional Encoder Block"""
    def __init__(self, in_c):
        super().__init__()
        self.convmixer1 = ConvMixerBlock(in_c, in_c*2, kernel_size = 3)
        self.convmixer2 = ConvMixerBlock(in_c*2, in_c*2, kernel_size = 3)
        self.pw = nn.Conv2d(in_c*2, in_c, kernel_size=1)
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.convmixer1(x)
        x = self.convmixer2(x)
        skip = self.act(self.pw(x))
        x = self.down(x)
        return x, skip

class EncoderBlock_T(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, in_c, size = (192, 256), num_head=8):
        super().__init__()
        self.H, self.W = size
        self.block1 = Block(embed_dim = in_c, nb_head = num_head, hidden_dim = in_c*2, dropout = 0.)
        self.block2 = Block(embed_dim = in_c, nb_head = num_head, hidden_dim = in_c*2, dropout = 0.)
        self.merge = PatchMerging2D(in_c)
    def forward(self, x):

        skip = self.block1(x)
        skip = self.block2(x)
        skip = skip.permute(0,2,3,1)

        x = self.merge(skip)

        x = x.permute(0,3,1,2)
        skip = skip.permute(0,3,1,2)

        return x, skip

class Attention_Gate(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_Gate,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, skip_c = None, num_head=8):
        super().__init__()
        if skip_c is not None:
            skip_c = out_c

        self.up = nn.Upsample(scale_factor = 2)
        self.att = Attention_block(F_g = in_c, F_l = skip_c, F_int= skip_c//2)

        self.conv1 = nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding = 'same')
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding = 'same')
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)

        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))

        return x


class PASPP(nn.Module):
    def __init__(self, inplanes, outplanes, output_stride=4, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        if output_stride == 4:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 4, 6, 10]
        elif output_stride == 2:
            dilations = [1, 12, 24, 36]
        elif output_stride == 16:
            dilations = [1, 2, 3, 4]
        elif output_stride == 1:
            dilations = [1, 16, 32, 48]
        else:
            raise NotImplementedError
        self._norm_layer = BatchNorm
        self.silu = nn.SiLU(inplace=True)
        self.conv1 = self._make_layer(inplanes, inplanes // 4)
        self.conv2 = self._make_layer(inplanes, inplanes // 4)
        self.conv3 = self._make_layer(inplanes, inplanes // 4)
        self.conv4 = self._make_layer(inplanes, inplanes // 4)
        self.atrous_conv1 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[0], padding=dilations[0])
        self.atrous_conv2 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[1], padding=dilations[1])
        self.atrous_conv3 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[2], padding=dilations[2])
        self.atrous_conv4 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[3], padding=dilations[3])
        self.conv5 = self._make_layer(inplanes // 2, inplanes // 2)
        self.conv6 = self._make_layer(inplanes // 2, inplanes // 2)
        self.convout = self._make_layer(inplanes, outplanes)

    def _make_layer(self, inplanes, outplanes):
        layer = []
        layer.append(nn.Conv2d(inplanes, outplanes, kernel_size = 1))
        layer.append(self._norm_layer(outplanes))
        layer.append(self.silu)
        return nn.Sequential(*layer)

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.conv2(X)
        x3 = self.conv3(X)
        x4 = self.conv4(X)

        x12 = torch.add(x1, x2)
        x34 = torch.add(x3, x4)

        x1 = torch.add(self.atrous_conv1(x1),x12)
        x2 = torch.add(self.atrous_conv2(x2),x12)
        x3 = torch.add(self.atrous_conv3(x3),x34)
        x4 = torch.add(self.atrous_conv4(x4),x34)

        x12 = torch.cat([x1, x2], dim = 1)
        x34 = torch.cat([x3, x4], dim = 1)

        x12 = self.conv5(x12)
        x34 = self.conv5(x34)
        x = torch.cat([x12, x34], dim=1)
        x = self.convout(x)
        return x

class PHSNet(nn.Module):
    def __init__(self):
        super().__init__(in_channels = 3, num_classes = 1)

        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding = 'same')
        self.trans_in = InputLayer(in_channels = 3, embed_dim = 64, patch_size = 4, image_size = (192,256))
        """Encoder"""
        self.c1 = EncoderBlock_C(16)
        self.c2 = EncoderBlock_C(32)
        self.c3 = EncoderBlock_C(64)
        self.c4 = EncoderBlock_C(128)
        self.c5 = EncoderBlock_C(256)

        self.v3 = EncoderBlock_T(64, (48,64), num_head=4)
        self.v4 = EncoderBlock_T(128, (24,32), num_head=8)
        self.v5 = EncoderBlock_T(256, (12,16), num_head=8)

        '''Selective block'''
        self.sb3 = SelectiveFusion(64, G=16, d =128)
        self.sb4 = SelectiveFusion(128, G=16, d = 256)
        self.sb5 = SelectiveFusion(256, G=16, d = 512)

        """Skip connection"""
        self.s1 = CBAM(gate_channels = 16)
        self.s2 = CBAM(gate_channels = 32)
        self.s3 = CBAM(gate_channels = 64)
        self.s4 = CBAM(gate_channels = 128)
        self.s5 = CBAM(gate_channels = 256)

        """Bottle Neck"""
        self.b5 = PASPP(1024, 512, output_stride=4, BatchNorm=nn.BatchNorm2d)

        """Decoder"""
        self.d5 = DecoderBlock(512, 256, num_head=8)
        self.d4 = DecoderBlock(256, 128, num_head=8)
        self.d3 = DecoderBlock(128, 64, num_head=4)
        self.d2 = DecoderBlock(64, 32, num_head=4)
        self.d1 = DecoderBlock(32, 16, num_head=2)
        self.out = nn.Conv2d(16, num_classes,kernel_size = 1)
    def forward(self, x):
        """Encoder"""
        xc = self.conv_in(x)
        xv = self.trans_in(x)

        xc, skip1 = self.c1(xc)
        xc, skip2 = self.c2(xc)

        xc, skipc = self.c3(xc)
        xv, skipv = self.v3(xv)

        skip3 = self.sb3(skipc, skipv)

        xc, skipc = self.c4(xc)
        xv, skipv = self.v4(xv)

        skip4 = self.sb4(skipc, skipv)

        xc, skipc = self.c5(xc)
        xv, skipv = self.v5(xv)

        skip5 = self.sb5(skipc, skipv)
        x = torch.cat((xc,xv), dim = 1)

        """Skip connection"""
        skip1 = self.s1(skip1)
        skip2 = self.s2(skip2)
        skip3 = self.s3(skip3)
        skip4 = self.s4(skip4)
        skip5 = self.s5(skip5)

        """BottleNeck"""
        x = self.b5(x)

        """Decoder"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)

        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.out(x)
        return x

if __name__ == "__main__"
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PHSNet()
    model.to(device)
    model.eval()
    
    batch_size = 2
    channels = 3
    height = 192
    width = 256
    
    input = torch.randn(batch_size, channels, height, width).to(device)

    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")        
    except Exception as e:
        print("Error in forwarding input")
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

