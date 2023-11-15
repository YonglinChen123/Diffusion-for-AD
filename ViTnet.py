import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from DDPM import CDF
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        #image_height = 95
        #image_width = 79
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            #print(2)
        )
        self.to_patch_embedding2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16),
            nn.LayerNorm(768),
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            # print(2)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, out2):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x2 = torch.cat((cls_tokens, x), dim=1)
        x2 += self.pos_embedding[:, :(n + 1)]
        x3 = self.dropout(x2)
        x4 = self.transformer(x3)
        x5 = x4.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #--------------------------------------------
        import numpy as np
        x5 = x5.cpu().detach().numpy()
        out = np.split(x5, 150)
        out2 = out2.cpu().detach().numpy()
        out21, out22, out23, out24, out25, out26, out27, out28, out29, out210, out211, out212, out213, out214, out215, out216, out217, out218, out219, out220, out221, out222, out223, out224, out225, out226, out227, out228, out229, out230, out231, out232, out233, out234, out235, out236, out237, out238, out239, out240, out241, out242, out243, out244, out245, out246, out247, out248, out249, out250, out251, out252, out253, out254, out255, out256, out257, out258, out259, out260, out261, out262, out263, out264, out265, out266, out267, out268, out269, out270, out271, out272, out273, out274, out275, out276, out277, out278, out279, out280, out281, out282, out283, out284, out285, out286, out287, out288, out289, out290, out291, out292, out293, out294, out295, out296, out297, out298, out299, out2100, out2101, out2102, out2103, out2104, out2105, out2106, out2107, out2108, out2109, out2110, out2111, out2112, out2113, out2114, out2115, out2116, out2117, out2118, out2119, out2120, out2121, out2122, out2123, out2124, out2125, out2126, out2127, out2128, out2129, out2130, out2131, out2132, out2133, out2134, out2135, out2136, out2137, out2138, out2139, out2140, out2141, out2142, out2143, out2144, out2145, out2146, out2147, out2148, out2149, out2150 = np.split(out2, 150)
        for i, Nout in enumerate (out):
            if i == 0:
                output = CDF(Nout,out21)
                output = output.T
                output0 = output
            if i == 1:
               output = CDF(Nout, out22)
               output = output.T
               output1 = torch.cat((output, output0), -2)
            if i == 2:
               output = CDF(Nout, out23)
               output = output.T
               output2 = torch.cat((output, output1), -2)
            if i == 3:
               output = CDF(Nout, out24)
               output = output.T
               output3 = torch.cat((output, output2), -2)
            if i == 4:
               output = CDF(Nout, out25)
               output = output.T
               output4 = torch.cat((output, output3), -2)
            if i == 5:
               output = CDF(Nout, out26)
               output = output.T
               output5 = torch.cat((output, output4), -2)
            if i == 6:
               output = CDF(Nout, out27)
               output = output.T
               output6 = torch.cat((output, output5), -2)
            if i == 7:
               output = CDF(Nout, out28)
               output = output.T
               output7 = torch.cat((output, output6), -2)
            if i == 8:
               output = CDF(Nout, out29)
               output = output.T
               output8 = torch.cat((output, output7), -2)
            if i == 9:
               output = CDF(Nout, out210)
               output = output.T
               output9 = torch.cat((output, output8), -2)
            if i == 10:
               output = CDF(Nout, out211)
               output = output.T
               output10 = torch.cat((output, output9), -2)
            if i == 11:
               output = CDF(Nout, out212)
               output = output.T
               output11 = torch.cat((output, output10), -2)
            if i == 12:
               output = CDF(Nout, out213)
               output = output.T
               output12 = torch.cat((output, output11), -2)
            if i == 13:
               output = CDF(Nout, out214)
               output = output.T
               output13 = torch.cat((output, output12), -2)
            if i == 14:
               output = CDF(Nout, out215)
               output = output.T
               output14 = torch.cat((output, output13), -2)
            if i == 15:
               output = CDF(Nout, out216)
               output = output.T
               output15 = torch.cat((output, output14), -2)
            if i == 16:
               output = CDF(Nout, out217)
               output = output.T
               output16 = torch.cat((output, output15), -2)
            if i == 17:
               output = CDF(Nout, out218)
               output = output.T
               output17 = torch.cat((output, output16), -2)
            if i == 18:
               output = CDF(Nout, out219)
               output = output.T
               output18 = torch.cat((output, output17), -2)
            if i == 19:
               output = CDF(Nout, out220)
               output = output.T
               output19 = torch.cat((output, output18), -2)
            if i == 20:
                output = CDF(Nout, out221)
                output = output.T
                output20 = torch.cat((output, output19), -2)
            if i == 21:
                output = CDF(Nout, out222)
                output = output.T
                output21 = torch.cat((output, output20), -2)
            if i == 22:
               output = CDF(Nout, out223)
               output = output.T
               output22 = torch.cat((output, output21), -2)
            if i == 23:
               output = CDF(Nout, out224)
               output = output.T
               output23 = torch.cat((output, output22), -2)
            if i == 24:
                output = CDF(Nout, out225)
                output = output.T
                output24 = torch.cat((output, output23), -2)
            if i == 25:
                output = CDF(Nout, out226)
                output = output.T
                output25 = torch.cat((output, output24), -2)
            if i == 26:
               output = CDF(Nout, out227)
               output = output.T
               output26 = torch.cat((output, output25), -2)
            if i == 27:
               output = CDF(Nout, out228)
               output = output.T
               output27 = torch.cat((output, output26), -2)
            if i == 28:
                output = CDF(Nout, out229)
                output = output.T
                output28 = torch.cat((output, output27), -2)
            if i == 29:
                output = CDF(Nout, out230)
                output = output.T
                output29 = torch.cat((output, output28), -2)
            if i == 30:
                output = CDF(Nout, out231)
                output = output.T
                output30 = torch.cat((output, output29), -2)
            if i == 31:
                output = CDF(Nout, out232)
                output = output.T
                output31 = torch.cat((output, output30), -2)
            if i == 32:
                output = CDF(Nout, out233)
                output = output.T
                output32 = torch.cat((output, output31), -2)
            if i == 33:
               output = CDF(Nout, out234)
               output = output.T
               output33 = torch.cat((output, output32), -2)
            if i == 34:
               output = CDF(Nout, out235)
               output = output.T
               output34 = torch.cat((output, output33), -2)
            if i == 35:
                output = CDF(Nout, out236)
                output = output.T
                output35 = torch.cat((output, output34), -2)
            if i == 36:
                output = CDF(Nout, out237)
                output = output.T
                output36 = torch.cat((output, output35), -2)
            if i == 37:
                output = CDF(Nout, out238)
                output = output.T
                output37 = torch.cat((output, output36), -2)
            if i == 38:
                output = CDF(Nout, out239)
                output = output.T
                output38 = torch.cat((output, output37), -2)
            if i == 39:
                output = CDF(Nout, out240)
                output = output.T
                output39 = torch.cat((output, output38), -2)
            if i == 40:
               output = CDF(Nout, out241)
               output = output.T
               output40 = torch.cat((output, output39), -2)
            if i == 41:
               output = CDF(Nout, out242)
               output = output.T
               output41 = torch.cat((output, output40), -2)
            if i == 42:
                output = CDF(Nout, out243)
                output = output.T
                output42 = torch.cat((output, output41), -2)
            if i == 43:
                output = CDF(Nout, out244)
                output = output.T
                output43 = torch.cat((output, output42), -2)
            if i == 44:
                output = CDF(Nout, out245)
                output = output.T
                output44 = torch.cat((output, output43), -2)
            if i == 45:
                output = CDF(Nout, out246)
                output = output.T
                output45 = torch.cat((output, output44), -2)
            if i == 46:
               output = CDF(Nout, out247)
               output = output.T
               output46 = torch.cat((output, output45), -2)
            if i == 47:
                output = CDF(Nout, out248)
                output = output.T
                output47 = torch.cat((output, output46), -2)
            if i == 48:
                output = CDF(Nout, out249)
                output = output.T
                output48 = torch.cat((output, output47), -2)
            if i == 49:
                output = CDF(Nout, out250)
                output = output.T
                output49 = torch.cat((output, output48), -2)
            if i == 50:
                output = CDF(Nout, out251)
                output = output.T
                output50 = torch.cat((output, output49), -2)
            if i == 51:
               output = CDF(Nout, out252)
               output = output.T
               output51 = torch.cat((output, output50), -2)
            if i == 52:
               output = CDF(Nout, out253)
               output = output.T
               output52 = torch.cat((output, output51), -2)
            if i == 53:
               output = CDF(Nout, out254)
               output = output.T
               output53 = torch.cat((output, output52), -2)
            if i == 54:
               output = CDF(Nout, out255)
               output = output.T
               output54 = torch.cat((output, output53), -2)
            if i == 55:
               output = CDF(Nout, out256)
               output = output.T
               output55 = torch.cat((output, output54), -2)
            if i == 56:
               output = CDF(Nout, out257)
               output = output.T
               output56 = torch.cat((output, output55), -2)
            if i == 57:
               output = CDF(Nout, out258)
               output = output.T
               output57 = torch.cat((output, output56), -2)
            if i == 58:
               output = CDF(Nout, out259)
               output = output.T
               output58 = torch.cat((output, output57), -2)
            if i == 59:
               output = CDF(Nout, out260)
               output = output.T
               output59 = torch.cat((output, output58), -2)
            if i == 60:
               output = CDF(Nout, out261)
               output = output.T
               output60 = torch.cat((output, output59), -2)
            if i == 61:
               output = CDF(Nout, out262)
               output = output.T
               output61 = torch.cat((output, output60), -2)
            if i == 62:
               output = CDF(Nout, out263)
               output = output.T
               output62 = torch.cat((output, output61), -2)
            if i == 63:
               output = CDF(Nout, out264)
               output = output.T
               output63 = torch.cat((output, output62), -2)
            if i == 64:
               output = CDF(Nout, out265)
               output = output.T
               output64 = torch.cat((output, output63), -2)
            if i == 65:
               output = CDF(Nout, out266)
               output = output.T
               output65 = torch.cat((output, output64), -2)
            if i == 66:
               output = CDF(Nout, out267)
               output = output.T
               output66 = torch.cat((output, output65), -2)
            if i == 67:
               output = CDF(Nout, out268)
               output = output.T
               output67 = torch.cat((output, output66), -2)
            if i == 68:
               output = CDF(Nout, out269)
               output = output.T
               output68 = torch.cat((output, output67), -2)
            if i == 69:
               output = CDF(Nout, out270)
               output = output.T
               output69 = torch.cat((output, output68), -2)
            if i == 70:
                output = CDF(Nout, out271)
                output = output.T
                output70 = torch.cat((output, output69), -2)
            if i == 71:
                output = CDF(Nout, out272)
                output = output.T
                output71 = torch.cat((output, output70), -2)
            if i == 72:
               output = CDF(Nout, out273)
               output = output.T
               output72 = torch.cat((output, output71), -2)
            if i == 73:
               output = CDF(Nout, out274)
               output = output.T
               output73 = torch.cat((output, output72), -2)
            if i == 74:
                output = CDF(Nout, out275)
                output = output.T
                output74 = torch.cat((output, output73), -2)
            if i == 75:
                output = CDF(Nout, out276)
                output = output.T
                output75 = torch.cat((output, output74), -2)
            if i == 76:
               output = CDF(Nout, out277)
               output = output.T
               output76 = torch.cat((output, output75), -2)
            if i == 77:
               output = CDF(Nout, out278)
               output = output.T
               output77 = torch.cat((output, output76), -2)
            if i == 78:
                output = CDF(Nout, out279)
                output = output.T
                output78 = torch.cat((output, output77), -2)
            if i == 79:
                output = CDF(Nout, out280)
                output = output.T
                output79 = torch.cat((output, output78), -2)
            if i == 80:
                output = CDF(Nout, out281)
                output = output.T
                output80 = torch.cat((output, output79), -2)
            if i == 81:
                output = CDF(Nout, out282)
                output = output.T
                output81 = torch.cat((output, output80), -2)
            if i == 82:
                output = CDF(Nout, out283)
                output = output.T
                output82 = torch.cat((output, output81), -2)
            if i == 83:
               output = CDF(Nout, out284)
               output = output.T
               output83 = torch.cat((output, output82), -2)
            if i == 84:
               output = CDF(Nout, out285)
               output = output.T
               output84 = torch.cat((output, output83), -2)
            if i == 85:
                output = CDF(Nout, out286)
                output = output.T
                output85 = torch.cat((output, output84), -2)
            if i == 86:
                output = CDF(Nout, out287)
                output = output.T
                output86 = torch.cat((output, output85), -2)
            if i == 87:
                output = CDF(Nout, out288)
                output = output.T
                output87 = torch.cat((output, output86), -2)
            if i == 88:
                output = CDF(Nout, out289)
                output = output.T
                output88 = torch.cat((output, output87), -2)
            if i == 89:
                output = CDF(Nout, out290)
                output = output.T
                output89 = torch.cat((output, output88), -2)
            if i == 90:
               output = CDF(Nout, out291)
               output = output.T
               output90 = torch.cat((output, output89), -2)
            if i == 91:
               output = CDF(Nout, out292)
               output = output.T
               output91 = torch.cat((output, output90), -2)
            if i == 92:
                output = CDF(Nout, out293)
                output = output.T
                output92 = torch.cat((output, output91), -2)
            if i == 93:
                output = CDF(Nout, out294)
                output = output.T
                output93 = torch.cat((output, output92), -2)
            if i == 94:
                output = CDF(Nout, out295)
                output = output.T
                output94 = torch.cat((output, output93), -2)
            if i == 95:
                output = CDF(Nout, out296)
                output = output.T
                output95 = torch.cat((output, output94), -2)
            if i == 96:
               output = CDF(Nout, out297)
               output = output.T
               output96 = torch.cat((output, output95), -2)
            if i == 97:
                output = CDF(Nout, out298)
                output = output.T
                output97 = torch.cat((output, output96), -2)
            if i == 98:
                output = CDF(Nout, out299)
                output = output.T
                output98 = torch.cat((output, output97), -2)
            if i == 99:
                output = CDF(Nout, out2100)
                output = output.T
                output99 = torch.cat((output, output98), -2)
            if i == 100:
                output = CDF(Nout, out2101)
                output = output.T
                output100 = torch.cat((output, output99), -2)
            if i == 101:
               output = CDF(Nout, out2102)
               output = output.T
               output101 = torch.cat((output, output100), -2)
            if i == 102:
               output = CDF(Nout, out2103)
               output = output.T
               output102 = torch.cat((output, output101), -2)
            if i == 103:
               output = CDF(Nout, out2104)
               output = output.T
               output103 = torch.cat((output, output102), -2)
            if i == 104:
               output = CDF(Nout, out2105)
               output = output.T
               output104 = torch.cat((output, output103), -2)
            if i == 105:
               output = CDF(Nout, out2106)
               output = output.T
               output105 = torch.cat((output, output104), -2)
            if i == 106:
               output = CDF(Nout, out2107)
               output = output.T
               output106 = torch.cat((output, output105), -2)
            if i == 107:
               output = CDF(Nout, out2108)
               output = output.T
               output107 = torch.cat((output, output106), -2)
            if i == 108:
               output = CDF(Nout, out2109)
               output = output.T
               output108 = torch.cat((output, output107), -2)
            if i == 109:
               output = CDF(Nout, out2110)
               output = output.T
               output109 = torch.cat((output, output108), -2)
            if i == 110:
               output = CDF(Nout, out2111)
               output = output.T
               output110 = torch.cat((output, output109), -2)
            if i == 111:
               output = CDF(Nout, out2112)
               output = output.T
               output111 = torch.cat((output, output110), -2)
            if i == 112:
               output = CDF(Nout, out2113)
               output = output.T
               output112 = torch.cat((output, output111), -2)
            if i == 113:
               output = CDF(Nout, out2114)
               output = output.T
               output113 = torch.cat((output, output112), -2)
            if i == 114:
               output = CDF(Nout, out2115)
               output = output.T
               output114 = torch.cat((output, output113), -2)
            if i == 115:
               output = CDF(Nout, out2116)
               output = output.T
               output115 = torch.cat((output, output114), -2)
            if i == 116:
               output = CDF(Nout, out2117)
               output = output.T
               output116 = torch.cat((output, output115), -2)
            if i == 117:
               output = CDF(Nout, out2118)
               output = output.T
               output117 = torch.cat((output, output116), -2)
            if i == 118:
               output = CDF(Nout, out2119)
               output = output.T
               output118 = torch.cat((output, output117), -2)
            if i == 119:
               output = CDF(Nout, out2120)
               output = output.T
               output119 = torch.cat((output, output118), -2)
            if i == 120:
                output = CDF(Nout, out2121)
                output = output.T
                output120 = torch.cat((output, output119), -2)
            if i == 121:
                output = CDF(Nout, out2122)
                output = output.T
                output121 = torch.cat((output, output120), -2)
            if i == 122:
               output = CDF(Nout, out2123)
               output = output.T
               output122 = torch.cat((output, output121), -2)
            if i == 123:
               output = CDF(Nout, out2124)
               output = output.T
               output123 = torch.cat((output, output122), -2)
            if i == 124:
                output = CDF(Nout, out2125)
                output = output.T
                output124 = torch.cat((output, output123), -2)
            if i == 125:
                output = CDF(Nout, out2126)
                output = output.T
                output125 = torch.cat((output, output124), -2)
            if i == 126:
               output = CDF(Nout, out2127)
               output = output.T
               output126 = torch.cat((output, output125), -2)
            if i == 127:
               output = CDF(Nout, out2128)
               output = output.T
               output127 = torch.cat((output, output126), -2)
            if i == 128:
                output = CDF(Nout, out2129)
                output = output.T
                output128 = torch.cat((output, output127), -2)
            if i == 129:
                output = CDF(Nout, out2130)
                output = output.T
                output129 = torch.cat((output, output128), -2)
            if i == 130:
                output = CDF(Nout, out2131)
                output = output.T
                output130 = torch.cat((output, output129), -2)
            if i == 131:
                output = CDF(Nout, out2132)
                output = output.T
                output131 = torch.cat((output, output130), -2)
            if i == 132:
                output = CDF(Nout, out2133)
                output = output.T
                output132 = torch.cat((output, output131), -2)
            if i == 133:
               output = CDF(Nout, out2134)
               output = output.T
               output133 = torch.cat((output, output132), -2)
            if i == 134:
               output = CDF(Nout, out2135)
               output = output.T
               output134 = torch.cat((output, output133), -2)
            if i == 135:
                output = CDF(Nout, out2136)
                output = output.T
                output135 = torch.cat((output, output134), -2)
            if i == 136:
                output = CDF(Nout, out2137)
                output = output.T
                output136 = torch.cat((output, output135), -2)
            if i == 137:
                output = CDF(Nout, out2138)
                output = output.T
                output137 = torch.cat((output, output136), -2)
            if i == 138:
                output = CDF(Nout, out2139)
                output = output.T
                output138 = torch.cat((output, output137), -2)
            if i == 139:
                output = CDF(Nout, out2140)
                output = output.T
                output139 = torch.cat((output, output138), -2)
            if i == 140:
               output = CDF(Nout, out2141)
               output = output.T
               output140 = torch.cat((output, output139), -2)
            if i == 141:
               output = CDF(Nout, out2142)
               output = output.T
               output141 = torch.cat((output, output140), -2)
            if i == 142:
                output = CDF(Nout, out2143)
                output = output.T
                output142 = torch.cat((output, output141), -2)
            if i == 143:
                output = CDF(Nout, out2144)
                output = output.T
                output143 = torch.cat((output, output142), -2)
            if i == 144:
                output = CDF(Nout, out2145)
                output = output.T
                output144 = torch.cat((output, output143), -2)
            if i == 145:
                output = CDF(Nout, out2146)
                output = output.T
                output145 = torch.cat((output, output144), -2)
            if i == 146:
               output = CDF(Nout, out2147)
               output = output.T
               output146 = torch.cat((output, output145), -2)
            if i == 147:
                output = CDF(Nout, out2148)
                output = output.T
                output147 = torch.cat((output, output146), -2)
            if i == 148:
                output = CDF(Nout, out2149)
                output = output.T
                output148 = torch.cat((output, output147), -2)
            if i == 149:
                output = CDF(Nout, out2150)
                output = output.T
                output149 = torch.cat((output, output148), -2)
        output = output149.cuda()
        #--------------------------------------------
        x51 = output.reshape(150, 32, 32, -1).permute(0, 3, 1, 2).contiguous()
        x52 = torch.cat((x51, x51), 1)
        x55 = torch.cat((x52, x51), 1)
        x55 = self.to_patch_embedding2(x55)
        b, n, _ = x55.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x55 = torch.cat((cls_tokens, x55), dim=1)
        x55 += self.pos_embedding[:, :(n + 1)]
        x55 = self.dropout(x55)
        x55 = self.transformer(x55)
        x55 = x55.mean(dim=1) if self.pool == 'mean' else x[:, 0]



        z = self.to_latent(x55)

        return z