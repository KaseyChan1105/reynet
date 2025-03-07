# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.transform import rotate


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class LearnableSpatialTransformWrapper(nn.Module):
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        # 输入图像的高度和宽度分别乘以 pad_coef，然后使用 F.pad 函数对图像进行反射填充。这是为了确保在旋转变换过程中不会造成图像内容的丢失
        # train_angle是否允许模型在训练过程中调整旋转角度
        super().__init__()
        self.impl = impl  # self.impl 属性存储了传递给构造函数的 impl_model 模块
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
            # nn.Parameter可训练的参数
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            # 对单个输入张量进行变换和逆变换
            transformed_x = self.transform(x)  # 进行变换
            transformed_and_inverse_x = self.inverse_transform(self.impl(transformed_x), x)  # 进行逆变换
            return transformed_and_inverse_x
        elif isinstance(x, tuple):
            # 对多个输入元组中的张量分别进行变换和逆变换
            x_trans = tuple(self.transform(elem) for elem in x)  # 对输入元组中的每个张量进行变换
            y_trans = self.impl(x_trans)  # 对变换后的输入元组进行操作
            y_and_inverse_x = tuple(
                self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))  # 对操作结果进行逆变换
            return y_and_inverse_x
        else:
            # 若输入类型不是张量或元组，则引发错误
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        # 使用 F.pad 函数对输入 x 进行反射填充，以确保在旋转变换时不会丢失图像内容
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        # 使用 rotate 函数对填充后的图像进行旋转操作，旋转角度由 self.angle 控制
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)

        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        ## 根据填充量裁剪逆旋转后的图像 y_padded，以还原原始图像尺寸
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h: y_height - pad_h, pad_w: y_width - pad_w]
        return y


class FFCSE_block(nn.Module):
    # 实现了涉及频谱和空间处理的块，以增强局部和全局特征
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        # in_cg 和 in_cl：计算局部特征和全局特征的通道数
        r = 16

        # 使用全局平均池化操作对输入进行降维
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 第一个卷积层，用于产生增强的特征表示
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        # 如果 in_cl 不为零，则使用卷积操作增强局部特征
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)

        # 如果 in_cg 不为零，则使用卷积操作增强全局特征
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)

        # Sigmoid 激活函数，用于门控操作
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 如果 x 不是元组，则将其转换为元组 (x, 0)
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        # 合并局部特征和全局特征，通过在通道维度上拼接
        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)

        # 对合并后的特征进行全局平均池化、卷积和 ReLU 操作，产生增强的特征表示
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        # 使用门控操作，根据卷积结果增强局部特征和全局特征
        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))

        return x_l, x_g


class FourierUnit(nn.Module):  # 4
    # 实现了基于傅里叶变换的处理单元。它对输入数据执行快速傅里叶变换（FFT）和逆快速傅里叶变换（IFFT），从而实现频谱处理
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups

        # 卷积层，用于对输入的实部和虚部进行卷积操作实现频谱处理
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        # if use_se:
        #    if se_kwargs is None:
        #        se_kwargs = {}
        #    self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        # 如果设置了空间缩放因子，进行相应的插值
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]  # 获取输入张量 x 的高度和宽度维度的大小，并将其存储在变量 orig_size 中
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)
            # F.interpolate 函数来对输入张量 x 进行插值，实现空间缩放。参数解释如下：
            # scale_factor: 空间缩放的比例因子。它可以是一个浮点数，表示放大或缩小的倍数。
            # self.spatial_scale_mode 是一个字符串，可能是 "bilinear"、"nearest" 等，用于指定插值方法。
            # align_corners: 这是一个布尔值，用于控制是否对角点进行对齐。

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)

        # 对输入进行 FFT 变换
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        # 参数 dim 指定了在哪些维度上应用变换，这个参数使用了之前定义的 fft_dim。
        # self.fft_norm 是用于规范化的参数，它控制了傅里叶变换的规范化方式
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # 在傅里叶变换后，我们得到了一个复数张量 ffted。为了将其分成实部和虚部，
        # 这里使用了 torch.stack 函数来将实部和虚部叠加在一起，并且通过 dim=-1 参数指定最后一个维度。

        # 对频谱数据进行处理（可选的 clamp 和 remove 操作）
        clamp = False
        remove = False
        if clamp:  # torch.clamp 函数对傅里叶变换后的数据进行裁剪
            ffted = torch.clamp(ffted, min=-10, max=10)
        if remove:
            fftedmin10 = torch.clamp(ffted, min=10)
            fftedmax10 = torch.clamp(ffted, max=-10)
            ffted = torch.where(ffted > 0, fftedmax10, fftedmin10)

        # 重新排列维度
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # .permute重新排列 .contiguous确保数据在内存中是连续的
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        # -1 表示剩余的维度会被自动计算以保持总体元素数量不变

        # 添加频谱位置编码（可选）
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            # 使用 torch.linspace 函数在 [0, 1] 范围内生成均匀间隔的数值，数值个数与频谱数据的高度一致，得到 coords_vert，这个张量表示在频谱数据的竖直方向上的位置信息
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            # expand 函数将 coords_hor 以相同的方式扩展为与频谱数据大小一致的位置编码张量
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # if self.use_se:
        #    ffted = self.se(ffted)

        # 进行卷积操作并应用激活函数和批归一化
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        # 重新排列维度并转换为复数
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # 对频谱数据进行 IFFT 变换
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        # 如果进行了空间缩放，进行逆插值以恢复原始尺寸
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):  # 3
    # 对输入数据应用频谱变换。它包括使用平均池化进行下采样，然后是卷积和频谱处理

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):  # 2

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        # ratio_gin，ratio_gout控制本地路径和全局路径之间信息流的比例
        # 如果将ratio_gin设置为0.2，那么输入通道的20％将通过全局分支处理
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        # nn.Identity 恒等函数
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)  # 将张量沿着维度1分割成两个块
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            if g2l_gate == 0:
                out_xl = self.convl2l(x_l)  # + self.convg2l(x_g) * g2l_gate
            else:
                out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):  # 1
    # FFC块与批归一化和激活函数结合在一起

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        # dilation=1 没有dilation的标准卷积
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        # self.bn_l 对局部路径的输出进行规范化，通道数为 out_channels - global_channels
        self.bn_g = gnorm(global_channels)
        # self.bn_g 对全局路径的输出进行规范化，通道数为 global_channels

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):  # 在FFCResNetGenerator
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            # spatial_transform_kwargs包含了配置可学习的空间变换所需的关键字参数
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
            # :-self.conv1.ffc.global_in_num 从通道维度开始的前 channels - self.conv1.ffc.global_in_num 个元素
            # -self.conv1.ffc.global_in_num: 表示从通道维度开始的后 self.conv1.ffc.global_in_num 个元素到末尾
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):  # network也
    # 用于在残差块的最后将局部特征和全局特征连接在一起，然后将其传递给下一层。这可以在元组中的张量之间进行连接，使得网络能够同时处理局部和全局特征
    # 将局部特征 x_l 和全局特征 x_g 进行连接，然后将连接的结果作为输出。这可以确保下一层的输入同时包含了这两种类型的特征
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):  # ngf：第一层中生成器滤波器的数量 n_blocks：网络中的残差块数量
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                          activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

