'''
#Code: https://github.com/PyRetri/PyRetri
       https://github.com/almazan/deep-image-retrieval
       https://github.com/noagarcia/keras_rmac
#Paper: ICLR2016《Particular object retrieval with integral max-pooling of CNN activations》
        IJCV2017《End-to-end Learning of Deep Visual Representations for Image Retrieval》
'''
class RMAC():
    """
    Regional Maximum activation of convolutions (R-MAC).
    c.f. https://arxiv.org/pdf/1511.05879.pdf
    Args:
        level_n (int): number of levels for selecting regions.
    """
    def __init__(self,level_n:int):
        super(RMAC, self).__init__()
        self.first_show = True
        self.cached_regions = dict()
        self.level_n = level_n

    def _get_regions(self, h: int, w: int) -> List:
        """
        Divide the image into several regions.
        Args:
            h (int): height for dividing regions.
            w (int): width for dividing regions.
        Returns:
            regions (List): a list of region positions.
        """
        if (h, w) in self.cached_regions:
            return self.cached_regions[(h, w)]

        m = 1
        n_h, n_w = 1, 1
        regions = list()
        if h != w:
            min_edge = min(h, w)
            left_space = max(h, w) - min(h, w)
            iou_target = 0.4
            iou_best = 1.0
            while True:
                iou_tmp = (min_edge ** 2 - min_edge * (left_space // m)) / (min_edge ** 2)

                # small m maybe result in non-overlap
                if iou_tmp <= 0:
                    m += 1
                    continue

                if abs(iou_tmp - iou_target) <= iou_best:
                    iou_best = abs(iou_tmp - iou_target)
                    m += 1
                else:
                    break
            if h < w:
                n_w = m
            else:
                n_h = m

        for i in range(self.level_n):
            region_width = int(2 * 1.0 / (i + 2) * min(h, w))
            step_size_h = (h - region_width) // n_h
            step_size_w = (w - region_width) // n_w

            for x in range(n_h):
                for y in range(n_w):
                    st_x = step_size_h * x
                    ed_x = st_x + region_width - 1
                    assert ed_x < h
                    st_y = step_size_w * y
                    ed_y = st_y + region_width - 1
                    assert ed_y < w
                    regions.append((st_x, st_y, ed_x, ed_y))

            n_h += 1
            n_w += 1

        self.cached_regions[(h, w)] = regions
        return regions

    def __call__(self, fea:torch.tensor) -> torch.tensor:
        final_fea = None
        if fea.ndimension() == 4:
            h, w = fea.shape[2:]       
            regions = self._get_regions(h, w)
            for _, r in enumerate(regions):
                st_x, st_y, ed_x, ed_y = r
                region_fea = (fea[:, :, st_x: ed_x, st_y: ed_y].max(dim=3)[0]).max(dim=2)[0]
                region_fea = region_fea / torch.norm(region_fea, dim=1, keepdim=True)
                if final_fea is None:
                    final_fea = region_fea
                else:
                    final_fea = final_fea + region_fea
        else:# In case of fc feature.
            assert fea.ndimension() == 2
            if self.first_show:
                print("[RMAC Aggregator]: find 2-dimension feature map, skip aggregation")
                self.first_show = False
            final_fea = fea
        return final_fea

#https://github.com/almazan/deep-image-retrieval
def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x


class ResNet_RMAC_FPN(ResNet):
    """ ResNet for RMAC (without ROI pooling)
    """
    def __init__(self, block, layers, model_name, out_dim=None, norm_features=False,
                       pooling='gem', gemp=3, center_bias=0, mode=1, n_cls=0,
                       dropout_p=None, without_fc=False, **kwargs):
        ResNet.__init__(self, block, layers, 0, model_name, **kwargs)
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias
        self.mode = mode
        self.n_cls = n_cls

        dim1 = 256 * block.expansion
        dim2 = 512 * block.expansion
        if out_dim is None: out_dim = dim1 + dim2
        #FPN
        if self.mode == 1:
            self.conv1x5 = nn.Conv2d(dim2, dim1, kernel_size=1, stride=1, bias=False)
            self.conv3c4 = nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling == 'gem':
            self.adpoolx5 = GeneralizedMeanPoolingP(norm=gemp)
            self.adpoolc4 = GeneralizedMeanPoolingP(norm=gemp)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.fc = nn.Linear(768 * block.expansion, out_dim)
        self.fc_name = 'fc'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x):
        x4, x5 = ResNet.forward(self, x, -1)
        
        if self.n_cls: 
            x = nn.Linear(x5, self.n_cls) #for classification
            return x

        # FPN
        if self.mode == 1:
            c5 = F.interpolate(x5, size=x4.shape[-2:], mode='nearest')

            c5 = self.conv1x5(c5)
            c5 = self.relu(c5)
            x4 = x4 + c5
            x4 = self.conv3c4(x4)
            x4 = self.relu(x4)

        if self.dropout is not None:
            x5 = self.dropout(x5)
            x4 = self.dropout(x4)

        if self.detach:
            # stop the back-propagation here, if needed
            x5 = Variable(x5.detach())
            x5 = self.id(x5)  # fake transformation
            x4 = Variable(x4.detach())
            x4 = self.id(x4)  # fake transformation

        # global pooling
        x5 = self.adpoolx5(x5)
        x4 = self.adpoolc4(x4)

        x = torch.cat((x4, x5), 1)

        if self.norm_features:
            x = l2_normalize(x, axis=1)

        x.squeeze_()
        if not self.without_fc:
            x = self.fc(x)

        x = l2_normalize(x, axis=-1)
        return x


def resnet18_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    kwargs.pop('scales', None)
    return backbone(BasicBlock, [2, 2, 2, 2], 'resnet18', **kwargs)

def resnet50_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 6, 3], 'resnet50', **kwargs)

def resnet101_fpn_rmac(backbone=ResNet_RMAC_FPN, n_cls=0, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 23, 3], 'resnet101', n_cls=0, **kwargs)

def resnet101_fpn0_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 23, 3], 'resnet101', mode=0, **kwargs)

def resnet152_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 8, 36, 3], 'resnet152', **kwargs)