'''
#Code: https://github.com/jaeyoon1603/Retrieval-RegionalAttention
#Paper: BMVC2018《Regional Attention Based Deep Feature for Image Retrieval》
'''
def get_rmac_region_coordinates(H, W, L):
    # Almost verbatim from Tolias et al Matlab implementation.
    # Could be heavily pythonized, but really not worth it...
    # Desired overlap of neighboring regions
    ovr = 0.4
    # Possible regions for the long dimension
    steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
    w = np.minimum(H, W)
    
    b = (np.maximum(H, W) - w) / (steps - 1)
    # steps(idx) regions for long dimension. The +1 comes from Matlab
    # 1-indexing...
    idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1
    
    # Region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx
    elif H > W:
        Hd = idx
    
    regions_xywh = []
    for l in range(1, L+1):
        wl = np.floor(2 * w / (l + 1))
        wl2 = np.floor(wl / 2 - 1)
        # Center coordinates
        if l + Wd - 1 > 0:
            b = (W - wl) / (l + Wd - 1)
        else:
            b = 0
        cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
        # Center coordinates
        if l + Hd - 1 > 0:
            b = (H - wl) / (l + Hd - 1)
        else:
            b = 0
        cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2
    
        for i_ in cenH:
            for j_ in cenW:
                regions_xywh.append([j_, i_, wl, wl])
    
    # Round the regions. Careful with the borders!
    for i in range(len(regions_xywh)):
        for j in range(4):
            regions_xywh[i][j] = int(round(regions_xywh[i][j]))
        if regions_xywh[i][0] + regions_xywh[i][2] > W:
            regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
        if regions_xywh[i][1] + regions_xywh[i][3] > H:
            regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
    return np.array(regions_xywh).astype(np.float32)

def pack_regions_for_network(all_regions):
    n_regs = np.sum([len(e) for e in all_regions])
    R = np.zeros((n_regs, 5), dtype=np.float32)
    cnt = 0
    # There should be a check of overflow...
    for i, r in enumerate(all_regions):
        try:
            R[cnt:cnt + r.shape[0], 0] = i
            R[cnt:cnt + r.shape[0], 1:] = r
            cnt += r.shape[0]
        except:
            continue
    assert cnt == n_regs
    R = R[:n_regs]
    # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
    R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
    R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
    return R

class L2Normalization(nn.Module):
    def __init__(self):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(L2Normalization, self).__init__()
        self.eps = 1e-8
        
    def forward(self, x):
        if x.is_cuda:
            caped_eps = Variable(torch.Tensor([self.eps])).cuda(torch.cuda.device_of(x).idx)
        else:
            caped_eps = Variable(torch.Tensor([self.eps]))
        x = torch.div(x.transpose(0,1),x.max(1)[0]).transpose(0,1) # max_normed
        norm = torch.norm(x,2,1) + caped_eps.expand(x.size()[0])
        y = torch.div(x.transpose(0,1),norm).transpose(0,1)
        return y


class Shift(nn.Module):
    def __init__(self,dim):
        super(Shift, self).__init__()
        #self.bias = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.Tensor(dim))
        self.bias.data.uniform_(-0.006, -0.03)
    def forward(self, input):
        output = torch.add(input, self.bias)
        
        return output
    
class RoIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        
        outputs = Variable(torch.zeros(num_rois, num_channels,
                                       self.pooled_height,
                                       self.pooled_width))
        if features.is_cuda:
            outputs = outputs.cuda(torch.cuda.device_of(features).idx)

        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].item())
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = torch.round(roi[1:]* self.spatial_scale).data.cpu().numpy().astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    is_empty = (hend <= hstart) or(wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        outputs[roi_ind, :, ph, pw] = torch.max(
                            torch.max(data[:, hstart:hend, wstart:wend], 1, keepdim = True)[0], 2, keepdim = True)[0].view(-1)  # noqa

        return outputs
    
class ContextAwareRegionalAttentionNetwork(nn.Module):
    def __init__(self, spatial_scale, pooled_height = 1, pooled_width = 1):
        super(ContextAwareRegionalAttentionNetwork, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        
        self.conv_att_1 = nn.Conv1d(4096, 64, 1, padding=0)
        self.sp_att_1 = nn.Softplus()
        self.conv_att_2 = nn.Conv1d(64, 1, 1, padding=0)
        self.sp_att_2 = nn.Softplus()
        

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        
        outputs = Variable(torch.zeros(num_rois, num_channels*2,
                                       self.pooled_height,
                                       self.pooled_width))
        if features.is_cuda:
            outputs = outputs.cuda(torch.cuda.device_of(features).idx)
            
        # Based on roi pooling code of pytorch but, the only difference is to change max pooling to mean pooling
        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].item())
            roi_start_w, roi_start_h, roi_end_w, roi_end_h =  torch.round(roi[1:]* self.spatial_scale).data.cpu().numpy().astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    is_empty = (hend <= hstart) or(wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        # mean pooling with both of regional feature map and global feature map
                        outputs[roi_ind, :, ph, pw] = torch.cat((torch.mean(
                            torch.mean(data[:, hstart:hend, wstart:wend], 1, keepdim = True), 2, keepdim = True).view(-1)
                            ,torch.mean(
                            torch.mean(data, 1, keepdim = True), 2, keepdim = True).view(-1)), 0 )  # noqa
                        
        # Reshpae
        outputs = outputs.squeeze(2).squeeze(2)
        outputs = outputs.transpose(0,1).unsqueeze(0) # (1, # channel, #batch * # regions)
        #Calculate regional attention weights with context-aware regional feature vectors
        k = self.sp_att_1(self.conv_att_1(outputs))
        k = self.sp_att_2(self.conv_att_2(k)) # (1, 1, #batch * # regions)
        k = torch.squeeze(k,1)
        
        return k
    
class NetForTraining(nn.Module):

    def __init__(self, res_net, num_classes):
        super(NetForTraining, self).__init__()
        
        self.l2norm = L2Normalization()
        
        #RoI max pooling
        self.r_mac_pool = RoIPool(1,1,0.03125)
        self.region_attention = ContextAwareRegionalAttentionNetwork(spatial_scale = 0.03125)
        self._initialize_weights()
        
        #This is for Imagenet classification. get the weights of fc from off-the-shelf Resnet101.
        self.fc = nn.Linear(2048,num_classes)
        self.resnet = res_net
        dic = self.resnet.state_dict()
        self.fc.weight.data = dic['fc.weight']
        self.fc.bias.data = dic['fc.bias']
          
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
   
    def forward(self, x ):
        #Calculate R-MAC regions (Region sampling)
        batched_rois =  [get_rmac_region_coordinates(x.shape[2],x.shape[3],5) for i in range(x.shape[0])]
        rois = Variable(torch.FloatTensor(pack_regions_for_network(batched_rois)))
        
        h = x
        #Extract feature map
        h,_ = self.resnet(x) #( #batch, #channel, #h, #w)
        #R-MAC
        g = self.r_mac_pool(h,rois) 
        g = g.squeeze(2).squeeze(2) # (#batch * # regions, #channel)
        #Regional Attention
        g2 = self.region_attention(h,rois)
        g2 = g2.squeeze(0).squeeze(0)# (# batch * region)
        #weighted mean
        g = torch.mul(g.transpose(1,0),g2).transpose(1,0)  # regional weighted feature (# batch * region, #channel)
        g = g.contiguous()
        g = g.view(torch.Size([h.size(0), -1 ,h.size(1)])) # (#batch, # region, # channel)
        g = torch.transpose(g,1,2)    # (#batch * #channel, #region)
        
        g_feat = torch.mean(g,2) #mean
        g_feat = self.l2norm(g_feat)
        
        g = self.fc(g_feat)

        return g_feat, g