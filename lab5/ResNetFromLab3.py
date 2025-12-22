# The script contains ResNet model from lab3 with all dependencies
# This model supports data augmentation during training


# Import of modules
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# PreEmphasis class from lab3/preproc.py
class PreEmphasis(torch.nn.Module):
    # Preemphasis procedure

    def __init__(self, coef: float = 0.97):
        
        super().__init__()
        
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.tensor) -> torch.tensor:
        
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        
        return F.conv1d(input, self.flipped_filter).squeeze(1)


# ResNet blocks from lab3/ResNetBlocks.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # 3x3 convolution with padding

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    # 1x1 convolution
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # Create basic ResNet block

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None, activation=nn.ReLU):
        
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = norm_layer(planes)
        self.relu       = activation(inplace=True)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = norm_layer(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# MaxoutLinear class from lab3/exercises_blank.py
class MaxoutLinear(nn.Module):
    # Maxout linear layer
    
    def __init__(self, *args, **kwargs):
        
        super(MaxoutLinear, self).__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        
        return torch.max(self.linear1(x), self.linear2(x))


# ResNet model from lab3/exercises_blank.py
class ResNet(nn.Module):
    # ResNet model for speaker recognition

    def __init__(self, block, layers, activation, num_filters, nOut, encoder_type='SP', n_mels=64, log_input=True, **kwargs):
        
        super(ResNet, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))

        self.inplanes     = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels       = n_mels
        self.log_input    = log_input

        self.torchfb        = torch.nn.Sequential(PreEmphasis(), 
                                                  torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
                                                                                       n_fft=512, 
                                                                                       win_length=400, 
                                                                                       hop_length=160, 
                                                                                       window_fn=torch.hamming_window, 
                                                                                       n_mels=n_mels))
        self.instancenorm   = nn.InstanceNorm1d(n_mels)

        self.conv1  = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(num_filters[0])
        self.relu   = activation(inplace=True)
        
        self.layer1 = self._make_layer(block, num_filters[0], layers[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=2, activation=activation)

        outmap_size = int(self.n_mels/8)

        self.attention = nn.Sequential(nn.Conv1d(num_filters[3]*outmap_size, 128, kernel_size=1), 
                                       nn.ReLU(), 
                                       nn.BatchNorm1d(128), 
                                       nn.Conv1d(128, num_filters[3]*outmap_size, kernel_size=1), 
                                       nn.Softmax(dim=2))
        
        if self.encoder_type == "SP":
            out_dim = num_filters[3]*outmap_size*2
        
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3]*outmap_size*2
        
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Sequential(MaxoutLinear(out_dim, nOut), nn.BatchNorm1d(nOut, affine=False))

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, activation=nn.ReLU):

        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=activation))
        self.inplanes = planes*block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=activation))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        
        return out

    def forward(self, x):

        with torch.no_grad():
            
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
                
                if self.log_input: x = x.log()
                
                x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size(0), -1, x.size(-1))

        if self.encoder_type == "ASP":
            w = self.attention(x)
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
            x = torch.cat((mu, sg), dim=1)
        elif self.encoder_type == "SP":
            mu = torch.mean(x, dim=2)
            sg = torch.sqrt((torch.mean(x ** 2, dim=2) - mu ** 2).clamp(min=1e-5))
            x = torch.cat((mu, sg), dim=1)
        else:
            raise ValueError('Undefined encoder')

        x = self.fc(x)

        return x


# MainModel function for lab5 (simplified version without trainfunc)
# Uses same default parameters as lab3: n_mels=40, encoder_type='SP'
def MainModel(nOut=512, encoder_type='SP', n_mels=40, log_input=True, **kwargs):
    # Create main model for speaker recognition using ResNet from lab3
    
    layers = [3, 4, 6, 3]
    activation = nn.ReLU
    num_filters = [32, 64, 128, 256]
    
    model = ResNet(BasicBlock, layers=layers, activation=activation, num_filters=num_filters, 
                   nOut=nOut, encoder_type=encoder_type, n_mels=n_mels, log_input=log_input, **kwargs)
    
    return model

