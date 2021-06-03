import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio as ta, torchaudio.functional as taF

from .config import Config, ConfigList, ConfigDict, configurable


@configurable()
class Reverberation(nn.Module):
    def __init__(self, sample_rate=16000, kernel_size=4096, max_reverb_time=0.5, min_reverb_time=0.0, auto_remake_kernel=True):
        minlen = kernel_size
        self.pad_left = kernel_size - 1
        self.pad_right = 0        
        self.sample_rate = sample_rate
        self.max_reverb_time = max_reverb_time
        self.min_reverb_time = min_reverb_time
        self.kernel_size = kernel_size
        self.auto_remake_kernel = auto_remake_kernel
        super().__init__(minlen, pad_left=self.pad_left, pad_right=self.pad_right)
        
        
        
    def make_kernel(self):
        device = 'cpu' if not hasattr(self, 'kernel') else self.kernel.device
        reverb_time = np.random.rand()**4 * (self.max_reverb_time - self.min_reverb_time) + self.min_reverb_time
        if reverb_time <= 3.0 / self.sample_rate:
            kernel = torch.zeros(self.kernel_size, device=device)
            kernel[-1] = 1.0
        else:
            alpha = np.log(1000) / reverb_time
            end = (self.kernel_size - 1) / self.sample_rate
            time = -torch.linspace(0, end, self.kernel_size, device=device).flip(dims=[0])
            envelope = torch.exp(alpha * time)
            mask = 1.0 - torch.randint(20, [self.kernel_size], device=device).bool().float() + 1e-10
            kernel = envelope * mask * torch.randn(self.kernel_size, device=device)
            kernel /= kernel.max()
        self.kernel = nn.Parameter(kernel.reshape(1, 1, -1), requires_grad=False)
        
    def reset(self, recursive=False):
        super().reset(recursive)
        self.make_kernel()        
        

    def forward(self, wav):  
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if self.training:
            if self.auto_remake_kernel:
                self.make_kernel() 
            wav = F.pad(wav, (self.pad_left, self.pad_right))
            return F.conv1d(wav, self.kernel)
        else:
            return wav
    


@configurable()
class STFT(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=512, hop_length=256, window_fn=torch.hann_window, 
                 pad_left=None, pad_right=None, pad_mode='constant', pad_value=0, normalize=True, trainable=False):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or n_fft // 2
        self.pad_left = pad_left if pad_left else n_fft // 2
        self.pad_right = pad_right if pad_right else n_fft - self.pad_left - 1
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        _wpad_left = (n_fft - self.win_length) // 2
        _wpad_right = n_fft - self.win_length - _wpad_left
        self.window = F.pad(window_fn(window_length=self.win_length), (_wpad_left, _wpad_right), 
                            mode='constant', value=0)
        self.win_sum = self.window.sum()
        self.normalize = normalize
        self.trainable = trainable
        super().__init__(minlen=n_fft, stride=self.hop_length, pad_left=self.pad_left, pad_right=self.pad_right, 
                         pad_mode=self.pad_mode, pad_value=self.pad_value)
        self.make_kernel()
        
    def make_kernel(self):
        n_fft = self.n_fft
        self.window = window = self.window.unsqueeze(dim=0)
        n_freq = self.n_freq
        _min_omega = 0.0
        _max_omega = math.pi
        time = torch.linspace(0, n_fft - 1, n_fft, dtype=torch.float32).unsqueeze(dim=0)
        omega = torch.linspace(_min_omega, _max_omega, n_freq, dtype=torch.float32).unsqueeze(dim=-1)
        omega_time = omega * time
        ckernel = torch.cos(omega_time) * window
        skernel = torch.sin(omega_time) * window
        self.kernel = nn.Parameter(torch.cat([ckernel, skernel], dim=0).unsqueeze(dim=1), requires_grad=self.trainable)     
        

    def forward(self, wav, wav_len=None):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        wav = F.pad(wav, (self.pad_left, self.pad_right), mode=self.pad_mode, value=self.pad_value)
        csspec = F.conv1d(wav, self.kernel, stride=self.hop_length)
        if self.normalize:
            csspec /= self.win_sum
        cspec, sspec = torch.chunk(csspec, 2, dim=1)
        spec = torch.view_as_complex(torch.stack([cspec, sspec], dim=-1))
        spec_len = torch.floor((wav_len + self.pad_left + self.pad_right - (self.n_fft - 1) - 1) / \
                                  float(self.hop_length) + 1).to(torch.int32) if wav_len is not None else None
        return spec, spec_len
        

        
        
@configurable()
class ColoredNoise(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, max_noise_energy=0.01, min_noise_energy=0.0, max_beta=2.0, min_beta=-1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft        
        self.max_noise_energy = max_noise_energy
        self.min_noise_energy = min_noise_energy
        self.max_beta = max_beta
        self.min_beta = min_beta
        n_freq = n_fft // 2
        delta_f = sample_rate / 2 / n_freq
        freq = torch.linspace(delta_f, sample_rate / 2, n_freq)
        inv_freq = freq**(-1)
        self.inv_freq = nn.Parameter(inv_freq.reshape(1, -1, 1, 1), requires_grad=False)      
        
        
    def forward(self, spec):
        if not self.training:
            return spec
        cspec, sspec = spec.real, spec.imag
        rel_noise_energy = np.random.rand()**4 * (self.max_noise_energy - self.min_noise_energy) + self.min_noise_energy
        beta = np.random.rand() * (self.max_beta - self.min_beta) + self.min_beta
        bsize, fsize, tsize = spec.shape
        device = spec.device
        wav_energy = (spec.real**2 + spec.imag**2).mean().detach()
        noise = torch.view_as_complex(torch.randn(bsize, fsize-1, tsize, 2, device=device) * self.inv_freq**(beta/2))
        noise_energy = (noise.real**2 + noise.imag**2).mean()
        amp = (rel_noise_energy * wav_energy / noise_energy).sqrt()
        noise *= amp 
        spec[:,1:,:] = spec[:,1:,:] + noise
        return spec    
    
    
                 
@configurable()
class STFT2Spec(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, power=2.0, n_mels=128, mel_trainable=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        n_freq = n_fft // 2 + 1
        self.power = power
        self.n_mels = n_mels
        self.mel_trainable = mel_trainable
        if n_mels > 0:
            self.mel_scale = nn.Parameter(taF.create_fb_matrix(n_freq, f_min=0.0, f_max=self.sample_rate/2, 
                                         n_mels=n_mels, sample_rate=self.sample_rate), requires_grad=mel_trainable)
    
    def forward(self, spec):
        spec = spec.real**2 + spec.imag**2
        if self.power != 2.0:
            spec = spec**(self.power / 2.0)
        if self.n_mels and self.n_mels > 0:
            spec = torch.matmul(spec.transpose(1, 2), self.mel_scale).transpose(1, 2)
        return spec
    
    
        
@configurable()    
class Decibel(nn.Module):
    def __init__(self, power=2.0, min_decibel=-100, normalize=True):
        super().__init__()
        self.power = power
        self.min_decibel = min_decibel
        self.normalize = normalize
        self.min_spec = None if min_decibel is None else 10**(min_decibel / 20 * power)
        
    def forward(self, x):
        if self.min_decibel:
            x = torch.max(x, torch.full_like(x, self.min_spec))
        x = 20 / self.power * torch.log10(x)
        if self.min_decibel and self.normalize:
            x = (x - self.min_decibel) / (-self.min_decibel)
        return x
        

    
@configurable()
class SpecAugment(nn.Module):
    def __init__(self, speed_stdev=0.1, max_tmask_size=10, max_fmask_size=10, mask_mode='mean', mask_value=0.0):
        super().__init__()
        self.speed_stdev = speed_stdev
        self.max_tmask_size = max_tmask_size
        self.max_fmask_size = max_fmask_size
        self.mask_mode = mask_mode
        self.mask_value = mask_value
        assert mask_mode in ('mean', 'constant')
        
    def forward(self, spec, spec_len=None):
        if not self.training:
            return spec, spec_len
        
        if self.speed_stdev and self.speed_stdev > 0:
            ssd = self.speed_stdev
            min_s = max(-0.5, -3 * ssd)
            max_s = min(0.5, 3 * ssd)
            scale_factor = 1.0 + np.clip(ssd * np.random.randn(), min_s, max_s)
            spec = F.interpolate(spec, scale_factor=scale_factor, mode='linear')
            spec_len = torch.floor(scale_factor * spec_len).to(torch.int32) if spec_len is not None else None
            
        bsize, fsize, tsize = spec.shape
        max_tmask_size = min(self.max_tmask_size, tsize)
        max_fmask_size = min(self.max_fmask_size, fsize)
        tmask_size = np.random.randint(max_tmask_size+1, size=bsize)
        tmask_start = np.random.randint(tsize - tmask_size)
        tmask_end = tmask_start + tmask_size
        fmask_size = np.random.randint(max_fmask_size+1, size=bsize)
        fmask_start = np.random.randint(fsize - fmask_size)
        fmask_end = fmask_start + fmask_size
        mask_value = torch.full([bsize], fill_value=self.mask_value, device=spec.device) \
                   if self.mask_mode == 'constant' else spec.mean(dim=(1,2)).detach()
        for b in range(bsize):
            spec[b,fmask_start[b]:fmask_end[b],:] = mask_value[b]
            spec[b,:,tmask_start[b]:tmask_end[b]] = mask_value[b]
        return spec, spec_len    
        
        
        
@configurable()    
class WavEmbed(nn.Module):
    def __init__(self, reverberation=Reverberation(), stft=STFT(), noise=ColoredNoise(), 
                 stft2spec=STFT2Spec(), decibel=Decibel(), spec_augment=SpecAugment(), 
                 add_reverberation=True, add_noise=True, add_spec_augment=True):
        super().__init__()
        self.reverberation = reverberation
        self.stft = stft
        self.noise = noise
        self.stft2spec = stft2spec
        self.decibel = decibel
        self.spec_augment = spec_augment
        self.add_reverberation = add_reverberation
        self.add_noise = add_noise
        self.add_spec_augment = add_spec_augment
        
        
    def forward(self, wav, wav_len=None):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if self.add_reverberation and self.reverberation is not None:
            wav = self.reverberation(wav)
        spec, spec_len = self.stft(wav, wav_len)
        if self.add_noise and self.noise is not None:
            spec = self.noise(spec)
        spec = self.stft2spec(spec)
        if self.decibel is not None:
            spec = self.decibel(spec)
        if self.add_spec_augment and self.spec_augment is not None:
            spec, spec_len = self.spec_augment(spec, spec_len)
        return spec, spec_len
