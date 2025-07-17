import time
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
import torch
import torch.nn as nn
from numpy import array
from numpy import count_nonzero
import numpy as np

class dvs_pooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)

        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        init_time_MLP = time.time()
        T, B, C, H, W = x.shape
        identity = x
        non_zero = count_nonzero(np.array(x.cpu().detach().numpy()))
        size = np.array(x.cpu().detach().numpy()).size
        print(f"MLP sparsity x: {1 - non_zero/size}")
        x = self.fc1_lif(x)
        non_zero = count_nonzero(np.array(x.cpu().detach().numpy()))
        size = np.array(x.cpu().detach().numpy()).size
        print(f"MLP sparsity postLIF x after lif: {1 - non_zero/size}")

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        non_zero = count_nonzero(np.array(x.cpu().detach().numpy()))
        size = np.array(x.cpu().detach().numpy()).size
        print(f"MLP sparsity postBN x after fc1: {1 - non_zero/size}")
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        non_zero = count_nonzero(np.array(x.cpu().detach().numpy()))
        size = np.array(x.cpu().detach().numpy()).size
        print(f"MLP sparsity postLIF x after fc2: {1 - non_zero/size}")
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        non_zero = count_nonzero(np.array(x.cpu().detach().numpy()))
        size = np.array(x.cpu().detach().numpy()).size
        print(f"MLP sparsity postBN x after fc2ConvBn: {1 - non_zero/size}")

        x = x + identity
        non_zero = count_nonzero(np.array(x.cpu().detach().numpy()))
        size = np.array(x.cpu().detach().numpy()).size
        print(f"MLP sparsity x after identity final: {1 - non_zero/size}")
        end_time_MLP = time.time()
        print(f"[Time] - MLP: {end_time_MLP - init_time_MLP:.4f} seconds")
        return x, hook


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mode="direct_xor",
        dvs=False,
        layer=0,
        attention_mode="T_STAtten",
        chunk_size=2,
        spike_mode="lif"
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.attention_mode = attention_mode
        if dvs:
            self.pool = dvs_pooling()
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        self.talking_heads_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)
        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.mode = mode
        self.layer = layer
        self.chunk_size = chunk_size
    
    def forward(self, x, hook=None):
        init_time_SSA_encoder = time.time()
        T, B, C, H, W = x.shape
        head_dim = C // self.num_heads
        identity = x
        N = H * W
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()
        if self.dvs:
            x_pool = self.pool(x)

        x_for_qkv = x.flatten(0, 1)
        
        non_zero = count_nonzero(np.array(x_for_qkv.cpu().detach().numpy()))
        size = np.array(x_for_qkv.cpu().detach().numpy()).size
        print(f"sparsity x_for_qkv: {1 - non_zero/size}")
        print("shape x_for_qkv:", x_for_qkv.shape)
        # Q
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        if self.dvs:
            q_conv_out = self.pool(q_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = (q_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous())
        
        non_zero = count_nonzero(np.array(q.cpu().detach().numpy()))
        size = np.array(q.cpu().detach().numpy()).size
        print(f"sparsity q: {1 - non_zero/size}")
        print("q shape:", q.shape)
        # K
        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (k_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous())
    
        non_zero = count_nonzero(np.array(k.cpu().detach().numpy()))
        size = np.array(k.cpu().detach().numpy()).size
        print(f"sparsity k: {1 - non_zero/size}")
        print("k shape:", k.shape)
        # V
        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (v_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous())
        # Shape: (T B head N C//head)
        
        non_zero = count_nonzero(np.array(v.cpu().detach().numpy()))
        size = np.array(v.cpu().detach().numpy()).size
        print(f"sparsity v: {1 - non_zero/size}")
        print("v shape:", v.shape)
        
        ###### Attention #####
        if self.attention_mode == "STAtten":
            init_time_STAtten = time.time()
            if self.dvs:
                scaling_factor = 1 / (H*H*self.chunk_size)
            else:
                scaling_factor = 1 / H
            init_time_reshape = time.time_ns()
            # Vectorized Attention
            num_chunks = T // self.chunk_size
            # Reshape q, k, v to process all chunks at once: (num_chunks, B, num_heads, chunk_size, N, head_dim)
            q_chunks = q.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            k_chunks = k.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            v_chunks = v.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            
            non_zero = count_nonzero(np.array(q_chunks.cpu().detach().numpy()))
            size = np.array(q_chunks.cpu().detach().numpy()).size
            print(f"sparsity Attention q_chunks: {1 - non_zero/size}")
            non_zero = count_nonzero(np.array(k_chunks.cpu().detach().numpy()))
            size = np.array(k_chunks.cpu().detach().numpy()).size
            print(f"sparsity Attention k_chunks: {1 - non_zero/size}")
            non_zero = count_nonzero(np.array(v_chunks.cpu().detach().numpy()))
            size = np.array(v_chunks.cpu().detach().numpy()).size
            print(f"sparsity Attention v_chunks: {1 - non_zero/size}")
            
            # Merge chunk_size and N dimensions: (num_chunks, B, num_heads, chunk_size * N, head_dim)
            q_chunks = q_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            k_chunks = k_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            v_chunks = v_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            end_time_reshape = time.time_ns()
            print(f"[Time] - Reshape: {end_time_reshape - init_time_reshape:.4f} nanoseconds")
            non_zero = count_nonzero(np.array(q_chunks.cpu().detach().numpy()))
            size = np.array(q_chunks.cpu().detach().numpy()).size
            print(f"sparsity Attention postReshape q_chunks: {1 - non_zero/size}")
            non_zero = count_nonzero(np.array(k_chunks.cpu().detach().numpy()))
            size = np.array(k_chunks.cpu().detach().numpy()).size
            print(f"sparsity Attention postReshape k_chunks: {1 - non_zero/size}")
            non_zero = count_nonzero(np.array(v_chunks.cpu().detach().numpy()))
            size = np.array(v_chunks.cpu().detach().numpy()).size
            print(f"sparsity Attention postReshape v_chunks: {1 - non_zero/size}")
            print("q_chunks shape:", q_chunks.shape)
            print("k_chunks shape:", k_chunks.shape)
            print("v_chunks shape:", v_chunks.shape)

            init_time_matmul = time.time_ns()
            # Compute attention for all chunks simultaneously
            attn = torch.matmul(k_chunks.transpose(-2, -1),
                                v_chunks) * scaling_factor  # (num_chunks, B, num_heads, head_dim, head_dim)
            out = torch.matmul(q_chunks, attn)  # (num_chunks, B, num_heads, chunk_size * N, head_dim)
            
            end_time_matmul = time.time_ns()
            print(f"[Time] - Matmul: {end_time_matmul - init_time_matmul:.4f} nanoseconds")
            non_zero = count_nonzero(np.array(out.cpu().detach().numpy()))
            size = np.array(out.cpu().detach().numpy()).size
            print(f"sparsity Attention out: {1 - non_zero/size}")

            init_time_reshapeTimeSpace = time.time_ns()
            # Reshape back to separate temporal and spatial dimensions
            out = out.reshape(num_chunks, B, self.num_heads, self.chunk_size, N, head_dim).permute(0, 3, 1, 2, 4, 5)
            # Flatten chunks back to T: (T, B, num_heads, N, head_dim)
            output = out.reshape(T, B, self.num_heads, N, head_dim)
            end_time_reshapeTimeSpace = time.time_ns()
            print(f"[Time] - Reshape Time-Space: {end_time_reshapeTimeSpace - init_time_reshapeTimeSpace:.4f} nanoseconds")
            print("shape :", output.shape)
            for t in range(T):
                non_zero = count_nonzero(np.array(output[t].cpu().detach().numpy()))
                size = np.array(output[t].cpu().detach().numpy()).size         
                print(f"sparsity Attention postReshape chunk - {t} output: {1 - non_zero/(size)}")
            init_time_Traspose_lif = time.time_ns()
            x = output.transpose(4,3).reshape(T, B, C, N).contiguous() # (T, B, head, C//h, N)
            x = self.attn_lif(x).reshape(T, B, C, H, W)
            end_time_Traspose_lif = time.time_ns()
            print(f"[Time] - Transpose and LIF: {end_time_Traspose_lif - init_time_Traspose_lif:.4f} nanoseconds")
            for t in range(T):
                non_zero = count_nonzero(np.array(x[t].cpu().detach().numpy()))
                size = np.array(x[t].cpu().detach().numpy()).size
                print(f"sparsity Attention postReshape chunk - {t} x: {1 - non_zero/(size)}")
            print("x shape:", x.shape)
            if self.dvs:
                x = x.mul(x_pool)
                x = x + x_pool

                        
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_after_qkv"] = x
            init_time_projBN = time.time()
            x = (
                self.proj_bn(self.proj_conv(x.flatten(0, 1)))
                .reshape(T, B, C, H, W)
                .contiguous()
            )
            end_time_projBN = time.time()
            print(f"[Time] - Projection and BN: {end_time_projBN - init_time_projBN:.4f} seconds")
            print("x shape after proj:", x.shape)



        """Spike-driven Transformer"""
        if self.attention_mode == "SDT":
            init_time_SDT = time.time()
            kv = k.mul(v)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
            if self.dvs:
                kv = self.pool(kv)
            kv = kv.sum(dim=-2, keepdim=True)
            kv = self.talking_heads_lif(kv)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
            x = q.mul(kv)
            if self.dvs:
                x = self.pool(x)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

            x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
            x = (
                self.proj_bn(self.proj_conv(x.flatten(0, 1)))
                .reshape(T, B, C, H, W)
                .contiguous()
            )

        assert self.attention_mode not in ["STAtten, SDT"] 

        x = x + identity
        non_zero = count_nonzero(np.array(x.cpu().detach().numpy()))
        size = np.array(x.cpu().detach().numpy()).size
        print(f"sparsity x: {1 - non_zero/size}")
        
        end_time_SSA_encoder = time.time()
        print(f"[Time] - SSA Encoder: {end_time_SSA_encoder - init_time_SSA_encoder:.4f} seconds")
        return x, v, hook
    

class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        attention_mode="STAtten",
        chunk_size=2,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            mode=attn_mode,
            dvs=dvs,
            layer=layer,
            attention_mode=attention_mode,
            chunk_size=chunk_size
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_mode=spike_mode,
            layer=layer,
        )

    def forward(self, x, hook=None):
       
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook
