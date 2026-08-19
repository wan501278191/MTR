"""
Trajectory Diffusion Refiner for MTR.

Innovation: Borrowed from DiffSemanticFusion_MaplessQCNet.
Uses DDPM with pyramid multi-resolution noise to refine MTR's deterministic predictions.
- Conditioning: MTR's 6-mode prediction serves as local_cond
- Noise: Pyramid noise (multi-resolution, temporally correlated)
- Schedule: Cosine beta schedule, 20 steps (fast inference)
- FiLM conditioning for timestep injection
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pyramid_noise_like(x, discount=0.9):
    """Generate multi-resolution (pyramid) noise.
    
    Unlike i.i.d. Gaussian, this produces spatially-correlated noise at multiple scales,
    which respects temporal smoothness of trajectory data.
    
    Borrowed from DiffSemanticFusion_MaplessQCNet/utils/diffusion.py
    """
    b, c, t = x.shape
    noise = torch.zeros_like(x)
    for i in range(4):
        scale = max(t // (2 ** i), 1)
        coarse = torch.randn(b, c, scale, device=x.device)
        if scale != t:
            coarse = F.interpolate(coarse, size=t, mode='linear', align_corners=False)
        noise = noise + coarse * (discount ** i)
    return noise / (noise.std() + 1e-8)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (from diffusion policy / DDPM)."""
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning (from DiffSemanticFusion UNet1D)."""
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        # FiLM: predict per-channel scale and bias from conditioning
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        
        self.act = nn.Mish()
        
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        
        # FiLM conditioning
        scale_bias = self.cond_proj(cond)  # (B, 2*C)
        scale, bias = scale_bias.chunk(2, dim=-1)  # (B, C) each
        h = h * (1 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)
        h = self.act(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    """1D U-Net for trajectory denoising with FiLM conditioning.
    
    Adapted from DiffSemanticFusion_MaplessQCNet/modules/uncertainty_conditional_unet1d.py
    
    Args:
        input_dim: dimension of trajectory (2 for xy)
        cond_dim: dimension of conditioning (MTR feature or deterministic prediction)
        hidden_dim: hidden dimension
    """
    def __init__(self, input_dim=2, cond_dim=128, hidden_dim=128, diffusion_step_embed_dim=64):
        super().__init__()
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Conditioning projection (MTR prediction -> cond features)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        cond_total = hidden_dim * 2  # time + cond
        
        # Downsample path
        self.down1 = ConditionalResidualBlock1D(input_dim, hidden_dim, cond_total)
        self.down2 = ConditionalResidualBlock1D(hidden_dim, hidden_dim * 2, cond_total)
        self.pool = nn.MaxPool1d(2)
        
        # Bottleneck
        self.mid = ConditionalResidualBlock1D(hidden_dim * 2, hidden_dim * 2, cond_total)
        
        # Upsample path
        self.up1 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, 2, stride=2)
        self.up2 = ConditionalResidualBlock1D(hidden_dim * 2, hidden_dim, cond_total)
        self.up3 = ConditionalResidualBlock1D(hidden_dim, input_dim, cond_total)
        
    def forward(self, x, t, cond):
        """
        Args:
            x: (B, T, input_dim) noisy trajectory
            t: (B,) diffusion timestep
            cond: (B, cond_dim) conditioning features
        Returns:
            (B, T, input_dim) predicted noise
        """
        x = x.permute(0, 2, 1)  # (B, input_dim, T)
        
        # Embed timestep and condition
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        c_emb = self.cond_proj(cond)  # (B, hidden_dim)
        cond_full = torch.cat([t_emb, c_emb], dim=-1)  # (B, hidden_dim*2)
        
        # Downsample
        h1 = self.down1(x, cond_full)  # (B, hidden, T)
        h2 = self.down2(self.pool(h1), cond_full)  # (B, hidden*2, T//2)
        
        # Bottleneck
        h = self.mid(h2, cond_full)
        
        # Upsample with skip connections
        h = self.up1(h)  # (B, hidden, T)
        if h.shape[-1] != h1.shape[-1]:
            h = F.interpolate(h, size=h1.shape[-1], mode='linear', align_corners=False)
        h = torch.cat([h, h1], dim=1)  # (B, hidden*2, T)
        h = self.up2(h, cond_full)  # (B, hidden, T)
        h = self.up3(h, cond_full)  # (B, input_dim, T)
        
        return h.permute(0, 2, 1)  # (B, T, input_dim)


class TrajectoryDiffusionRefiner(nn.Module):
    """Diffusion-based trajectory refinement module.
    
    Uses DDPM with cosine schedule and pyramid noise to generate diverse
    trajectory corrections conditioned on MTR's deterministic prediction.
    
    During training: learns to denoise (standard DDPM epsilon prediction).
    During inference: generates diverse refined trajectories via sampling.
    """
    def __init__(self, traj_dim=2, cond_dim=256, hidden_dim=128,
                 num_diffusion_steps=20, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.num_steps = num_diffusion_steps
        self.traj_dim = traj_dim
        
        # Cosine beta schedule (better for small step counts)
        betas = self._cosine_beta_schedule(num_diffusion_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', F.pad(alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Noise prediction network
        self.noise_pred_net = ConditionalUNet1D(
            input_dim=traj_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
        )
    
    @staticmethod
    def _cosine_beta_schedule(num_steps, beta_start=0.0001, beta_end=0.02):
        """Cosine schedule (squaredcos_cap_v2 from DiffSemanticFusion)."""
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, beta_start, beta_end)
    
    def add_noise(self, x_original, t, noise=None):
        """Add pyramid noise to trajectory (forward diffusion)."""
        if noise is None:
            noise = pyramid_noise_like(x_original.permute(0, 2, 1)).permute(0, 2, 1)
        
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[t])[:, None, None]
        sqrt_one_minus = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None]
        
        noisy = sqrt_alpha * x_original + sqrt_one_minus * noise
        return noisy, noise
    
    def compute_loss(self, traj_original, cond_features):
        """Training loss: MSE between predicted and actual noise.
        
        Args:
            traj_original: (B, T, traj_dim) ground truth trajectory
            cond_features: (B, cond_dim) MTR prediction features
        Returns:
            loss: scalar MSE loss
        """
        B = traj_original.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=traj_original.device)
        
        noisy_traj, noise = self.add_noise(traj_original, t)
        pred_noise = self.noise_pred_net(noisy_traj, t, cond_features)
        
        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def sample(self, cond_features, traj_shape, num_samples=1):
        """DDPM sampling loop for inference.
        
        Args:
            cond_features: (B, cond_dim)
            traj_shape: (B, T, traj_dim)
            num_samples: number of diverse samples per condition
        Returns:
            (num_samples, B, T, traj_dim) refined trajectory samples
        """
        B, T, D = traj_shape
        samples = []
        
        for _ in range(num_samples):
            # Start from pyramid noise
            x = pyramid_noise_like(
                torch.randn(B, D, T, device=cond_features.device)
            ).permute(0, 2, 1)  # (B, T, D)
            
            for t in reversed(range(self.num_steps)):
                t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)
                
                pred_noise = self.noise_pred_net(x, t_batch, cond_features)
                
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                alpha_cumprod_prev = self.alphas_cumprod_prev[t]
                beta = self.betas[t]
                
                mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise)
                
                if t > 0:
                    noise = pyramid_noise_like(
                        torch.randn(B, D, T, device=x.device)
                    ).permute(0, 2, 1)
                    sigma = torch.sqrt(beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))
                    x = mean + sigma * noise
                else:
                    x = mean
            
            samples.append(x)
        
        return torch.stack(samples, dim=0)  # (num_samples, B, T, D)
    
    @torch.no_grad()
    def refine_trajectory(self, mtr_prediction, cond_features, refinement_weight=0.3):
        """Refine MTR's deterministic prediction with diffusion.
        
        Instead of sampling from pure noise, we start from MTR's prediction
        (partial denoising — observation-conditioned refinement).
        
        Args:
            mtr_prediction: (B, T, traj_dim) MTR's best trajectory
            cond_features: (B, cond_dim) MTR feature embedding
            refinement_weight: how much to perturb (0=no change, 1=full denoise)
        Returns:
            (B, T, traj_dim) refined trajectory
        """
        B, T, D = mtr_prediction.shape
        
        # Start from MTR prediction + partial noise (not full noise)
        # This is the key innovation: deterministic prediction as starting point
        start_step = int(self.num_steps * refinement_weight)
        
        if start_step == 0:
            return mtr_prediction
        
        t_start = torch.full((B,), start_step, device=mtr_prediction.device, dtype=torch.long)
        noisy, _ = self.add_noise(mtr_prediction, t_start)
        x = noisy
        
        for t in reversed(range(start_step)):
            t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)
            pred_noise = self.noise_pred_net(x, t_batch, cond_features)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod_prev[t]
            beta = self.betas[t]
            
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise)
            
            if t > 0:
                noise = pyramid_noise_like(
                    torch.randn(B, D, T, device=x.device)
                ).permute(0, 2, 1)
                sigma = torch.sqrt(beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))
                x = mean + sigma * noise
            else:
                x = mean
        
        return x
