import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np # For np.prod

class HerPN(nn.Module):
    """
    HerPN (Hermite Polynomial with basis-wise Normalization) activation module.
    Supports multi-dimensional inputs (e.g., 1D, 2D, 3D feature maps).

    As proposed in the paper "AESPA: Accuracy Preserving Low-degree
    Polynomial Activation for Fast Private Inference"[cite: 5].
    This implementation focuses on degree d=2, using bases h_0, h_1, h_2.
    The HerPN block is defined by the formula (Equation 5 in the paper):
    f(x) = gamma * sum_{i=0 to d} (f_tilde_i * (h_i(x) - mu_i) / sqrt(sigma_i^2 + epsilon)) + beta [cite: 132]
    """
    def __init__(self, num_features: int, degree: int = 2, epsilon: float = 1e-5, momentum: float = 0.1):
        """
        Args:
            num_features (int): Number of features (channels) in the input tensor.
            degree (int): The highest degree of Hermite polynomials to use (d).
                          The paper focuses on d=2, resulting in h_0, h_1, h_2[cite: 124].
            epsilon (float): A small constant added to the variance for numerical stability.
            momentum (float): Momentum for updating running mean and variance.
        """
        super(HerPN, self).__init__()
        self.num_features = num_features
        self.degree = degree
        self.epsilon = epsilon
        self.momentum = momentum

        if self.degree != 2:
            raise NotImplementedError(
                "This implementation currently supports only degree d=2 "
                "(using h_0, h_1, h_2) as primarily evaluated in the AESPA paper."
            )

        # Fixed Hermite coefficients (f_tilde_i) for ReLU expansion [cite: 123]
        # These correspond to the orthonormal Hermite polynomials h_n(x) = (1/sqrt(n!))H_n(x) [cite: 117]
        self.register_buffer('f_tilde_0', torch.tensor(1.0 / math.sqrt(2.0 * math.pi))) # For h_0
        self.register_buffer('f_tilde_1', torch.tensor(0.5))                             # For h_1
        self.register_buffer('f_tilde_2', torch.tensor(1.0 / (2.0 * math.sqrt(math.pi)))) # For h_2

        # Learnable scale (gamma) and shift (beta) parameters applied after summation [cite: 131, 132]
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Buffers for running means and variances for each Hermite basis polynomial (h_i)
        # This enables basis-wise normalization[cite: 130].
        for i in range(self.degree + 1):
            self.register_buffer(f'running_mean_h{i}', torch.zeros(num_features))
            self.register_buffer(f'running_var_h{i}', torch.ones(num_features))
        
        self.hermite_coeffs_map = {
            0: self.f_tilde_0,
            1: self.f_tilde_1,
            2: self.f_tilde_2
        }

    def _hermite_polynomials(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Computes the first few orthonormal Hermite polynomials h_n(x).
        h_n(x) = (1/sqrt(n!)) * H_n(x) [cite: 117]
        H_0(x) = 1
        H_1(x) = x
        H_2(x) = x^2 - 1
        """
        # x shape: (N, C, ...) where C is num_features
        h_list = []
        # h_0(x) = 1
        h_list.append(torch.ones_like(x))
        if self.degree >= 1:
            # h_1(x) = x
            h_list.append(x)
        if self.degree >= 2:
            # h_2(x) = (x^2 - 1) / sqrt(2)
            h_list.append((1.0 / math.sqrt(2.0)) * (x.pow(2) - 1.0))
        return h_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HerPN module.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, [D1, D2, ...]),
                              where N is batch size, C is num_features,
                              and D1, D2, ... are spatial dimensions.
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        if x.numel() == 0: # Handle empty input like PyTorch's BatchNorm
            return x

        if x.ndim < 2:
            raise ValueError(
                f"Expected input with at least 2 dimensions (N, C, ...), but got {x.ndim} dimensions"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input with {self.num_features} channels (dim 1), "
                f"but got {x.shape[1]} channels for input shape {x.shape}"
            )

        hermite_bases = self._hermite_polynomials(x) # List of tensors, e.g., [h_0(x), h_1(x), h_2(x)]

        aggregated_normalized_sum = torch.zeros_like(x)

        # Determine dimensions for mean/variance calculation (batch + spatial dims)
        dims_to_reduce = (0,) + tuple(range(2, x.ndim))

        # Reshape view for broadcasting (1, C, 1, 1, ...)
        param_reshape_view = (1, self.num_features) + tuple([1] * (x.ndim - 2))

        for i in range(self.degree + 1):
            h_i = hermite_bases[i] # Current Hermite basis, e.g., h_0(x)
            f_tilde_i = self.hermite_coeffs_map[i] # Corresponding Hermite coefficient
            
            running_mean_buffer = getattr(self, f'running_mean_h{i}')
            running_var_buffer = getattr(self, f'running_var_h{i}')

            if self.training:
                batch_mean = h_i.mean(dim=dims_to_reduce, keepdim=False) # Shape (C,)
                batch_var_biased = h_i.var(dim=dims_to_reduce, unbiased=False, keepdim=False) # Shape (C,)
                
                running_mean_buffer.mul_(1.0 - self.momentum).add_(self.momentum * batch_mean.detach())
                
                # Calculate num_elements_per_channel for unbiased variance factor
                num_elements_per_channel = x.numel() // x.shape[1] if x.shape[1] != 0 else 0

                if num_elements_per_channel > 1:
                    batch_var_unbiased_for_running = batch_var_biased.detach() * \
                                                     (num_elements_per_channel / (num_elements_per_channel - 1.0))
                else:
                    batch_var_unbiased_for_running = batch_var_biased.detach()
                
                running_var_buffer.mul_(1.0 - self.momentum).add_(self.momentum * batch_var_unbiased_for_running)
                
                mean_for_norm = batch_mean.view(param_reshape_view)
                var_for_norm = batch_var_biased.view(param_reshape_view)
            else: # Evaluation mode
                mean_for_norm = running_mean_buffer.view(param_reshape_view)
                var_for_norm = running_var_buffer.view(param_reshape_view)

            h_i_normalized = (h_i - mean_for_norm) / torch.sqrt(var_for_norm + self.epsilon)
            aggregated_normalized_sum = aggregated_normalized_sum + f_tilde_i * h_i_normalized
        
        gamma_reshaped = self.gamma.view(param_reshape_view)
        beta_reshaped = self.beta.view(param_reshape_view)
        output = gamma_reshaped * aggregated_normalized_sum + beta_reshaped
        
        return output

    def get_polynomial_coeffs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the effective polynomial coefficients a0, a1, a2 for the function
        f(x) = a0 + a1*x + a2*x^2, based on the module's current (evaluation mode) parameters.
        This is valid for degree=2 configuration.

        The HerPN output is:
        gamma * sum_{i=0 to 2} (f_tilde_i * (h_i(x) - mu_i) / sqrt(sigma_i^2 + epsilon)) + beta

        h_0(x) = 1
        h_1(x) = x
        h_2(x) = (x^2 - 1) / sqrt(2)

        This method expands this formula into a0 + a1*x + a2*x^2 and returns
        the coefficients a0, a1, a2, each of shape (num_features,).
        These coefficients are derived using the running means (mu_i) and
        running variances (sigma_i^2) stored in the buffers, which are typically
        used during evaluation.
        """
        if self.degree != 2:
            raise NotImplementedError(
                "Polynomial coefficient calculation is only implemented for degree d=2."
            )

        # Retrieve f_tilde coefficients from registered buffers
        ft0 = self.f_tilde_0
        ft1 = self.f_tilde_1
        ft2 = self.f_tilde_2

        # Retrieve running means and variances (these are mu_i and sigma_i^2 from the formula)
        # These are typically used in evaluation mode.
        mu0 = self.running_mean_h0
        mu1 = self.running_mean_h1
        mu2 = self.running_mean_h2
        
        var0 = self.running_var_h0
        var1 = self.running_var_h1
        var2 = self.running_var_h2

        # Normalization factors: nf_i = 1 / sqrt(sigma_i^2 + epsilon)
        nf0 = 1.0 / torch.sqrt(var0 + self.epsilon)
        nf1 = 1.0 / torch.sqrt(var1 + self.epsilon)
        nf2 = 1.0 / torch.sqrt(var2 + self.epsilon)

        # Coefficients of the intermediate sum (before gamma and beta):
        # AggregatedSum = S0_terms + S1_terms*x + S2_terms*x^2
        
        # S0_terms: Constant part from the sum of normalized Hermite polynomials
        # Contribution from h0: ft0 * (1 - mu0) * nf0
        # Contribution from h1: -ft1 * mu1 * nf1
        # Contribution from h2: -ft2 * ( (1/sqrt(2)) + mu2 ) * nf2
        s0_terms = ft0 * (1.0 - mu0) * nf0 \
                 - ft1 * mu1 * nf1 \
                 - ft2 * ((1.0 / math.sqrt(2.0)) + mu2) * nf2
        
        # S1_terms: Coefficient of x from the sum
        # Contribution from h1: ft1 * nf1 * x
        s1_terms = ft1 * nf1
        
        # S2_terms: Coefficient of x^2 from the sum
        # Contribution from h2: ft2 * (1/sqrt(2)) * nf2 * x^2
        s2_terms = ft2 * (1.0 / math.sqrt(2.0)) * nf2

        # Final polynomial coefficients: Output = gamma * AggregatedSum + beta
        # a0 + a1*x + a2*x^2 = gamma * (S0_terms + S1_terms*x + S2_terms*x^2) + beta
        a0 = self.gamma * s0_terms + self.beta
        a1 = self.gamma * s1_terms
        a2 = self.gamma * s2_terms
        
        return a0, a1, a2

    def extra_repr(self) -> str:
        return f'{self.num_features}, degree={self.degree}, epsilon={self.epsilon}, momentum={self.momentum}'
