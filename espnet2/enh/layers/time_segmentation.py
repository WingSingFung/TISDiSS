import torch
import torch.nn as nn
from packaging.version import parse as V

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")


class TimeSegmentation(nn.Module):
    """Time Segmentation and Merging Module
    
    Used to split time series data into overlapping segments and then remerge them.
    This is particularly useful for processing long sequences, reducing memory usage and improving computational efficiency.
    """
    
    def __init__(self, segment_size: int = 96):
        """Initialize Time Segmentation Module
        
        Args:
            segment_size (int): Size of the segmentation chunks, default is 96
        """
        super().__init__()
        self.segment_size = segment_size
    
    def split_feature(self, x):
        """Split features into overlapping segments
        
        Args:
            x (torch.Tensor): Input features [B, D, T]
            
        Returns:
            torch.Tensor: Segmented features [B, D, segment_size, n_chunks]
        """
        B, D, T = x.size()
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.segment_size, 1),
            padding=(self.segment_size, 0),
            stride=(self.segment_size // 2, 1),
        )
        return unfolded.reshape(B, D, self.segment_size, -1)

    def merge_feature(self, x, length=None):
        """Merge segmented chunks back into original sequence
        
        Args:
            x (torch.Tensor): Segmented features [B, D, L, n_chunks]
            length (int, optional): Target length, automatically calculated if None
            
        Returns:
            torch.Tensor: Merged features [B, D, length]
        """
        B, D, L, n_chunks = x.size()
        hop_size = self.segment_size // 2
        if length is None:
            length = (n_chunks - 1) * hop_size + L
            padding = 0
        else:
            padding = (0, L)

        seq = x.reshape(B, D * L, n_chunks)
        x = torch.nn.functional.fold(
            seq,
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )
        norm_mat = torch.nn.functional.fold(
            input=torch.ones_like(seq),
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )

        x /= norm_mat

        return x.reshape(B, D, length)
