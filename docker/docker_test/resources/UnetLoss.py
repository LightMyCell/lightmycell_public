import torch
from torch import nn

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        self.mae_loss = nn.L1Loss(reduction='none')  # Set reduction to 'none'

    def forward(self, out_images, target_images, epoch):
        num_pairs = target_images.shape[0]
        num_channels = target_images.size(1)  # Get the number of channels
    
        # The (pred, gt) pairs in the batches are evaluated separately, and then the loss is the average
        # If this wasn't so, and let's say there were 2 pairs per batch, wouldn't it be looking through 
        # the channels for both targets at the same time? As in if one of them for example had a proper
        # target for channel 1 (as in it has any value different from 0), it would consider both of them
        # to have a proper target for channel 1? like here -> target_images[:, channel, :, :]
        
        pair_losses = []
        for pair_idx in range(num_pairs):								
													
            total_loss = 0										
		    # Iterate over channels									 											
            for channel in range(num_channels):							
		        # Create mask for the current channel
                mask = torch.any(target_images[pair_idx, channel, :, :] != 0) 

		        # Compute element-wise MAE loss for the current channel
                image_loss = self.mae_loss(out_images[pair_idx, channel, :, :], target_images[pair_idx, channel, :, :])               

                if mask:
                    total_loss += torch.sum(image_loss)							# Mean instead of sum? image_loss.mean()

            pair_losses.append(total_loss)

		
        # Calculate the mean loss across all pairs
        combined_loss = sum(pair_losses) / max(len(pair_losses), 1)  # Avoid division by zero
        #print("Losses:")
        #print(sum(pair_losses))
        #print(max(len(pair_losses), 1))
        #print(combined_loss)
        
        return combined_loss
        
if __name__ == "__main__":
    g_loss = GenLoss()

