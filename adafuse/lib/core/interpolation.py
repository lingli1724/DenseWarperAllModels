import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Spatial-Temporal Graph Convolution (ST-GCN)
class SpatialTemporalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(1, 1), padding=(1, 1)):
        super(SpatialTemporalGraphConv, self).__init__()
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size[0], 1), stride=(stride[0], 1), padding=(padding[0], 0))
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size[1]), stride=(1, stride[1]), padding=(0, padding[1]))

    def forward(self, x):
        # x shape: (batch_size, channels, joints, frames)
        x = self.spatial_conv(x)  # Spatial convolution
        x = self.temporal_conv(x)  # Temporal convolution
        return x

# 2. Encoder: Modify for matching Decoder
class Encoder(nn.Module):
    def __init__(self, in_channels, num_joints, num_frames_in):
        super(Encoder, self).__init__()

        # First block: Keep frame size, increase channels
        self.conv1 = SpatialTemporalGraphConv(in_channels, 128, kernel_size=(3, 2), stride=(1, 1))
        
        # Second block: Halve frames, increase channels
        self.conv2 = SpatialTemporalGraphConv(128, 128, kernel_size=(3, 2), stride=(1, 1))  # Reduce frames by half
        
        # Third block: Further halve frames
        self.conv3 = SpatialTemporalGraphConv(128, 256, kernel_size=(3, 2), stride=(1, 1))  # Reduce frames again
        
        # Fourth block: Keep channels, keep frames at 8
        self.conv4 = SpatialTemporalGraphConv(256, 256, kernel_size=(3, 2), stride=(1, 1))  # Keep frames

        self.conv5 = SpatialTemporalGraphConv(256, 256, kernel_size=(3, 2), stride=(1, 1))  # Keep frames

        self.conv6 = SpatialTemporalGraphConv(256, 256, kernel_size=(3, 2), stride=(1, 1))  # Keep frames
        # Final output size: [batch_size, 256, num_joints, 8]
        self.num_joints = num_joints
        self.num_frames_out = 8  # Set as fixed 8 frames after the convolutions

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Output shape: [batch_size, 128, num_joints, num_frames]
        x = F.relu(self.conv2(x))  # Output shape: [batch_size, 128, num_joints, num_frames/2]
        x = F.relu(self.conv3(x))  # Output shape: [batch_size, 256, num_joints, num_frames/4]
        x = F.relu(self.conv4(x))  # Output shape: [batch_size, 256, num_joints, 8]
        x = F.relu(self.conv5(x))  # Output shape: [batch_size, 256, num_joints, 8]
        x = F.relu(self.conv6(x))  # Output shape: [batch_size, 256, num_joints, 8]
        return x

# 3. Decoder: Adjust temporal kernel size and stride to avoid reducing frames too fast
class Decoder(nn.Module):
    def __init__(self, num_joints, num_frames_out):
        super(Decoder, self).__init__()

        # Decoder starts with input shape [batch_size, 256, num_joints, 8]
        # First block: reduce feature channels from 256 to 128, frames remain the same
        self.conv1 = SpatialTemporalGraphConv(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Second block: reduce frames from 8 to 4, keep feature channels at 128
        self.conv2 = SpatialTemporalGraphConv(128, 128, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0))
        
        # Third block: reduce frames from 4 to 2, feature channels remain at 128
        self.conv3 = SpatialTemporalGraphConv(128, 128, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0))
        
        # Fourth block: reduce frames from 2 to 1, feature channels from 128 to 96
        self.conv4 = SpatialTemporalGraphConv(128, 96, kernel_size=(3, 2), stride=(1, 1), padding=(1, 0))  # No further frame reduction

        # Deconvolution: upsample frames back to the desired output frames (64 frames)
        self.deconv1 = nn.ConvTranspose2d(96, 192, kernel_size=(3, 1), stride=(1, 1),padding=(1, 0))  # Upsample to frames = 1

        # Final FC layer to output joint coordinates (x, y, z) for 64 frames
        self.fc = nn.Linear(num_joints*192, num_joints*3*num_frames_out)  # Output shape [batch_size, 3, num_joints, 64]

        self.njoints=num_joints

        self.num_frames_out=num_frames_out

    def forward(self, x):
        # Apply the spatial-temporal convolutions to reduce the frame count
        x = F.relu(self.conv1(x))  # Output shape: [batch_size, 128, num_joints, 8]
        x = F.relu(self.conv2(x))  # Output shape: [batch_size, 128, num_joints, 4]
        x = F.relu(self.conv3(x))  # Output shape: [batch_size, 128, num_joints, 2]
        x = F.relu(self.conv4(x))  # Output shape: [batch_size, 96, num_joints, 1]
        # Apply deconvolution to upsample back to 64 frames
        x = F.relu(self.deconv1(x))  # Output shape: [batch_size, 192, num_joints, 1]

        # Fully connected layer to output 3D joint positions
        x = x.view(x.shape[0],-1)
        x = self.fc(x)  # Output shape: [batch_size, 3, num_joints, 64]
        x = x.view(x.shape[0],3,self.njoints,self.num_frames_out)
        return x

# 4. AutoEncoder: Combine Encoder and Decoder
class AutoEncoder(nn.Module):
    def __init__(self, in_channels, num_joints, num_frames_in, num_frames_out):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_channels, num_joints, num_frames_in)
        self.decoder = Decoder(num_joints, num_frames_out)

    def forward(self, x):
        encoded = self.encoder(x)  # Encoded features: [batch_size, 256, num_joints, 8]
        decoded = self.decoder(encoded)  # Decoded 3D coordinates: [batch_size, 3, num_joints, 64]
        return decoded

# 5. Loss Functions
# Triplet Loss (Action Consistency)
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.relu(self.margin + pos_dist - neg_dist)
        return loss.mean()

# InfoNCE Loss (Contrastive Loss)
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        pos_score = torch.exp(F.cosine_similarity(anchor, positive) / self.temperature)
        neg_scores = torch.exp(F.cosine_similarity(anchor.unsqueeze(1), negatives) / self.temperature)
        neg_sum = torch.sum(neg_scores, dim=1)
        loss = -torch.log(pos_score / (pos_score + neg_sum))
        return loss.mean()

# Interpolation Loss (MSE for Motion Continuity)
class InterpolationLoss(nn.Module):
    def __init__(self):
        super(InterpolationLoss, self).__init__()

    def forward(self, pred, target):
        return F.mse_loss(pred, target)

# 6. Training Step
def train_step(split,model, optimizer, data, ground_truth, positive=None, negative=None, negatives=None):
    if split == 'train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    # Forward pass
    encoded = model.module.encoder(data)
    decoded = model.module.decoder(encoded)

    # Compute losses
    triplet_loss_fn = TripletLoss()
    nce_loss_fn = InfoNCELoss()
    interp_loss_fn = InterpolationLoss()

    # Compute each loss
    #print(encoded.shape)
    #print(decoded.shape)
    #loss_triplet = triplet_loss_fn(encoded, positive, negative)
    #loss_nce = nce_loss_fn(encoded, positive, negatives)
    loss_interp = interp_loss_fn(decoded, ground_truth)

    # Total loss (keeping all coefficients as 1, according to the paper)
    #total_loss = loss_triplet + loss_nce + loss_interp
    total_loss=loss_interp
    if split == 'train':
        total_loss.backward()
        optimizer.step()

    return total_loss.item()

# 7. Example usage
if __name__ == '__main__':
    batch_size = 8
    num_joints = 25
    num_frames_in = 2  # Input frame count
    num_frames_out = 5  # Output frame count (after interpolation)
    in_channels = 3  # (x, y, z) for each joint

    # Example data
    input_data = torch.randn(batch_size, in_channels, num_joints, num_frames_in)
    positive_samples = torch.randn(batch_size, 256, num_joints, 8)  # Positive samples for triplet loss
    negative_samples = torch.randn(batch_size, 256, num_joints, 8)  # Negative samples for triplet loss
    negative_batch = torch.randn(batch_size, 5, 256, num_joints, 8)  # Batch of negative samples for InfoNCE loss
    ground_truth = torch.randn(batch_size, 3, num_joints, num_frames_out)  # Ground truth for interpolation loss

    model = AutoEncoder(in_channels, num_joints, num_frames_in, num_frames_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training step
    loss = train_step(model, optimizer, input_data, ground_truth, positive_samples, negative_samples, negative_batch)
    print(f"Training loss: {loss}")

class PoseInterpolator():
    def __init__(self):
        self.parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.njoints=len(self.parents)
    def slerp(self,p0,p1,t):
        p0_norm = torch.norm(p0)
        p1_norm = torch.norm(p1)

        if p0_norm == 0 or p1_norm == 0:
            return (1 - t) * p0 + t * p1 

        p0_unit = p0 / p0_norm
        p1_unit = p1 / p1_norm
        dot_product = torch.dot(p0_unit, p1_unit)
        omega = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        
        sin_omega = torch.sin(omega)
        if sin_omega < 1e-6:
            return (1 - t) * p0 + t * p1

        factor_1 = torch.sin((1 - t) * omega) / sin_omega
        factor_2 = torch.sin(t * omega) / sin_omega
        interpolated_joint = factor_1 * p0 + factor_2 * p1

        return interpolated_joint

    def interpolate(self, source_pose, target_pose, num_steps=5):
        interpolated_poses = []
        for step in range(num_steps):
            t = step / (num_steps - 1)
            interpolated_pose = torch.zeros_like(source_pose)
            interpolated_pose[0] = source_pose[0] + (target_pose[0] - source_pose[0]) * t

            for joint in range(1,len(self.parents)):
                interpolated_pose[joint] = self.slerp(source_pose[joint] - source_pose[self.parents[joint]], target_pose[joint] - target_pose[self.parents[joint]], t)
            
            for joint in range(1,len(self.parents)):
                interpolated_pose[joint] = interpolated_pose[joint] + interpolated_pose[self.parents[joint]]

            interpolated_poses.append(interpolated_pose)

        return torch.stack(interpolated_poses)
    

