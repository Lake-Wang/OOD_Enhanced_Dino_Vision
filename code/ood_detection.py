from collections import deque
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models
import tqdm
from scipy.spatial import distance
from numpy.linalg import pinv

class OODDetector:
    def __init__(self, device, weights_path=None, threshold=0.95):
        """
        Initialize the OOD Detector.
        """
        self.device = device
        self.threshold = threshold

        # Resize transform to ensure images are 224x224
        self.resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        # Load the ResNet backbone
        if weights_path is None:
            self.encoder = self.load_default_dino_weights()
        else:
            self.encoder = torchvision_models.resnet50(pretrained=False)
            self.load_dino_weights(weights_path)

        self.encoder.eval()
        self.encoder = self.encoder.to(self.device)

        # Placeholders for dataset stats
        self.mean_vector = None
        self.threshold_value = None
        self.cov_inv = None

    def compute_dataset_stats(self, sampled_dataset):
        """
        Compute mean vector, inverse covariance, and Mahalanobis threshold from a sampled dataset.
        """
        with torch.no_grad():
            all_features = []
            for image in sampled_dataset:
                image = self.resize_transform(image).to(self.device)
                features = self.encoder(image.unsqueeze(0)).detach().squeeze()
                all_features.append(features)

            features_matrix = torch.stack(all_features)
            features_matrix = features_matrix.mean(dim=(2, 3))
            #features_matrix = features_matrix[:, :256]
            #print(features_matrix.shape)
            self.mean_vector = features_matrix.mean(dim=0, keepdim=True)
            #cov_matrix = torch.cov(features_matrix.T)
            #cov_matrix = torch.clamp(cov_matrix, min=0)
            #cov_matrix += 1e-5 * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
            #print('cov det checkig:', np.linalg.det(cov_matrix.cpu()))
            #self.cov_inv = torch.inverse(cov_matrix)
            #self.cov_inv = torch.tensor(pinv(cov_matrix.cpu().numpy()), device=self.mean_vector.device)

            #diff, cov_inv, distances = self.compute_mahalanobis_distance(features_matrix, self.mean_vector, self.cov_inv)
            #distances = [distance.mahalanobis(arr.cpu().flatten(), self.mean_vector.cpu().flatten(), self.cov_inv.cpu()) for arr in features_matrix]
            distances = [distance.euclidean(arr.cpu().flatten(), self.mean_vector.cpu().flatten()) for arr in features_matrix]
            dist_tensor = torch.Tensor(distances)
            #dist_tensor = torch.nan_to_num(torch.Tensor(distances), nan=1e6)
            #print(dist_tensor)
            nan_count = torch.isnan(dist_tensor).sum().item()
            #print(f"Number of NaN values: {nan_count}")

            self.threshold_value = torch.nanquantile(dist_tensor, self.threshold).item()
            print(f"Computed threshold: {self.threshold_value}")
            #print('diff:', diff)
            #print('cov_inv:', torch.min(cov_inv))

    def detect(self, images):
        """
        Identify rare samples using Mahalanobis distance for batches of global and local views.
        """
        #global_image = [images[i][j] for i in range(batch_size) for j in [0,1]]
        #global_image = [images[j][i] for i in range(batch_size) for j in [0,1]]
        global_image = [images[j][i] for i in range(len(images[0])) for j in [0,1]]

        images_resized = [self.resize_transform(image).to(self.device) for image in global_image]
        images_resized = torch.stack(images_resized)

        #print("Reshaped images_resized:", images_resized.shape)

        with torch.no_grad():
            features = self.encoder(images_resized).detach()
            features = features.mean(dim=(2, 3))
            #features = features[:, :256]
            #distances = self.compute_mahalanobis_distance(features, self.mean_vector, self.cov_inv)
            #distances = [distance.mahalanobis(arr.cpu().flatten(), self.mean_vector.cpu().flatten(), self.cov_inv.cpu()) for arr in features]
            distances = [distance.euclidean(arr.cpu().flatten(), self.mean_vector.cpu().flatten()) for arr in features]
            nan_count = torch.isnan(torch.Tensor(distances)).sum().item()
            #print(f"Number of NaN values: {nan_count}")
            dist_tensor = torch.nan_to_num(torch.Tensor(distances), nan=1e6)

            #print(dist_tensor)
            rare_indices = (dist_tensor > self.threshold_value).nonzero(as_tuple=True)[0]

        return torch.unique(rare_indices // 2)

    def load_default_dino_weights(self):
        """
        Load the default DINO ResNet50 (ImageNet-pretrained).
        """
        print("Loading default DINO ResNet50 weights from torch.hub...")
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        model = torch.nn.Sequential(*(list(model.children())[:-2]))  # Use up to the second-last layer
        return model

    def extract_features(self, image_tensor):
        with torch.no_grad():
          feature_map = self.encoder(image_tensor.unsqueeze(0).to(self.device)).squeeze(0)  # Shape: (2048, 7, 7)
          feature_vector = feature_map.mean(dim=(1, 2))  # Shape: (2048,)
        return feature_vector.cpu().numpy()

    def load_dino_weights(self, weights_path):
        """
        Load weights for the ResNet backbone from a DINO checkpoint.

        Args:
        - weights_path (str): Path to the .pth file.
        """
        print(f"Loading DINO weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device)
        if "teacher" in checkpoint:
            resnet_weights = checkpoint["teacher"]  # DINO-specific weight structure
            state_dict = {k.replace("backbone.", ""): v for k, v in resnet_weights.items() if "backbone" in k}
            self.encoder.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Invalid checkpoint structure for DINO. Expected 'teacher' key.")

    ### @staticmethod
    # def compute_mahalanobis_distance(features, mean, cov_inv):
    #     """
    #     Compute Mahalanobis distance for given features.

    #     Args:
    #     - features (torch.Tensor): Feature vectors (N x D).
    #     - mean (torch.Tensor): Mean vector (1 x D).
    #     - cov_inv (torch.Tensor): Inverse covariance matrix (D x D).

    #     Returns:
    #     - distances (torch.Tensor): Mahalanobis distances (N).
    #     """
    #     diff = features - mean
    #     dist = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))
    #     dist = torch.sqrt(torch.matmul(diff, torch.matmul(inv_cov_matrix, diff.T)))

    #     return diff, cov_inv, dist


from collections import deque

import torch
from collections import deque

class MemoryBuffer:
    def __init__(self, max_size=32):
        """
        Initialize a fixed-size memory buffer.
        Each entry is a list containing 2 global views and 6 local views.
        """
        self.buffer = deque(maxlen=max_size)

    def add(self, samples):
        """
        Add a new sample (list of tensors: 2 global + 6 local views) to the buffer.
        Args:
            samples (list of tensors): [global1, global2, local1, ..., local6].
        """
        self.buffer.append(samples)

    def get(self):
        """
        Retrieve the buffer content as a list of lists of tensors.
        Returns:
            list: A list where each element is a list of tensors.
        """
        return list(self.buffer)

    def concat_with_batch(self, img_):
        """
        Concatenate the memory buffer with the current batch (img_) to match its shape.
        Args:
            img_ (list of tensors): 
                - img_[0] and img_[1]: Global views of shape [32, 3, 224, 224].
                - img_[2:8]: Local views of shape [32, 3, 96, 96].
        Returns:
            list of tensors: Updated img_ with concatenated memory buffer content.
        """
        # Separate global and local tensors from memory buffer
        buffer_global = torch.stack(
            [torch.stack([entry[0], entry[1]], dim=0) for entry in self.buffer], dim=0
        )  # Shape: [32, 2, 3, 224, 224]
        buffer_local = torch.stack(
            [torch.stack(entry[2:], dim=0) for entry in self.buffer], dim=0
        )  # Shape: [32, 6, 3, 96, 96]

        # Reshape buffer for concatenation
        buffer_global = buffer_global.view(-1, 3, 224, 224)  # Shape: [64, 3, 224, 224]
        buffer_local = buffer_local.view(-1, 3, 96, 96)      # Shape: [192, 3, 96, 96]

        # Concatenate buffer with the current batch
        updated_img = []
        updated_img.append(torch.cat([img_[0], buffer_global[:32]], dim=0))  # Global view 1
        updated_img.append(torch.cat([img_[1], buffer_global[32:]], dim=0))  # Global view 2

        for i in range(6):  # Concatenate with the 6 local views
            updated_img.append(torch.cat([img_[2 + i], buffer_local[32 * i: 32 * (i + 1)]], dim=0))

        return updated_img

    def clear(self):
        """
        Clear the memory buffer.
        """
        self.buffer.clear()



