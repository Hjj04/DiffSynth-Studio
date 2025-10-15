# examples/wanvideo/model_training/dataset.py
import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

class RealVideoDataset(Dataset):
    """
    A PyTorch Dataset for loading videos based on a metadata CSV file.
    The CSV file is expected to have 'video_path' and 'text' columns.
    """
    def __init__(self,
                 metadata_csv_path: str,
                 videos_root_dir: str,
                 num_frames: int = 16,
                 height: int = 64,
                 width: int = 64):
        """
        Args:
            metadata_csv_path (str): Path to the metadata.csv file.
            videos_root_dir (str): The root directory where video files are stored.
            num_frames (int): The number of frames to sample from each video.
            height (int): The target height to resize frames to.
            width (int): The target width to resize frames to.
        """
        super().__init__()
        
        self.metadata = pd.read_csv(metadata_csv_path)
        self.videos_root_dir = videos_root_dir
        self.num_frames = num_frames
        
        # Define the transformations to be applied to each frame
        self.transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True), # Converts uint8 [0,255] to float32 [0,1]
            transforms.Resize((height, width), antialias=True),
        ])

    def __len__(self) -> int:
        """Returns the total number of videos in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        """
        Retrieves a single data point: a text caption and its corresponding video tensor.
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds")
            
        # Get video info from metadata
        video_info = self.metadata.iloc[idx]
        relative_video_path = video_info['video_path']
        text_caption = video_info['text']
        
        # Construct the full path to the video file
        full_video_path = os.path.join(self.videos_root_dir, relative_video_path)

        try:
            # Read the video file using torchvision.io
            # This returns a tensor of shape [T, H, W, C] with uint8 values
            video_frames, _, _ = torchvision.io.read_video(full_video_path, pts_unit='sec', output_format='TCHW')

            # --- Frame Sampling Logic ---
            total_frames = video_frames.shape[0]
            if total_frames >= self.num_frames:
                # If the video is long enough, take the first `num_frames` frames
                sampled_frames = video_frames[:self.num_frames]
            else:
                # If the video is too short, loop the frames to reach `num_frames`
                multiples = self.num_frames // total_frames
                remainder = self.num_frames % total_frames
                sampled_frames = torch.cat(
                    [video_frames] * multiples + [video_frames[:remainder]],
                    dim=0
                )
            
            # Apply transformations to the sampled frames
            # The output will be of shape [T, C, H, W] with float32 values [0, 1]
            processed_frames = self.transform(sampled_frames)
            
            return text_caption, processed_frames

        except Exception as e:
            print(f"Error loading or processing video at index {idx} ({full_video_path}): {e}")
            # On error, return the data from the first sample as a fallback
            return self.__getitem__(0)