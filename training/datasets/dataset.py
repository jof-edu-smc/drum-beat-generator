import os
import csv
import json
import torch # type: ignore
import numpy as np # type: ignore
import hashlib
import pretty_midi # type: ignore
from torch.utils.data import Dataset # type: ignore
from tqdm import tqdm # type: ignore 
from pathlib import Path

class DrumDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        genre_filter='all',
        device='mps',
    ):
        
        # Initialize the dataset with metadata and filter by genre if specified.
        self.metadata_path = metadata_path
        self.device = device
        self.genre = genre_filter
        self.available_genres = set()
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            if self.metadata is None:
                raise ValueError(f"Metadata file {self.metadata_path} is empty or not properly formatted.")
        
        self.available_genres = self.metadata.get('list_of_genres', [])
        self.dataset = [] 
        self.invalid_dataset = []  # To keep track of invalid MIDI files
        
        for item in tqdm(self.metadata['files'], desc="Processing MIDI files", unit="file"):
            midi_path = item['midi_path']
            if genre_filter != 'all' and item['genre'] != genre_filter:
                continue
            
            info = {
                'bar': np.zeros((1, 128, 16), dtype=np.float32),  # Placeholder for the tensor
                'genre': item['genre'],
                'drums': item['drums'],
                'tempo': item.get('tempo', 120),
                'time_signature': item.get('time_signature', '4/4'),
            }
            
            if info['time_signature'] is None:
                info['time_signature'] = '4/4' 
            
            if info['tempo'] is None:
                info['tempo'] = 120
                
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                total_duration = midi_data.get_end_time()
                if total_duration < (60.0 / info['tempo']) * 4:  # Less than 1 bar at default tempo
                    raise ValueError(f"Warning: MIDI file {midi_path} is too short for at least one bar. Skipping.")
                
                processed_bars = self.process_midi_to_tensor_list(midi_path, info)
                
                if not processed_bars or len(processed_bars) == 0:
                    raise ValueError(f"No valid bars were processed from MIDI file: {midi_path}. Ensure the file contains drum notes and is long enough for at least one bar.")
                
                for bar in processed_bars:
                    if bar.shape != (1, 128, 16) or type(bar) is not torch.Tensor:
                        raise ValueError(f"Processed bar tensor has unexpected shape: {bar.shape}. Expected shape is (1, 128, 16).")
                    info['bar'] = bar.to(self.device)
                    self.dataset.append(info)
                    
            except Exception as e:
                self.invalid_dataset.append((info, str(e)))
                        
        print(f"Processing & Validation complete. {len(self.dataset)} valid MIDI files, {len(self.invalid_dataset)} invalid MIDI files.")


    def process_midi_to_tensor_list(self, midi_path, info, stride=16):
        """
        Convert non-Groove MIDI files without Groove-specific quantization.
        Parameters:
            - midi_path: Path to the MIDI file to process.
            - stride: Number of time steps to move for each new bar segment (default: 16, which corresponds to 1 bar at 16th note resolution).
        Returns: A list of PyTorch tensors, each with shape (1, 128, 16), representing 1-bar segments of the MIDI file without applying Groove-specific quantization.
        """
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        tempo = info.get('tempo', 120)
        step_duration = (60.0 / tempo) / 4.0

        max_end_time = 0.0
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                max_end_time = max(max_end_time, note.end)
        total_steps = int(np.ceil(max_end_time / step_duration))
        if total_steps < 16:
            return []

        master_matrix = np.zeros((128, total_steps), dtype=np.float32)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                continue
            for note in instrument.notes:
                start_bin = int(np.floor(note.start / step_duration))
                end_bin = int(np.ceil(note.end / step_duration))
                if end_bin <= start_bin:
                    end_bin = start_bin + 1
                start_bin = min(max(0, start_bin), total_steps - 1)
                end_bin = min(max(1, end_bin), total_steps)
                master_matrix[note.pitch, start_bin:end_bin] = 1.0

        tensor_list = []
        for start_idx in range(0, total_steps - 15, stride):
            bar_slice = master_matrix[:, start_idx:start_idx + 16]
            bar_tensor = torch.from_numpy(bar_slice).unsqueeze(0).to(self.device)
            tensor_list.append(bar_tensor)

        return tensor_list

    def save_bars(self, bars, item, target_split):
        """
        Saves the processed bar tensors to disk and updates the train/eval bar paths.

        Parameters:
        - bars: A list of PyTorch tensors, each representing a 1-bar segment of the MIDI file.
        - item: A dictionary containing metadata and file path for the original MIDI file.
        - target_split: A string indicating whether the bars belong to the 'train' or 'eval' split.
        """
        target_dir = self.pt_train if target_split == 'train' else self.pt_eval
        stem = Path(item['path']).stem
        source_tag = item['source']
        genre_tag = item['genre'].replace(' ', '_')

        for index, bar in enumerate(bars):
            bar_path = os.path.join(target_dir, f'{source_tag}_{genre_tag}_{stem}_bar{index}.pt')
            torch.save({
                'bar': bar,
                'bar_path': bar_path,
                'src_midi_file': item['path'],
                'source': item['source'],
                'genre': item['genre'],
                'bpm': item['bpm'],
                'time_signature': item['time_signature'],
            }, bar_path)

            if target_split == 'train':
                self.train_bars.append(bar_path)
            else:
                self.eval_bars.append(bar_path)
    
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def save_gan_tensor_to_drums(tensor_input, output_path="gan_drums.mid", bpm=120):
        """
        Converts GAN output into a Groove-compatible MIDI file of ANY length.
        Parameters:
        - A single 2D tensor: (128, 16)
        - A batched 3D/4D tensor: (N, 1, 128, 16)
        - A Python list of tensors: [tensor1, tensor2, ...]
        Returns: The file path to the saved MIDI file.
        """
        if isinstance(tensor_input, list):
            matrices = [t.detach().squeeze().cpu().numpy() for t in tensor_input]
            matrix = np.concatenate(matrices, axis=1)
        elif torch.is_tensor(tensor_input):
            t = tensor_input.detach().squeeze().cpu()
            if t.ndim == 3:
                matrix = t.permute(1, 0, 2).reshape(128, -1).numpy()
            elif t.ndim == 2:
                matrix = t.numpy()
            else:
                raise ValueError(f"Unexpected tensor shape: {tensor_input.shape}")
        else:
            raise TypeError("Input must be a PyTorch tensor or a list of tensors.")

        num_pitches, total_steps = matrix.shape
        drum_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        drum_track = pretty_midi.Instrument(program=0, is_drum=True, name="GAN Drum Kit")
        step_duration = (60.0 / bpm) / 4.0

        for pitch in range(num_pitches):
            for step in range(total_steps):
                if matrix[pitch, step] > 0.5:
                    start_time = step * step_duration
                    end_time = start_time + step_duration
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    drum_track.notes.append(note)

        drum_midi.instruments.append(drum_track)
        drum_midi.write(output_path)
        return output_path