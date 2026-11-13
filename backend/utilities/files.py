import os
import torch
import numpy as np
import pretty_midi # type: ignore
from google.cloud import storage # type: ignore

def get_weights_from_gcs(bucket_name, blob_name, local_path):
    """
    returns model weights from GCS bucket and saves them to local_path
    Parameters:
    - bucket_name: str, name of the GCS bucket
    - blob_name: str, name of the blob (file) in the GCS bucket
    - local_path: str, local file path where the blob will be saved
    Returns: dictionary of model weights loaded from the local file
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    
    return None
    

def _tensor_input_to_matrix(tensor_input):
        if isinstance(tensor_input, list):
            matrices = [t.detach().squeeze().cpu().numpy() for t in tensor_input]
            return np.concatenate(matrices, axis=1)
        if torch.is_tensor(tensor_input):
            t = tensor_input.detach().squeeze().cpu()
            if t.ndim == 3:
                return t.permute(1, 0, 2).reshape(128, -1).numpy()
            if t.ndim == 2:
                return t.numpy()
            raise ValueError(f"Unexpected tensor shape: {tensor_input.shape}")
        raise TypeError("Input must be a PyTorch tensor or a list of tensors.")


def _save_midi(midi_data, output_path, is_dev):
        if is_dev:
            print(f"Saving MIDI file to: {output_path}")
            midi_data.write(output_path)
        else:
            client = storage.Client()
            bucket_name = os.getenv("GCS_BUCKET_NAME", "some_drum_thingys")
            blob = client.bucket(bucket_name).blob(output_path)
            blob.upload_from_string(midi_data.write(), content_type='audio/midi')


def save_gan_tensor_to_drums(
    tensor_input, 
    output_path="gan_drums.mid", 
    bpm=120, 
    is_dev=False
):
        """
        Converts GAN output into a Groove-compatible MIDI file of ANY length.
        Parameters:
        - A single 2D tensor: (128, 16)
        - A batched 3D/4D tensor: (N, 1, 128, 16)
        - A Python list of tensors: [tensor1, tensor2, ...]
        Returns: The file path to the saved MIDI file.
        """
        matrix = _tensor_input_to_matrix(tensor_input)

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
        _save_midi(drum_midi, output_path, is_dev)
        return output_path


def append_gan_tensor_to_drums(
    base_midi_path,
    tensor_input,
    output_path="gan_drums.mid",
    bpm=120,
    is_dev=False
):
        """Append generated drum bars to an existing MIDI file."""
        drum_midi = pretty_midi.PrettyMIDI(base_midi_path)
        drum_track = next(
            (instrument for instrument in drum_midi.instruments if instrument.is_drum),
            None
        )
        if drum_track is None:
            drum_track = pretty_midi.Instrument(program=0, is_drum=True, name="GAN Drum Kit")
            drum_midi.instruments.append(drum_track)

        matrix = _tensor_input_to_matrix(tensor_input)
        step_duration = (60.0 / bpm) / 4.0
        bar_duration = step_duration * 16
        original_end = max(
            (note.end for instrument in drum_midi.instruments for note in instrument.notes),
            default=0.0
        )
        append_start = np.ceil(original_end / bar_duration) * bar_duration

        for pitch in range(matrix.shape[0]):
            for step in range(matrix.shape[1]):
                if matrix[pitch, step] > 0.5:
                    start_time = append_start + step * step_duration
                    drum_track.notes.append(pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start_time,
                        end=start_time + step_duration
                    ))

        _save_midi(drum_midi, output_path, is_dev)
        return output_path
    
def generate_gcs_url(file_path):
    """
    Generates a public URL for a file stored in Google Cloud Storage.
    Assumes the file is already uploaded to GCS and is publicly accessible.
    """
    client = storage.Client()
    bucket_name = os.getenv("GCS_BUCKET_NAME", "some_drum_thingys")  # Ensure this environment variable is set
    blob_name = file_path.split("/")[-1]  # Assuming the file name is the last part of the path
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(f"The file {blob_name} does not exist in bucket {bucket_name}.")

    return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"

def process_midi_to_tensor_list(midi_path, target_bpm=None, time_signature=None, stride=16, device=torch.device("cpu")):
        """
        Extracts overlapping 1-bar chunks from a MIDI file.
        Parameters:
        - midi_path: Path to the MIDI file to process.
        - target_bpm: The BPM to use for quantization (if None, it will be estimated from the MIDI file).
        - time_signature: The time signature to use for quantization (e.g., '4-4'). If None, it will be estimated from the MIDI file or defaulted to '4-4'.
        - stride: Number of time steps to move for each new bar segment (default: 16, which corresponds to 1 bar at 16th note resolution).
        - device: The PyTorch device to place the resulting tensors on (default: CPU).
        Returns: A list of PyTorch tensors, each with shape (1, 128, 16).
        """
        
        if target_bpm is None or time_signature is None:
            raise ValueError("Both target_bpm and time_signature must be provided together for accurate slicing.")

        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            return f"Error processing {midi_path}: {str(e)}"

        tempo = target_bpm if target_bpm else midi_data.estimate_tempo()
        step_duration = (60.0 / tempo) / float(time_signature[0])
        max_end_time = 0
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                for note in instrument.notes:
                    if note.end > max_end_time:
                        max_end_time = note.end

        total_steps = int(np.ceil(max_end_time / step_duration))
        if total_steps < 16:
            return []

        master_matrix = np.zeros((128, total_steps), dtype=np.float32)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                continue
            for note in instrument.notes:
                start_bin = int(np.round(note.start / step_duration))
                end_bin = int(np.round(note.end / step_duration))
                if start_bin == end_bin:
                    end_bin += 1
                start_bin = min(max(0, start_bin), total_steps - 1)
                end_bin = min(max(1, end_bin), total_steps)
                for t in range(start_bin, end_bin):
                    master_matrix[note.pitch, t] = 1.0

        tensor_list = []
        for start_idx in range(0, total_steps - 15, stride):
            bar_slice = master_matrix[:, start_idx:start_idx + 16]
            bar_tensor = torch.from_numpy(bar_slice).unsqueeze(0).to(device)
            tensor_list.append(bar_tensor)

        return tensor_list