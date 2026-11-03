import torch
import numpy as np
import pretty_midi # type: ignore

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