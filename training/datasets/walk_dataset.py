import os
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from tqdm import tqdm

apple_loop_dir = Path('/Library/Audio/Apple Loops')

app_root = Path('./app.py').resolve().parent


@dataclass
class MidiInfo:
    tracks: int = 0
    ticks_per_beat: int = 0
    tempo: int = 120
    time_signature: Optional[str] = None
    key_signature: Optional[str] = None
    notes: int = 0
    duration: float = 0.0
    programs: list[int] = field(default_factory=list)


@dataclass
class AppleLoopInfo:
    midi: MidiInfo = field(default_factory=MidiInfo)


def _read_variable_length(data: bytes, position: int, end: int) -> tuple[int, int]:
    value = 0
    while position < end:
        byte = data[position]
        position += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, position
    return value, position


def _parse_midi_data(data: bytes, info: AppleLoopInfo) -> None:
    """Parse an embedded MIDI file and update ``info`` with its properties."""
    if len(data) < 14 or data[:4] != b'MThd':
        return
    # print(f"Processing MIDI data: {len(data)} bytes")

    header_length = struct.unpack('>I', data[4:8])[0]
    if header_length < 6 or 8 + header_length > len(data):
        return

    info.midi.tracks = struct.unpack('>H', data[10:12])[0]
    division = struct.unpack('>H', data[12:14])[0]
    if not division & 0x8000:
        info.midi.ticks_per_beat = division

    position = 8 + header_length
    total_ticks = 0
    tempo = 500000
    programs: set[int] = set()

    for _ in range(info.midi.tracks):
        if position + 8 > len(data) or data[position:position + 4] != b'MTrk':
            break

        track_length = struct.unpack('>I', data[position + 4:position + 8])[0]
        track_end = min(position + 8 + track_length, len(data))
        track_position = position + 8
        track_ticks = 0
        running_status: Optional[int] = None

        while track_position < track_end:
            delta, track_position = _read_variable_length(
                data, track_position, track_end
            )
            track_ticks += delta
            if track_position >= track_end:
                break

            status = data[track_position]
            if status & 0x80:
                track_position += 1
                if status < 0xF0:
                    running_status = status
            elif running_status is not None:
                status = running_status
            else:
                break

            if status == 0xFF:
                if track_position + 2 > track_end:
                    break
                meta_type = data[track_position]
                meta_length = data[track_position + 1]
                track_position += 2
                meta_end = track_position + meta_length
                if meta_end > track_end:
                    break

                if meta_type == 0x51 and meta_length == 3:
                    tempo = int.from_bytes(data[track_position:meta_end], 'big')
                    if tempo:
                        info.midi.tempo = round(60000000 / tempo)
                elif meta_type == 0x58 and meta_length >= 2:
                    info.midi.time_signature = (
                        f"{data[track_position]}/"
                        f"{2 ** data[track_position + 1]}"
                    )
                elif meta_type == 0x59 and meta_length == 2:
                    sf = data[track_position]
                    if sf > 127:
                        sf -= 256
                    key = {
                        (-7, 0): 'Cb major', (0, 0): 'C major',
                        (1, 0): 'G major', (2, 0): 'D major',
                        (3, 0): 'A major', (4, 0): 'E major',
                        (5, 0): 'B major', (6, 0): 'F# major',
                        (7, 0): 'C# major', (-7, 1): 'Ab minor',
                        (0, 1): 'A minor', (1, 1): 'E minor',
                        (2, 1): 'B minor', (3, 1): 'F# minor',
                        (4, 1): 'C# minor', (5, 1): 'G# minor',
                        (6, 1): 'D# minor', (-1, 1): 'D minor',
                    }.get((sf, data[track_position + 1]))
                    if key:
                        info.midi.key_signature = key
                track_position = meta_end
            elif status in (0xF0, 0xF7):
                sysex_length, track_position = _read_variable_length(
                    data, track_position, track_end
                )
                track_position = min(track_position + sysex_length, track_end)
            else:
                event_type = status & 0xF0
                event_length = 1 if event_type in (0xC0, 0xD0) else 2
                if track_position + event_length > track_end:
                    break
                if event_type == 0x90 and data[track_position + 1] != 0:
                    info.midi.notes += 1
                if event_type == 0xC0:
                    programs.add(data[track_position])
                track_position += event_length

        total_ticks = max(total_ticks, track_ticks)
        position = position + 8 + track_length

    if info.midi.ticks_per_beat and tempo:
        info.midi.duration = (
            total_ticks * (tempo / 1e6) / info.midi.ticks_per_beat
        )
    info.midi.programs = sorted(programs)


def _extract_midi(data: bytes) -> Optional[bytes]:
    """Return the complete embedded MIDI payload, if one is present."""
    start = data.find(b'MThd')
    if start < 0 or start + 14 > len(data):
        return None

    header_length = struct.unpack('>I', data[start + 4:start + 8])[0]
    tracks = struct.unpack('>H', data[start + 10:start + 12])[0]
    position = start + 8 + header_length
    for _ in range(tracks):
        if position + 8 > len(data) or data[position:position + 4] != b'MTrk':
            return None
        track_length = struct.unpack('>I', data[position + 4:position + 8])[0]
        position += 8 + track_length
        if position > len(data):
            return None
    return data[start:position]


def parse_caf_metadata(input_file_path, output_file_dir):
    """Parse an Apple loop and write its embedded MIDI file.

    Args:
        input_file_path (str | Path): Path to a CAF or AIFF file.
        output_file_dir (str | Path): Directory in which to write the MIDI file.

    Returns:
        tuple[int, str | None, str | None, str | None]:
            Metadata-found flag, genre, instrument, and MIDI output path.
    """
    input_path = Path(input_file_path)
    data = input_path.read_bytes()
    if data[:4] == b'FORM' and data[8:12] not in (b'AIFF', b'AIFC'):
        # print(f"Not a valid CAF or AIFF file: {input_path}")
        return None, None, None, None, None
    if data[:4] != b'caff' and data[:4] != b'FORM':
        # print(f"Not a valid CAF or AIFF file: {input_path}")
        return None, None, None, None, None

    strings = ''.join(chr(byte) if 32 <= byte < 127 else '.' for byte in data)
    genre = None
    drums = None
    if 'category.' in strings and 'genre.' in strings and 'drum' in strings.lower():
        genre = strings.split('genre.', 1)[1].split('\x00', 1)[0].split('.', 1)[0]
        drums = strings.split('category.', 1)[1].split('\x00', 1)[0].split('.', 1)[0]

    if genre is not None and drums is not None:
        midi_data = _extract_midi(data)
        midi_output_path = None
        if midi_data is not None:
            output_dir = Path(output_file_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            midi_output = output_dir / f'{input_path.stem}.mid'
            midi_output.write_bytes(midi_data)
            midi_output_path = str(midi_output)
            midi_info = AppleLoopInfo()
            _parse_midi_data(midi_data, midi_info)
            # print(
            #     f"Extracted MIDI from {input_path} to {midi_output} "
            #     f"({midi_info.midi.notes} notes, {midi_info.midi.tempo} BPM)"
            # )
            # print(f"Found drum loop metadata for file: {input_path} | Genre: {genre} | Drums: {drums} | MIDI output: {midi_output_path}")
            return genre, drums, midi_info.midi.tempo, midi_info.midi.time_signature, midi_output_path
        # print(f"Found drum loop metadata for file: {input_path} | Genre: {genre} | Drums: {drums}, but no MIDI data found.")
        return genre, drums, None, None, None
    return None, None, None, None, None

# Walk through the directory and rebuild the same folder structure in the app directory
# Then run aif_to_mid.v1.py on the files from the Apple Loops directory to the corresponding directory in the app
metadata = {
    'list_of_genres': [],
    'files': []
}  

for root, dirs, files in os.walk(apple_loop_dir):
    for dir in tqdm(dirs, desc=f"Processing directories in {root}", unit="dir"):
        # Create the same directory structure in the app directory
        old_dir = Path(root) / dir
        new_dir = app_root / Path(root).relative_to(apple_loop_dir) / dir
        new_dir.mkdir(parents=True, exist_ok=True)
        
        for file in os.listdir(old_dir):
            if file.endswith('.aif') or file.endswith('.aiff') or file.endswith('.caf'):
                old_file_path = old_dir / file
                # For example: subprocess.run(['python', 'aif_to_mid.v1.py', str(old_file_path), str(new_file_path)])
                genre, drums, tempo, time_signature, midi_path = parse_caf_metadata(old_file_path, new_dir)
                if midi_path is not None and genre is not None and drums is not None:
                    if genre == '':
                        genre = 'Other Genre'
                    if 'drums' in drums.lower():
                        metadata['files'].append({
                            "midi_path": str(midi_path),
                            'genre': genre, 
                            'drums': drums, 
                            'tempo': tempo, 
                            'time_signature': time_signature 
                        })
                        if genre not in metadata['list_of_genres']:
                            metadata['list_of_genres'].append(genre)
                    else:
                        # print(f"Non-drum entry found for {old_file_path}. Skipping metadata entry.")
                        pass
                else:
                    # print(f"No MIDI data found for {old_file_path}. Skipping metadata entry.")
                    pass
# write metadata.json to the app directory
print("=" * 50)
print(len(metadata['files']), "unique drum loop metadata entries found.")
# Print number of duplicates found in the metadata['files'] list
seen = set()
duplicates = 0
for entry in metadata['files']:
    entry_tuple = (entry['midi_path'])
    if entry_tuple in seen:
        duplicates += 1
    else:
        seen.add(entry_tuple)
print(duplicates, "duplicate entries found in metadata.")
metadata_file_path = app_root / 'metadata.json'
with open(metadata_file_path, 'w') as f:
    json.dump(metadata, f, indent=4)
    print(f"Metadata written to {metadata_file_path}")