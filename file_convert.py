import os
import argparse
import soundfile as sf # type: ignore

def convert_audio(input_file: str, output_file: str, output_format: str) -> None:
    """
    Convert an audio file to a different format using soundfile.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to the output audio file.
        output_format (str): Desired output format (e.g., 'wav', 'mp3', 'flac').
    """
    # Ensure the output format is valid
    valid_formats = ['wav', 'mp3', 'flac']
    if output_format not in valid_formats:
        raise ValueError(f"Invalid output format '{output_format}'. Valid formats are: {valid_formats}")
    print(f"Converting {input_file} to {output_format} format...")

    # Load the audio file
    audio, sr = sf.read(input_file)

    # Write the audio file in the desired format
    sf.write(output_file, audio, sr, format=output_format)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Convert audio files to different formats.")
    argument_parser.add_argument("-i", "--input", type=str, help="Path to the input audio file.")
    argument_parser.add_argument("-o", "--output", type=str, help="Path to the output audio file (without extension).")
    argument_parser.add_argument("-f", "--format", type=str, help="Desired output format (e.g., 'wav', 'mp3', 'flac').")
    args = argument_parser.parse_args()
    
    if os.path.isdir(args.input) and os.path.isdir(args.output):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac')):
                    input_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(input_file_path, args.input)
                    output_file_path = os.path.join(args.output, os.path.splitext(relative_path)[0] + f".{args.format}")
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    convert_audio(input_file_path, output_file_path, args.format)
    elif os.path.isfile(args.input) and os.path.isdir(args.output):
        input_file_name = os.path.basename(args.input)
        output_file_path = os.path.join(args.output, os.path.splitext(input_file_name)[0] + f".{args.format}")
        convert_audio(args.input, output_file_path, args.format)
    elif os.path.isfile(args.input) and os.path.isfile(args.output):
        convert_audio(args.input, args.output, args.format)
    else:
        raise ValueError("Invalid input/output paths. Please provide valid file or directory paths.")