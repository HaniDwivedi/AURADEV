import argparse
from pathlib import Path


def split_file(file_path, chunk_size, output_dir=None):
    """Split a file into binary-safe chunks and return their paths."""
    source = Path(file_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    destination_dir = Path(output_dir) if output_dir else source.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    chunk_count = (source.stat().st_size + chunk_size - 1) // chunk_size
    chunk_paths = [
        destination_dir
        / f"{source.stem}.part{chunk_number:04d}{source.suffix}"
        for chunk_number in range(1, chunk_count + 1)
    ]
    existing_parts = [path for path in chunk_paths if path.exists()]
    if existing_parts:
        raise FileExistsError(f"Output part already exists: {existing_parts[0]}")

    with source.open("rb") as input_file:
        for chunk_path in chunk_paths:
            bytes_remaining = chunk_size
            with chunk_path.open("xb") as output_file:
                while bytes_remaining:
                    chunk = input_file.read(min(bytes_remaining, 1024 * 1024))
                    if not chunk:
                        break
                    output_file.write(chunk)
                    bytes_remaining -= len(chunk)

    return chunk_paths


def main():
    parser = argparse.ArgumentParser(
        description="Split a file into smaller binary-safe parts."
    )
    parser.add_argument("file", help="Path to the file to split")
    parser.add_argument(
        "--chunk-size-mb",
        type=float,
        default=150,
        help="Maximum size of each part in MiB (default: 150)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for the parts (default: alongside the input file)",
    )
    args = parser.parse_args()

    if args.chunk_size_mb <= 0:
        parser.error("--chunk-size-mb must be greater than zero")

    try:
        parts = split_file(
            args.file,
            int(args.chunk_size_mb * 1024 * 1024),
            args.output_dir,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    for part in parts:
        print(f"Created: {part}")
    print(f"Split complete: {len(parts)} part(s)")


if __name__ == "__main__":
    main()
