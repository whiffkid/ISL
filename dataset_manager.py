import os
import sys

# Auto-reexec into local virtual environment if dependencies are not in current Python
try:
    import tensorflow
except ImportError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.realpath(sys.executable) != os.path.realpath(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)

import argparse
import subprocess
import zipfile
import glob
from download_dataset import CATEGORIES, get_download_url, download_file
from keypoint_extraction import save_data, VALID_EXTENSIONS

def print_status(raw_dir="raw_dataset", keypoint_dir="keypoint_data"):
    print("\n=======================================================")
    print("ISL (INCLUDE) Dataset Status & Vocabulary Coverage")
    print("=======================================================")
    print(f"{'Category':<22} | {'Downloaded':<12} | {'Extracted Keypoints'}")
    print("-" * 58)

    for cat in CATEGORIES.keys():
        # Check if raw videos exist
        cat_raw_exists = False
        if os.path.exists(raw_dir):
            for item in os.listdir(raw_dir):
                item_path = os.path.join(raw_dir, item)
                if os.path.isdir(item_path) and (cat in item.lower() or item.lower() in cat):
                    cat_raw_exists = True
                    break

        # Check keypoint classes
        keypoint_count = 0
        if os.path.exists(keypoint_dir):
            keypoint_count = len([d for d in os.listdir(keypoint_dir) if os.path.isdir(os.path.join(keypoint_dir, d)) and not d.startswith('.')])

        downloaded_str = "✓ Yes" if cat_raw_exists else "- No"
        print(f"{cat:<22} | {downloaded_str:<12} | Total dataset classes: {keypoint_count}")

    print("=======================================================\n")


def extract_category_keypoints(raw_dir="raw_dataset", export_dir="keypoint_data", max_frames=30, skip_frame=2):
    """Scan raw dataset directory and extract keypoints for all discovered action folders."""
    if not os.path.exists(raw_dir):
        print(f"Directory '{raw_dir}' does not exist.")
        return

    # Find all subdirectories that contain video files
    action_dirs = []
    for root, dirs, files in os.walk(raw_dir):
        video_files = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]
        if video_files:
            action_name = os.path.basename(root)
            action_dirs.append((action_name, root, video_files))

    if not action_dirs:
        print(f"No video files found inside '{raw_dir}'.")
        return

    print(f"\nFound {len(action_dirs)} action classes with video files.")
    for action_name, action_path, video_files in action_dirs:
        print(f"\nProcessing class '{action_name}' ({len(video_files)} videos)...")
        for video_file in video_files:
            save_data(action_name, video_file, export_dir, os.path.dirname(action_path), max_frames, skip_frame)


def download_and_setup(categories, raw_dir="raw_dataset", export_dir="keypoint_data", auto_extract=True, cleanup_raw=True):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs("downloads", exist_ok=True)

    if "all" in categories:
        target_cats = list(CATEGORIES.keys())
    else:
        target_cats = [c.strip().lower() for c in categories if c.strip().lower() in CATEGORIES]

    if not target_cats:
        print("No valid categories specified.")
        return

    print(f"Target categories to download ({len(target_cats)}): {target_cats}")

    for cat in target_cats:
        zip_files = CATEGORIES[cat]
        print(f"\n>>> Processing Category: {cat}")
        for zip_name in zip_files:
            url = get_download_url(zip_name)
            dest_zip = os.path.join("downloads", zip_name)
            
            if not os.path.exists(dest_zip) or not zipfile.is_zipfile(dest_zip):
                print(f"Downloading {zip_name}...")
                success = download_file(url, dest_zip)
                if not success or not os.path.exists(dest_zip) or not zipfile.is_zipfile(dest_zip):
                    print(f"Warning: Incomplete archive {zip_name}. Retrying fresh download...")
                    if os.path.exists(dest_zip):
                        try:
                            os.remove(dest_zip)
                        except Exception:
                            pass
                    cmd = ["curl", "-L", "--retry", "5", "-o", dest_zip, url]
                    subprocess.run(cmd)
            else:
                print(f"Using cached archive: {dest_zip}")

            if not os.path.exists(dest_zip) or not zipfile.is_zipfile(dest_zip):
                print(f"Failed to obtain valid zip for {zip_name}, skipping.")
                continue

            print(f"Extracting {dest_zip}...")
            with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)

    if auto_extract:
        print("\nStarting automated keypoint extraction...")
        extract_category_keypoints(raw_dir=raw_dir, export_dir=export_dir)

        if cleanup_raw:
            print("\n🧹 Cleaning up raw heavy video files and zip archives to save disk space (<15GB limit)...")
            try:
                import shutil
                # Remove extracted raw video directories
                for item in os.listdir(raw_dir):
                    item_path = os.path.join(raw_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                # Remove downloaded zip files
                if os.path.exists("downloads"):
                    for zf in os.listdir("downloads"):
                        zp = os.path.join("downloads", zf)
                        if os.path.isfile(zp):
                            os.remove(zp)
                print("✓ Raw videos & zip archives cleaned! Keypoint .npy dataset preserved in keypoint_data/.")
            except Exception as e:
                print(f"Cleanup note: {e}")


def main():
    parser = argparse.ArgumentParser(description="Universal ISL Dataset Manager for downloading & processing INCLUDE dataset categories")
    parser.add_argument("--status", action="store_true", help="Show current dataset status and classes")
    parser.add_argument("--download", type=str, help="Category name, comma-separated names, or 'all'")
    parser.add_argument("--extract_only", action="store_true", help="Run keypoint extraction on all videos in raw_dataset")
    parser.add_argument("--no_cleanup", action="store_true", help="Keep heavy raw video files after keypoint extraction")
    parser.add_argument("--raw_dir", type=str, default="raw_dataset", help="Raw dataset video directory")
    parser.add_argument("--export_dir", type=str, default="keypoint_data", help="Keypoint export directory")
    args = parser.parse_args()

    if args.status or (not args.download and not args.extract_only):
        print_status(args.raw_dir, args.export_dir)
        return

    if args.extract_only:
        extract_category_keypoints(raw_dir=args.raw_dir, export_dir=args.export_dir)
        return

    if args.download:
        cats = [c.strip() for c in args.download.split(",")]
        download_and_setup(
            cats,
            raw_dir=args.raw_dir,
            export_dir=args.export_dir,
            cleanup_raw=not args.no_cleanup
        )


if __name__ == "__main__":
    main()
