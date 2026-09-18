import os
import sys
import argparse
import subprocess
import zipfile

CATEGORIES = {
    "greetings": ["Greetings_1of2.zip", "Greetings_2of2.zip"],
    "seasons": ["Seasons_1of1.zip"],
    "colours": ["Colours_1of2.zip", "Colours_2of2.zip"],
    "pronouns": ["Pronouns_1of2.zip", "Pronouns_2of2.zip"],
    "days_and_time": ["Days_and_Time_1of3.zip", "Days_and_Time_2of3.zip", "Days_and_Time_3of3.zip"],
    "electronics": ["Electronics_1of2.zip", "Electronics_2of2.zip"],
    "animals": ["Animals_1of2.zip", "Animals_2of2.zip"],
    "clothes": ["Clothes_1of2.zip", "Clothes_2of2.zip"],
    "jobs": ["Jobs_1of2.zip", "Jobs_2of2.zip"],
    "places": ["Places_1of4.zip", "Places_2of4.zip", "Places_3of4.zip", "Places_4of4.zip"],
    "home": ["Home_1of4.zip", "Home_2of4.zip", "Home_3of4.zip", "Home_4of4.zip"],
    "people": ["People_1of5.zip", "People_2of5.zip", "People_3of5.zip", "People_4of5.zip", "People_5of5.zip"],
    "society": ["Society_1of3.zip", "Society_2of3.zip", "Society_3of3.zip"],
    "transport": ["Means_of_Transportation_1of2.zip", "Means_of_Transportation_2of2.zip"],
}


def get_download_url(filename):
    return f"https://zenodo.org/records/4010759/files/{filename}?download=1"


def download_file(url, dest_path):
    """Resilient resumable multi-segment transfer with aria2c or curl fallback."""
    dest_dir = os.path.dirname(os.path.abspath(dest_path))
    dest_file = os.path.basename(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    # Prefer aria2c for multi-connection parallel acceleration & auto-resume
    if subprocess.run(["which", "aria2c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        cmd = [
            "aria2c",
            "-x", "8",
            "-s", "8",
            "-k", "1M",
            "--continue=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--retry-wait=2",
            "--max-tries=20",
            "-d", dest_dir,
            "-o", dest_file,
            url
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0 and os.path.exists(dest_path):
            return True

    # Fallback to curl with resume
    cmd = [
        "curl", "-L", "-C", "-",
        "--retry", "10",
        "--retry-delay", "2",
        "--retry-connrefused",
        "-o", dest_path,
        url
    ]
    res = subprocess.run(cmd)
    return res.returncode == 0


def download_and_extract_category(category_name, output_dir="raw_dataset"):
    category_key = category_name.lower()
    if category_key not in CATEGORIES:
        print(f"Error: Unknown category '{category_name}'.")
        print(f"Available categories: {list(CATEGORIES.keys())}")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("downloads", exist_ok=True)

    zip_files = CATEGORIES[category_key]

    print(f"\n=======================================================")
    print(f"Downloading Category: '{category_name}' ({len(zip_files)} zip parts)")
    print(f"=======================================================")

    for zip_name in zip_files:
        download_url = get_download_url(zip_name)
        dest_zip = os.path.join("downloads", zip_name)

        if not os.path.exists(dest_zip) or not zipfile.is_zipfile(dest_zip):
            print(f"\nDownloading: {zip_name} from {download_url}...")
            success = download_file(download_url, dest_zip)
            if not success or not os.path.exists(dest_zip) or not zipfile.is_zipfile(dest_zip):
                print(f"Warning: Partial/corrupted download for {zip_name}. Retrying fresh download...")
                if os.path.exists(dest_zip):
                    try:
                        os.remove(dest_zip)
                    except Exception:
                        pass
                cmd = ["curl", "-L", "--retry", "5", "-o", dest_zip, download_url]
                subprocess.run(cmd)

        if not os.path.exists(dest_zip) or not zipfile.is_zipfile(dest_zip):
            print(f"Error: Failed to obtain valid zip archive {zip_name}")
            return

        print(f"Extracting {dest_zip} to '{output_dir}'...")
        with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    print(f"\nCategory '{category_name}' extracted successfully into '{output_dir}'!")


def main():
    parser = argparse.ArgumentParser(description="Download official ISL (INCLUDE) dataset categories from Zenodo")
    parser.add_argument("--category", type=str, default="list",
                        help=f"Category name to download, or 'list' to show options. Choices: {list(CATEGORIES.keys())}")
    parser.add_argument("--output_dir", type=str, default="raw_dataset", help="Directory where video folders will be extracted")
    args = parser.parse_args()

    if args.category == "list":
        print("\nAvailable ISL (INCLUDE) Dataset Categories on Zenodo:")
        for cat, zips in CATEGORIES.items():
            print(f"  - {cat:<20} ({len(zips)} parts: {', '.join(zips)})")
        print("\nExample command to download:")
        print("  python download_dataset.py --category seasons")
        print("  python download_dataset.py --category colours")
        return

    download_and_extract_category(args.category, args.output_dir)


if __name__ == "__main__":
    main()
