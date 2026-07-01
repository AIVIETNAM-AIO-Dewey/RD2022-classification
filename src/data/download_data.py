import os
import zipfile
import gdown
from pathlib import Path

def download_and_extract():
    # Google Drive File ID from the link:
    # https://drive.google.com/file/d/1oMDFwlGwObPg0kmyFVGVoYEMtc8ubOWT/view
    file_id = "1oMDFwlGwObPg0kmyFVGVoYEMtc8ubOWT"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Target directories
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / "rdd2022_india.zip"
    
    # Download
    print(f"Downloading dataset from Google Drive (ID: {file_id})...")
    try:
        gdown.download(url, str(zip_path), quiet=False)
    except Exception as e:
        print(f"Error downloading via gdown: {e}")
        print("Please check if gdown is installed and the Google Drive link is accessible.")
        return

    # Extract
    if zip_path.exists():
        print("Extracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            print("Extraction completed successfully!")
            
            # Clean up the zip file after extraction
            os.remove(zip_path)
            print("Cleaned up the zip file.")
        except Exception as e:
            print(f"Error during extraction: {e}")
    else:
        print("Download failed, zip file not found.")

if __name__ == "__main__":
    download_and_extract()
