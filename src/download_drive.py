import requests
import re
import os

def download_file(file_id, output_path):
    print(f"Downloading {file_id} to {output_path}...")
    session = requests.Session()
    # First get the view page to get cookies
    view_url = f"https://drive.google.com/file/d/{file_id}/view"
    res = session.get(view_url)
    
    # Check for direct download link in HTML or JS
    # Typically window['_DRIVE_DOWNLOAD_URL'] or similar
    download_urls = re.findall(r'https://[^\s"\']+(?:download|export)[^\s"\']+', res.text)
    print("Found download URLs:", len(download_urls))
    
    # Try the standard download URL with session
    dl_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0"
    r = session.get(dl_url, stream=True)
    
    if r.status_code == 200 and 'text/html' not in r.headers.get('content-type', ''):
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        print(f"Successfully saved {output_path} ({os.path.getsize(output_path)} bytes)")
        return True
    
    # If confirmation is required for large files:
    confirm = None
    for k, v in r.cookies.items():
        if k.startswith('download_warning'):
            confirm = v
            break
    if not confirm:
        m = re.search(r'confirm=([0-9A-Za-z_]+)', r.text)
        if m:
            confirm = m.group(1)
            
    if confirm:
        dl_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm}"
        r = session.get(dl_url, stream=True)
        if r.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully saved {output_path} with confirmation ({os.path.getsize(output_path)} bytes)")
            return True
            
    print(f"Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}")
    print("Preview:\n", r.text[:300])
    return False

if __name__ == "__main__":
    download_file('183VP8jxKacXYDT3y2uAwFh1DhXi9BFX5D5T1-d9QaEk', 'data/raw/POWER_Point_Daily.csv')
    download_file('1n_K9I3CqlFuXEw9-3s6eLTCIfL8hs5eDkS4ZTMhSo3I', 'data/raw/El_nino_dataset.csv')
