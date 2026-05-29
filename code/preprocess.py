import subprocess
import sys

# Cài đặt thư viện ngay khi script bắt đầu (chạy trên container)
subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio", "numpy", "patchify", "boto3", "monai", "torch"])

# Sau khi cài xong mới import
import boto3
import numpy as np
import rasterio
import io
import os

def create_and_upload_chips(region, date, raw_bucket, proc_bucket, patch_size=256, forest_threshold=0.01):
    s3_client = boto3.client('s3')
    
    def read_s3_tif(key):
        response = s3_client.get_object(Bucket=raw_bucket, Key=key)
        with rasterio.open(io.BytesIO(response['Body'].read())) as src:
            return src.read(1)
    
    prefix = f"raw/region={region}/date={date}/"
    b04_key = f"{prefix}B04.tif"
    b08_key = f"{prefix}B08.tif"
    mask_key = f"raw/region={region}/masks/forest_mask.tif"
    
    try:
        red = read_s3_tif(b04_key)
        nir = read_s3_tif(b08_key)
        raw_mask = read_s3_tif(mask_key)
        
        img = np.stack([red, nir], axis=0).astype(np.float32) / 10000.0
        img = np.clip(img, 0, 1)
        binary_mask = (raw_mask == 10).astype(np.float32)
        
        channels, h, w = img.shape
        n_h = h // patch_size
        n_w = w // patch_size
        
        count = 0
        skipped = 0
        
        for i in range(n_h):
            for j in range(n_w):
                y, x = i * patch_size, j * patch_size
                img_chip = img[:, y:y+patch_size, x:x+patch_size]
                mask_chip = binary_mask[y:y+patch_size, x:x+patch_size]
                
                forest_density = np.mean(mask_chip)
                if forest_density < forest_threshold:
                    skipped += 1
                    continue
                
                img_buffer = io.BytesIO()
                np.save(img_buffer, img_chip)
                img_buffer.seek(0)
                img_key = f"processed/region={region}/images/chip_{count}.npy"
                s3_client.upload_fileobj(img_buffer, proc_bucket, img_key)
                
                mask_buffer = io.BytesIO()
                np.save(mask_buffer, mask_chip)
                mask_buffer.seek(0)
                mask_key_out = f"processed/region={region}/masks/mask_{count}.npy"
                s3_client.upload_fileobj(mask_buffer, proc_bucket, mask_key_out)
                count += 1
        
        print(f"Region {region} {date}: Created {count} chips, Skipped {skipped}")
        return count
    except Exception as e:
        print(f"Error {region} {date}: {e}")
        return 0

def main():
    raw_bucket = os.environ.get('RAW_BUCKET', 'forest-carbon-dung-raw')
    proc_bucket = os.environ.get('PROC_BUCKET', 'forest-carbon-dung-processed-v2')
    
    regions_dates = {
        'amazon': ['2023-09-14', '2023-07-26', '2023-07-11', '2023-08-25',
                   '2023-09-04', '2023-07-29', '2023-07-21', '2023-09-09',
                   '2023-08-08', '2023-07-14', '2023-08-30', '2023-07-16',
                   '2023-09-19', '2023-08-03', '2023-07-06'],
        'vietnam': ['2024-02-16', '2024-02-06', '2024-02-26', '2024-01-22',
                    '2024-02-01', '2024-01-27', '2024-01-17', '2024-02-11', '2024-01-12'],
        'central_africa': ['2023-06-22']
    }
    
    total_chips = 0
    for region, dates in regions_dates.items():
        for date in dates:
            chips = create_and_upload_chips(region, date, raw_bucket, proc_bucket)
            total_chips += chips
            print(f"Total chips so far: {total_chips}")
    
    print(f"\n✅ TOTAL CHIPS CREATED: {total_chips}")

if __name__ == "__main__":
    main()
