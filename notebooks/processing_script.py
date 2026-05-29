# processing_script.py
import argparse
import os
import boto3
import numpy as np
import rasterio
import io
import time
import gc
import json 

def read_s3_tif(key, raw_bucket, s3_client):
    """Read GeoTIFF files from S3"""
    try:
        response = s3_client.get_object(Bucket=raw_bucket, Key=key)
        with rasterio.io.MemoryFile(response['Body'].read()) as memfile:
            with memfile.open() as src:
                return src.read(1).astype(np.float32)
    except Exception as e:
        print(f"   ❌ Error reading {key}: {e}")
        raise e

def process_one_image(region, date, raw_bucket, proc_bucket, s3_client, 
                      patch_size=256, forest_threshold=0.01):
    """Handle 1 image at a time for memory efficient"""
    prefix = f"raw/region={region}/date={date}/"
    print(f"  Processing {region} {date}")
    
    # Read 4 bands
    blue = read_s3_tif(f"{prefix}B02.tif", raw_bucket, s3_client)
    green = read_s3_tif(f"{prefix}B03.tif", raw_bucket, s3_client)
    red = read_s3_tif(f"{prefix}B04.tif", raw_bucket, s3_client)
    nir = read_s3_tif(f"{prefix}B08.tif", raw_bucket, s3_client)
    raw_mask = read_s3_tif(f"raw/region={region}/masks/forest_mask.tif", raw_bucket, s3_client)
    
    # Stack & normalize
    img = np.stack([blue, green, red, nir], axis=0).astype(np.float32) / 10000.0
    img = np.clip(img, 0, 1)
    
    # Free each band separately
    del blue, green, red, nir
    gc.collect()
    
    # Mask binarization
    binary_mask = (raw_mask == 10).astype(np.float32)
    del raw_mask
    gc.collect()
    
    # Tiling
    _, h, w = img.shape
    n_h = h // patch_size
    n_w = w // patch_size
    
    count = 0
    skipped = 0
    proc_bucket_new = "forest-carbon-dung-processed-extended"
    
    for i in range(n_h):
        for j in range(n_w):
            y, x = i * patch_size, j * patch_size
            img_chip = img[:, y:y+patch_size, x:x+patch_size].copy()
            mask_chip = binary_mask[y:y+patch_size, x:x+patch_size].copy()
            
            forest_density = np.mean(mask_chip)
            if forest_density < forest_threshold:
                skipped += 1
                del img_chip, mask_chip
                continue
            
            # Upload chip
            img_buffer = io.BytesIO()
            np.save(img_buffer, img_chip)
            img_buffer.seek(0)
            img_key = f"processed/region={region}/date={date}/images/chip_{count}.npy"
            s3_client.upload_fileobj(img_buffer, proc_bucket_new, img_key)
            
            mask_buffer = io.BytesIO()
            np.save(mask_buffer, mask_chip)
            mask_buffer.seek(0)
            mask_key = f"processed/region={region}/date={date}/masks/mask_{count}.npy"
            s3_client.upload_fileobj(mask_buffer, proc_bucket_new, mask_key)
            
            del img_chip, mask_chip
            count += 1
    
    print(f"  ✅ {region} {date}: {count} chips, {skipped} skipped")
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--config-bucket', type=str, required=True)  # Bucket including the file config
    parser.add_argument('--raw-bucket', type=str, default='forest-carbon-dung-raw-extended')
    parser.add_argument('--proc-bucket', type=str, default='forest-carbon-dung-processed-4bands')
    args = parser.parse_args()
    
    # Read dates from S3 (instead of arguments)
    s3_client=boto3.client('s3')
    config_key = f"config/{args.region}_dates.json"

    response = s3_client.get_object(Bucket=args.config_bucket, Key=config_key)
    dates=json.loads(response['Body'].read().decode('utf-8'))

    print(f'Processing {len(dates)} images for {args.region}')
    
    for date in dates:
        try:
            process_one_image(args.region, date, args.raw_bucket, args.proc_bucket, s3_client)
        except Exception as e:
            print(f"  ❌ Failed {date}: {e}")
        gc.collect()
        time.sleep(1)
    
    print(f"✅ Done processing {args.region}")

if __name__ == "__main__":
    main()