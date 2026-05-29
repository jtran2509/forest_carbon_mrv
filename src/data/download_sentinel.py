import pystac_client
import planetary_computer
import stackstac
import numpy as np

def fetch_sentinel_data(bbox, timeframe="2023-01-01/2023-12-31", max_cloud_cover=10):
    # Connect with Microsoft Planetary Computer
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )
    # Search for Sentinel-2 L2A 
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        bbox=bbox,
        datetime=timeframe,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}} #Only takes images with cloud composition < threshold
    )

    items = search.item_collection()
    print(f"Found {len(items)} images satisfied!")
    if len(items) == 0:
        print("No Sentinel-2 items matched the search criteria.")
        return None
    # Pick the first image found (usually has best quality)
    # Selecting important bands: RED (B04), Green (B03), Blue (B02) and near-infrared (B08)-critical for detecting vegetation (forests)
    bands = ["B04", "B03", "B02", "B08"]

    # Stackstac creates a lazy loaded dask array
    data_stack = stackstac.stack(
        items[0],
        assets=bands,
        bounds_latlon=bbox,
        epsg=4326 # Standard WGS84 coordinate system
    )

    # Removes cloud by taking the median pixel value over time 
    composite = data_stack.median(dim="time").compute()

    # Normalization- Sentinel-2 DN are 10,000 * Reflectance => scale them to 0.0 -> 1.0 for the Unet
    composite = composite / 10000.0

    # Clip values to [0, 1] to remove outliers/sensor noise
    composite = np.clip(composite, 0, 1)

    # Load data into the memory (usually pipe straight to S3)
    # final_data = data_stack.compute()
    # return final_data
    return composite

#format: [min_longitude, min_latitude, max_longitude, max_latitude]
# Get coordinates for areas of interests
amazon_aoi = [-62.5, -3.5, -62.4, -3.4] # Amazon (Manaus, Brazil)
vietnam_aoi = [107.5, 12.5, 107.8, 12.8] # Se Asia - Central highlands, Vietnam - high forest density
africa_aoi = [11.4, -0.1, 11.7, 0.2] #Central Africa (Congo Basin, Gabon)
if __name__ == "__main__":
    raster_data = fetch_sentinel_data(amazon_aoi)
    if raster_data is not None:
        print(f"Data successfully fetched! Shape: {raster_data.shape}")