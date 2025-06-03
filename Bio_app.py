import streamlit as st
import ee
import geemap.foliumap as geemap
import geopandas as gpd
import folium
from shapely.geometry import Point
from streamlit_folium import st_folium
import json
import pandas as pd
import joblib

# -----------------------------------------------
# 1. Initialize Earth Engine with service account
# -----------------------------------------------
# Convert secrets into credentials
service_account_info = st.secrets["GEE"]
credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"], 
    key_data=json.loads(st.secrets["GEE"].to_json())
)

ee.Initialize(credentials)
st.success("✅ Google Earth Engine authenticated!")


st.title("🌿 AGBD Prediction with GEE and ML")

# -----------------------------------------------
# 2. Define Area of Interest (AOI)
# -----------------------------------------------
st.subheader("1️⃣ Define Your Area of Interest (AOI)")
option = st.radio("Choose AOI method:", ("📂 Upload a file", "📍 Enter coordinates manually"))

m = folium.Map(location=[7.54, -5.55], zoom_start=6)
geometry = None

def gdf_to_ee_featurecollection(gdf):
    # Convert GeoDataFrame geometry to GeoJSON-like format
    geojson = gdf.__geo_interface__
    features = geojson['features']
    
    # Convert each feature to ee.Feature
    ee_features = []
    for f in features:
        geom = ee.Geometry(f["geometry"])
        ee_feat = ee.Feature(geom)
        ee_features.append(ee_feat)
    
    # Return as a FeatureCollection
    return ee.FeatureCollection(ee_features)


if option == "📂 Upload a file":
    uploaded_file = st.file_uploader("Upload a .geojson or zipped shapefile (.zip)", type=["geojson", "zip"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".zip"):
                gdf = gpd.read_file(f"/vsizip/{uploaded_file.name}")
            else:
                gdf = gpd.read_file(uploaded_file)
            geometry = gdf.geometry
            folium.GeoJson(gdf).add_to(m)
            st.success("✅ File loaded successfully.")

            # Convert to ee.FeatureCollection
            aoi_fc = gdf_to_ee_featurecollection(gdf)

        except Exception as e:
            st.error(f"Error reading file: {e}")

elif option == "📍 Enter coordinates manually":
    lat = st.number_input("Latitude", value="min", placeholder="Enter the latitude...", format="%.6f")
    lon = st.number_input("Longitude", value="min", placeholder="Enter the longitude...", format="%.6f")
    buffer_radius = st.slider("Buffer size (in meters)", 100, 10000, 100)
    point = Point(lon, lat)
    gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    gdf_utm = gdf.to_crs(epsg=32630)
    gdf_buffer = gdf_utm.buffer(buffer_radius).to_crs(epsg=4326)
    geometry = gdf_buffer
    folium.GeoJson(gdf_buffer).add_to(m)

    # Convert to ee.FeatureCollection
    aoi_fc = gdf_to_ee_featurecollection(gdf_buffer)

st_folium(m, width=700, height=500)

def extract_coordinates(geom):
    if geom.geom_type == "Polygon":
        return [[lon, lat] for lon, lat in geom.exterior.coords]

    elif geom.geom_type == "MultiPolygon":
        # Flatten the outer rings of all polygons into a single list
        coords = []
        for polygon in geom.geoms:
            coords += [[lon, lat] for lon, lat in polygon.exterior.coords]
        return coords

    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

# -----------------------------------------------
# 3. Convert AOI to Earth Engine Geometry
# -----------------------------------------------
pred_data_fc = None
if geometry is not None:
    #aoi_geom = geometry.values[0] if hasattr(geometry, 'values') else geometry[0]
    #aoi_json = json.loads(gpd.GeoSeries([aoi_geom]).to_json())["features"][0]["geometry"]
    #pred_data_fc = ee.FeatureCollection([ee.Feature(ee.Geometry(aoi_json))])

    # Extract coordinates
    coordinates = extract_coordinates(geometry)
    lat_min = min([c[1] for c in coordinates])
    lat_max = max([c[1] for c in coordinates])
    lon_min = min([c[0] for c in coordinates])
    lon_max = max([c[0] for c in coordinates])

    # Selection of the points inside the AOI
    cordinate = []
    polygon = Polygon(coordinates)
    for lat in lats:
        for lon in lons:
            p = [lon, lat]
            point = Point(p)
            if polygon.contains(point):
                cordinate.append(p)



    #ids = []
    lats = []
    lons = []
    #bios =[]
    k =0
    for i,c in enumerate(tqdm(ivoire_cordinate,total=len(ivoire_cordinate))):
        lat = c[1]
        lon = c[0]
        #point = Point([lon,lat])
        #if polygon.contains(point):
	#id = "zone_"+str(k)
  	#bio = -1
  	#ids.append(id)
  	lats.append(lat)
  	lons.append(lon)
  	#bios.append(bio)
  	k = k+1
		
    prediction_data = pd.DataFrame({"identifiant":ids,
                      "Latitude":lats,
                      "Longitude":lons,
                      "Biomass":bios})

    pred_data = prediction_data[['Latitude', 'Longitude',	'Biomass']]

    geometry = [Point(x,y) for x,y in zip(pred_data['Longitude'], pred_data['Latitude'])]
    gdf = gpd.GeoDataFrame(pred_data,  crs = 'EPSG:4326', geometry = geometry)
    gdf = gdf.to_crs('EPSG:32630')
    gdf['geometry'] = gdf['geometry'].buffer(30)
    gdf.to_crs('EPSG:4326',inplace=True)

    features = []

    for _, row in gdf.iterrows():
        geojson = row['geometry'].__geo_interface__
    	geom = ee.Geometry(geojson)
    	props = row.drop('geometry').to_dict()
    	feature = ee.Feature(geom, props)
    	features.append(feature)

    pred_data_fc = ee.FeatureCollection(features)

    st.success("🌍 AOI is ready for processing.")

if pred_data_fc:
    st.subheader("2️⃣ Processing Satellite Data...")

    # Load Sentinel-2 spectral reflectance data.
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    # Create a function to mask clouds using the Sentinel-2 QA band.
    def maskS2clouds(image):
        qa = image.select('QA60')
    	# Bits 10 and 11 are clouds and cirrus, respectively.
    	cloudBitMask = ee.Number(2).pow(10).int()
    	cirrusBitMask = ee.Number(2).pow(11).int()
    	# Both flags should be set to zero, indicating clear conditions.
    	mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
        qa.bitwiseAnd(cirrusBitMask).eq(0))
    	# Return the masked and scaled data.
    	return image.updateMask(mask).divide(10000)

    def maskLowQA(img):
	qaBand = 'cs_cdf'
  	clearThreshold = 0.6
  	mask = img.select(qaBand).gte(clearThreshold)
  	return img.updateMask(mask)

    # Create a single composite image for a given period.
    start_date = '2023-01-01'
    end_date = '2023-12-31'


    def processing_s2_images(start_date, end_date, aoi):
	image_coll = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start_date, end_date).filterBounds(aoi)\
  				.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    				.map(maskS2clouds) \
   				.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12'])

  	proj = ee.Image(image_coll.first()).select('B4').projection()
  	csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
  	csPlusBands = csPlus.first().bandNames()
  	filteredS2WithCs = image_coll.linkCollection(csPlus,csPlusBands)
  	s2Processed = filteredS2WithCs.map(maskLowQA).map(lambda image: image.clip(aoi))
  	composite = s2Processed.median().setDefaultProjection(proj)
  	return composite

    # Compute the median composite and clip to the boundary.
    S2_composite = processing_s2_images(start_date, end_date, aoi_fc)


    # Vegetation Indices calculation

    # NDVI (Normalized Difference Vegetation Index)
    ndvi = S2_composite.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # EVI (Enhanced Vegetation Index)
    evi = S2_composite.expression(
    	'G * ((NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L))', {
        'NIR': S2_composite.select('B8'),
        'RED': S2_composite.select('B4'),
        'BLUE': S2_composite.select('B2'),
        'G': 2.5,  # Gain factor
        'C1': 6,
        'C2': 7.5,
        'L': 1  # Canopy background adjustment
    	}).rename('EVI')

    # NDRE (Normalized Difference Red Edge Index)
    ndre = S2_composite.normalizedDifference(['B8', 'B5']).rename('NDRE')

    # 4. RVI (Ratio Vegetation Index)
    rvi = S2_composite.expression(
    	'NIR / RED', {
        'NIR': S2_composite.select('B8'),
        'RED': S2_composite.select('B4')
    	}).rename('RVI')

    # GNDVI (Green Normalized Difference Vegetation Index)
    gndvi = S2_composite.normalizedDifference(['B8', 'B3']).rename('GNDVI')

    # SAVI (Soil Adjusted Vegetation Index)
	L = 0.5  # Soil adjustment factor, typically between 0 and 1
	savi = S2_composite.expression(
    	'((NIR - Red) / (NIR + Red + L)) * (1 + L)', {
        'NIR': S2_composite.select('B8'),
        'Red': S2_composite.select('B4'),
        'L': L
    	}).rename('SAVI')

    # MSAVI (Modified Soil Adjusted Vegetation Index)
    msavi = S2_composite.expression(
    	'(2 * NIR + 1 - sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - RED))) / 2', {
        'NIR': S2_composite.select('B8'),
        'RED': S2_composite.select('B4')
    	}).rename('MSAVI')

    # WDRVI (Wide Dynamic Range Vegetation Index)
    wdrvi = S2_composite.expression(
    	'a * ((NIR - RED) / (NIR + RED))', {
        'NIR': S2_composite.select('B8'),
        'RED': S2_composite.select('B4'),
        'a': 0.2  # Weighting coefficient for dense vegetation
    	}).rename('WDRVI')

    # CIred-edge (Red Edge Chlorophyll Index)
    ciredge = S2_composite.expression(
    	'(NIR / RedEdge) - 1', {
        'NIR': S2_composite.select('B8'),
        'RedEdge': S2_composite.select('B5')
    	}).rename('CIred_edge')

    # CCCI (Canopy Chlorophyll Content Index)
    ccci = S2_composite.expression(
    	'NDRE / NDVI', {
        'NDRE': ndre,
        'NDVI': ndvi
    	}).rename('CCCI')

    # RESI (Red Edge Simple Ratio Index)
    resi = S2_composite.expression(
    	'((RE3 + RE2 - RE1) / (RE3 + RE2 + RE1))', {
        'RE1': S2_composite.select('B5'),
        'RE2': S2_composite.select('B6'),
        'RE3': S2_composite.select('B7')
    	}).rename('RESI')


    
    # Load other datasets
    # Import other datasets such as the Shuttle Radar Topography Mission (SRTM) digital elevation model (DEM). 
    # Add the SRTM DEM data and calculate slope.

    # Load SRTM DEM
    srtm = ee.Image("USGS/SRTMGL1_003")

	#srtm_resampled = srtm.resample('bilinear').reproject({
	#  'crs': srtm.projection(),
	#  'scale': 10 })
	#print('Resampled Resolution:', resampled.projection().nominalScale())

    # Clip Elevation to the boundary
    elevation = srtm.clip(tai_forest_boundary)

    # Derive slope from the SRTM
    slope = ee.Terrain.slope(srtm).clip(tai_forest_boundary)


    # Load Sentinel-1 data
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
    	.filterDate(start_date, end_date) \
    	.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    	.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
    	.filter(ee.Filter.eq('instrumentMode', 'IW')) \
    	.filterBounds(tai_forest_boundary)

    # Create a median composite for VV and VH
    vv = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).select('VV').median().clip(tai_forest_boundary)
    vh = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).select('VH').median().clip(tai_forest_boundary)

    # Calculate the ratio of VH to VV
    vv_vh_ratio = vv.divide(vh).rename('VV/VH')



    # Load ALOS PALSAR-2 dataset and filter by date
    palsar2 = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH') \
                .filter(ee.Filter.date(start_date, end_date)) \
                .filterBounds(tai_forest_boundary)

    # Select HH and HV polarizations and calculate median
    sarhh = palsar2.select('HH').median().clip(tai_forest_boundary)
    sarhv = palsar2.select('HV').median().clip(tai_forest_boundary)

    # Calculate the HH/HV ratio
    hh_hv_ratio = sarhh.divide(sarhv).rename('HH_HV_Ratio')

    # Merge the predictor variables
    merged_dataset = (
    	S2_composite
    	.addBands(ndvi)
    	.addBands(evi)
    	.addBands(ndre)
    	.addBands(rvi)
    	.addBands(gndvi)
    	.addBands(savi)
    	.addBands(msavi)
    	.addBands(wdrvi)
    	.addBands(ciredge)
    	.addBands(ccci)
    	.addBands(resi)
    	.addBands(elevation)
    	.addBands(slope)
    	.addBands(vv.rename('VV'))
    	.addBands(vh.rename('VH'))
    	.addBands(sarhh.rename('HH'))
    	.addBands(sarhv.rename('HV'))
    	.addBands(vv_vh_ratio.rename('VV_VH_ratio'))
    	.addBands(hh_hv_ratio.rename('HH_HV_ratio'))
    	)

    # Clip the output image to the farm boundary
    clippedmergedCollection = merged_dataset.clipToCollection(tai_forest_boundary)

    # Bands to include in the regression
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12',
         'NDVI', 'EVI', 'NDRE', 'RVI', 'GNDVI', 'SAVI', 'MSAVI', 'WDRVI', 'CIred_edge', 'CCCI',
         'elevation', 'slope', 'VV', 'VH', 'HH', 'HV', 'VV_VH_ratio', 'HH_HV_ratio']

    #prediction_features = clippedmergedCollection.sampleRegions(
    #    collection=pred_data_fc,
    #    scale=10,
    #    geometries=True
    #)

    # Reduce each polygon in pred_data_fc over the clipped composite
    prediction_features = clippedmergedCollection.select(bands).reduceRegions(**{
    	'collection': ee.FeatureCollection(pred_data_fc),
    	'reducer': ee.Reducer.mean(),
    	'scale': 10,
    	'tileScale': 4  # Optional: improves performance for large regions
	})


    # Convert to client-side dictionary
    features_info = prediction_features.getInfo()

    # Extract features from GeoJSON-like structure
    features_list = features_info['features']

    # Convert to DataFrame
    data = []
    for f in features_list:
        props = f['properties']
    	geom = f['geometry']
    	props['geometry'] = geom  # optional
    	data.append(props)

    df_pred = pd.DataFrame(data)
    # Drop the first ('Biomass') and last ('geometry' columns
    training_set = df_pred.drop(columns=['Biomass', 'geometry'])


    # later reload the pickle file
    sdc_reload = pk.load(open("sdc.pkl","rb"))
    pca_reload = pk.load(open("pca.pkl","rb"))

    # List of band columns to apply PCA on
    band_columns = ['B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']

    # Select the band data
    X = training_set[band_columns]
    X_pca = sdc_reload.transform(X)
    principalComponents = pca_reload.transform(X_pca)

    # Create a DataFrame with the principal components
    pca_columns = [f'PC{i+1}' for i in range(principalComponents.shape[1])]
    pca_df = pd.DataFrame(principalComponents, columns=pca_columns)

    X_index = training_set.drop(columns=band_columns)

    # Add the principal components back to the original DataFrame
    pred_data = pd.concat([pca_df,X_index],axis=1)
    pred_data = pred_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude', 'CIred_edge': 'CIred-edge'})


    st.subheader("3️⃣ Predicting AGBD")

    try:
	loaded_model = joblib.load("agbd_model_bench.joblib")
	bands_used_in_training = loaded_model.feature_names_in_
        df = df[bands_used_in_training]  # ensure columns match
        biomass = loaded_model.predict(pred_data)
	biomass_df = pd.DataFrame(biomass, columns=["Biomass"])
	biomass_df["Biomass"] = biomass_df["Biomass"].astype(int)
	biomass_df['Biomass'] = biomass_df['Biomass'].apply(lambda x: max(x, 0))

	# Compute Carbon
	biomass_df["Carbon"] = (biomass_df["Biomass"] * 0.47).astype(int)
	    
	# Combine with coordinates
	coord = pred_data[['latitude', 'longitude']].reset_index(drop=True)
	predictions = pd.concat([coord, biomass_df], axis=1)
        
        #if 'longitude' in predictions.columns and 'latitude' in predictions.columns:
        #    st.map(predictions[['latitude', 'longitude']])

        # ✅ Download option
        st.download_button("📥 Download predictions", predictions.to_csv(index=False), file_name="AGBD_predictions.csv")

	# ✅ Convert to GeoDataFrame
	geometry = [Point(xy) for xy in zip(predictions['longitude'], predictions['latitude'])]
	geo_df = gpd.GeoDataFrame(predictions, geometry=geometry, crs="EPSG:4326")

	# User selects which to visualize
	selected_metric = st.selectbox("📊 Select value to visualize:", options=["Biomass", "Carbon"])

	# Center the map
	center_lat = geo_df.geometry.y.mean()
	center_lon = geo_df.geometry.x.mean()
	m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

	# Create a color scale
	min_val = geo_df[selected_metric].min()
	max_val = geo_df[selected_metric].max()
	colormap = folium.LinearColormap(
    			colors=["blue", "green", "yellow", "orange", "red"],
    			vmin=min_val, vmax=max_val,
    			caption=f"{selected_metric} (g/m²)"
			)

	# Add points to map
	for _, row in geo_df.iterrows():
	    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        color=colormap(row[selected_metric]),
        		fill=True,
        		fill_opacity=0.8,
        		popup=folium.Popup(
            		f"Biomass: {row['Biomass']} g/m²<br>Carbon: {row['Carbon']} g/m²", max_width=200
        		)
    			).add_to(m)

	# Add colormap to the map
	colormap.add_to(m)

	# Show map in Streamlit
	st_folium(m, width=700, height=500)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


