import numpy as np
import pickle
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ee
import math
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Load ML model
# -----------------------------
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Load JSON dataset
# -----------------------------
with open("output.json", "r") as f:
    dataset = json.load(f)

# -----------------------------
# Initialize app
# -----------------------------
app = FastAPI(title="Earthquake API")

# Setup CORS to allow requests from Flutter Web / Chrome
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific domains like "http://localhost:X"
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# =============================
# ===== PREDICT ENDPOINT ======
# =============================

class InputData(BaseModel):
    Magnitude: float
    Rjb_km: float
    Vs30_m_s: float
    Hypo_Depth_km: float
    Critical_Accel_g: float
    PGA_g: float

def preprocess(data: InputData):
    log_Rjb = np.log1p(data.Rjb_km)
    log_Vs30 = np.log1p(data.Vs30_m_s)
    log_H = np.log1p(data.Hypo_Depth_km)
    log_Ac = np.log1p(data.Critical_Accel_g)
    log_PGA = np.log1p(data.PGA_g)
    M_R = data.Magnitude * data.Rjb_km

    return np.array([[
        data.Magnitude,
        log_Rjb,
        log_Vs30,
        log_H,
        log_Ac,
        log_PGA,
        M_R
    ]])

@app.post("/predict")
def predict(data: InputData):
    X = preprocess(data)
    log_pred = model.predict(X)[0]
    final_pred = np.expm1(log_pred)

    return {
        "Predicted_Max_RotD_Disp_cm": float(final_pred)
    }

# =============================
# ===== GET PARAMS ============
# =============================

class LocationInput(BaseModel):
    latitude: float
    longitude: float

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

@app.post("/getparams")
def get_params(loc: LocationInput):

    lat = loc.latitude
    lon = loc.longitude

    points = []

    # Compute distances
    for item in dataset:
        d = haversine(lat, lon, item["Latitude"], item["Longitude"])
        points.append((d, item))

    # Sort by distance
    points.sort(key=lambda x: x[0])

    # Take 6 nearest (like hexagon idea)
    k = 6
    nearest = points[:k]

    distances = np.array([p[0] for p in nearest])
    distances = np.where(distances == 0, 1e-6, distances)

    weights = 1 / distances
    weights = weights / weights.sum()

    keys = [
        "Magnitude", "Rjb_km", "Vs30_m_s",
        "Hypo_Depth_km", "Critical_Accel_g", "PGA_g"
    ]

    weighted_params = {}

    for key in keys:
        weighted_params[key] = float(sum(
            weights[i] * nearest[i][1]["Averages"][key]
            for i in range(k)
        ))

    # Return RSN + coords used
    rsn_list = " ".join(str(p[1]["RSN"]) for p in nearest)
    lat_list = " ".join(str(p[1]["Latitude"]) for p in nearest)
    lon_list = " ".join(str(p[1]["Longitude"]) for p in nearest)

    return {
        "Interpolated_Params": weighted_params,
        "RSNs_used": rsn_list,
        "Latitudes_used": lat_list,
        "Longitudes_used": lon_list
    }

# =============================
# ===== ROOT ==================
# =============================

@app.get("/")
def home():
    return {"message": "API running"}

# -------------------------
# 🌱 SOIL DATA (SoilGrids with retry)
# -------------------------
def get_soil(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=clay&property=sand&property=bdod&depth=0-30cm"

    for i in range(3):
        try:
            res = requests.get(url, timeout=5)

            if res.status_code == 200:
                data = res.json()
                layers = data.get('properties', {}).get('layers', [])

                clay = sand = bdod = 0

                for layer in layers:
                    try:
                        val = layer['depths'][0]['values']['mean']
                        if layer['name'] == 'clay':
                            clay = val
                        elif layer['name'] == 'sand':
                            sand = val
                        elif layer['name'] == 'bdod':
                            bdod = val
                    except:
                        continue

                return clay, sand, bdod

            else:
                print(f"⚠️ Soil retry {i+1}: {res.status_code}")

        except Exception as e:
            print(f"💥 Soil retry {i+1}: {e}")

        time.sleep(1)

    print("❌ Soil data failed completely")
    return 25, 45, 1400

# -------------------------
# 🚀 FOS API
# -------------------------
@app.get("/fos")
def compute_fos(lat: float, lon: float, rainfall: float = 0.0):
    try:
        print(f"\n🌍 Request → lat={lat}, lon={lon}, rainfall={rainfall}")

        point = ee.Geometry.Point([lon, lat])

        # -------------------------
        # 🌱 Soil (SoilGrids)
        # -------------------------
        clay, sand, bdod = get_soil(lat, lon)
        print(f"🌱 Soil → Clay={clay}, Sand={sand}, BDOD={bdod}")

        # -------------------------
        # ⛰️ Slope (GEE SAFE)
        # -------------------------
        try:
            dem = ee.Image("USGS/SRTMGL1_003")
            slope_img = ee.Terrain.slope(dem)

            sample = slope_img.sample(point, 30).first()

            slope_val = sample.get('slope').getInfo() if sample else 0
        except Exception as e:
            print("💥 Slope fetch failed:", e)
            slope_val = 0

        print(f"⛰️ Slope = {slope_val}")

        theta = math.radians(slope_val)

        # -------------------------
        # ⚙️ PARAMETERS
        # -------------------------
        phi = 20 + 0.3 * sand - 0.2 * clay
        cohesion = 0.5 * clay
        gamma = bdod * 9.81 / 1000

        # -------------------------
        # 💧 PORE PRESSURE
        # -------------------------
        u = rainfall * 0.1
        z = 2

        # -------------------------
        # 🧮 FOS
        # -------------------------
        denom = gamma * z * math.sin(theta) * math.cos(theta)

        if denom == 0:
            fos = None
        else:
            num = cohesion + (
                (gamma * z * (math.cos(theta) ** 2) - u)
                * math.tan(math.radians(phi))
            )
            fos = num / denom

        print(f"🧮 FOS = {fos}")

        return {
            "clay": clay,
            "sand": sand,
            "phi": phi,
            "c": cohesion,
            "gamma": gamma,
            "slope": slope_val,
            "rainfall": rainfall,
            "FOS": fos
        }

    except Exception as e:
        print(f"💥 ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=3000)
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app,host='0.0.0.0', port=port)