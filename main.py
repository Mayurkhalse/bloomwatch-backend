from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.responses import JSONResponse
import firebase_admin
from firebase_admin import credentials, firestore,db
import uvicorn
import pyrebase
from fastapi import Body
import requests
from dotenv import load_dotenv
import os
import google.generativeai as genai
import re
import json
from statistics import mean
from collections import Counter, defaultdict
import ee
from collections import defaultdict
from datetime import datetime
from shapely.geometry import Polygon
import math
import json
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content as genai_content

# --- Firebase setup ---
cred = credentials.Certificate("bloomwatch-70da0-firebase-adminsdk-fbsvc-042a03c59e.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://bloomwatch-70da0-default-rtdb.firebaseio.com/'
})

firebaseConfig = {
  "apiKey": "AIzaSyA9S2gTCxOwt5_Yql0K9N5VtijGjr5qUmo",
  "authDomain": "bloomwatch-70da0.firebaseapp.com",
  "databaseURL": "https://bloomwatch-70da0-default-rtdb.firebaseio.com",
  "projectId": "bloomwatch-70da0",
  "storageBucket": "bloomwatch-70da0.firebasestorage.app",
  "messagingSenderId": "977068987023",
  "appId": "1:977068987023:web:cb5caf72b64048621516fa"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
dbf = firestore.client()

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-1.5-pro")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set. Set it in .env")

# configure SDK (works if package is installed and supports configure)
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    # not fatal — we fall back to REST below
    print("genai.configure failed (SDK may be unavailable). Will fallback to REST:", e)




def extract_text_from_content_block(content_block):
    """
    Safely extracts text from a Content object (protobuf) returned by Gemini SDK.
    """
    # Content object may have different fields depending on type
    # Commonly: content_block.text
    if isinstance(content_block, genai_content.Content):
        if content_block.WhichOneof("content") == "text":
            return content_block.text
    return ""



def transform_region_data(region_data: dict):
    """
    Transform region_data JSON into mockBloomEvents structure.
    Historical data = dates < today
    Predicted data = dates >= today
    Skip any data entries with a year < 2023.
    """
    today = datetime.today().date()
    min_year = 2023  # ✅ ignore data before this year

    events = []
    event_id = 1

    for lat_key, lon_dict in region_data.items():
        # Convert coordinate keys like "45_099985251651" to float 45.099985251651
        lat = float(lat_key.replace("_", ".", 1).replace("_", ""))
        
        for lon_key, data in lon_dict.items():
            lon = float(lon_key.replace("_", ".", 1).replace("_", ""))

            historical = []
            predicted = []

            for date_str, values in data.items():
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

                # ✅ Skip old data (before 2023)
                if date_obj.year < min_year:
                    continue

                # Build trend object (without chlorophyll)
                trend = {
                    "date": date_str,
                    "severity": (
                        1 if values["severity"] == "Low"
                        else 2 if values["severity"] == "Moderate"
                        else 3
                    ),
                    "severityLabel": values["severity"],  # keep original label
                    "evi": values["evi"],
                    "ndvi": values["ndvi"]
                }

                # Sort into historical/predicted
                if date_obj < today:
                    historical.append(trend)
                else:
                    predicted.append(trend)

            # ✅ Only add if at least one valid record exists
            if not (historical or predicted):
                continue

            # Create final bloom event object
            all_trends = historical + predicted
            event = {
                "id": str(event_id),
                "coordinates": [lat, lon],
                "severity": max((t["severity"] for t in all_trends), default=1),
                "affectedArea": 0,
                "date": all_trends[-1]["date"] if all_trends else str(today),
                "historicalTrends": historical,
                "predictedTrends": predicted
            }

            events.append(event)
            event_id += 1

    return events


# //supporting functions
def convert_daily_to_monthly(daily_data: dict) -> dict:
    """
    Converts daily NDVI/EVI data into monthly data.
    
    Args:
        daily_data (dict): Daily JSON data with lat/lon and daily NDVI/EVI.
        
    Returns:
        dict: Monthly data grouped by rounded lat/lon and first of month,
              with max NDVI/EVI per month and corresponding severity/flag.
    """
    monthly_data = {}

    for key, region in daily_data.items():
        lat = region["lat"]
        lon = region["lon"]

        # Format lat/lon keys (like your example)
        lat_key = f"{round(lat, 1)}".replace(".", "_")
        lon_key = f"{round(lon, 1)}".replace(".", "_")

        # Group data by month
        month_group = defaultdict(list)
        for date_str, values in region.items():
            if date_str in ["lat", "lon"]:
                continue
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            month_key = date_obj.strftime("%Y-%m")
            month_group[month_key].append(values)

        # Pick max NDVI/EVI per month
        monthly_region_data = {}
        for month, entries in month_group.items():
            max_entry = max(entries, key=lambda x: (x["ndvi"], x["evi"]))
            monthly_region_data[f"{month}-01"] = {
                "ndvi": max_entry["ndvi"],
                "evi": max_entry["evi"],
                "flag": max_entry.get("flag", 0),
                "severity": max_entry.get("severity", "Low")
            }

        monthly_data.setdefault(lat_key, {})[lon_key] = monthly_region_data

    return monthly_data

def format_and_save_data(data):
    ref = db.reference('/region_data/')  # root of the DB

    daily_data = data.regional_data  # use the correct attribute

    monthly_data = convert_daily_to_monthly(daily_data)
    ref.update(monthly_data)

    # Optional: save locally
    with open("output.json", "w") as f:
        import json
        json.dump(monthly_data, f, indent=2)
    return monthly_data


#Chatbot helper functions :
# ------------------ helper analysis functions ------------------

def linear_slope(xs, ys):
    """Simple least-squares slope (xs list of numbers, ys list of numbers)"""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def compute_event_metrics(events, top_n=5):
    """
    events is list of event objects from transform_region_data (id, coordinates, historicalTrends, predictedTrends, ...)
    returns a dict of summary metrics and top hotspots
    """
    all_dates = []
    severity_counter = Counter()
    all_ndvi = []
    all_evi = []
    hotspots = []

    for ev in events:
        coords = ev.get("coordinates")
        trends = (ev.get("historicalTrends") or []) + (ev.get("predictedTrends") or [])
        if not trends:
            continue

        dates = []
        ndvis = []
        evis = []
        for t in trends:
            ds = t.get("date")
            try:
                d = datetime.datetime.strptime(ds, "%Y-%m-%d").date()
            except Exception:
                continue
            dates.append(d)
            all_dates.append(d)
            ndv = float(t.get("ndvi", 0) or 0)
            evi = float(t.get("evi", 0) or 0)
            ndvis.append(ndv)
            evis.append(evi)
            all_ndvi.append(ndv)
            all_evi.append(evi)
            # severity may be numeric 1/2/3 or label — normalize to label
            sev = t.get("severityLabel") or t.get("severity") or "Unknown"
            severity_counter[str(sev)] += 1

        if not dates:
            continue

        # trend slope for NDVI (days -> ndvi)
        base_date = min(dates)
        xs = [(d - base_date).days for d in dates]
        slope_ndvi = linear_slope(xs, ndvis) if len(xs) >= 2 else 0.0
        slope_evi = linear_slope(xs, evis) if len(xs) >= 2 else 0.0

        hotspot = {
            "coordinates": coords,
            "max_ndvi": max(ndvis) if ndvis else None,
            "avg_ndvi": mean(ndvis) if ndvis else None,
            "last_date": max(dates).isoformat(),
            "trend_ndvi_slope_per_day": slope_ndvi,
            "trend_evi_slope_per_day": slope_evi,
            "severity": ev.get("severity"),
            "id": ev.get("id"),
        }
        hotspots.append(hotspot)

    hotspots_sorted = sorted(hotspots, key=lambda x: (x["max_ndvi"] or 0), reverse=True)[:top_n]

    summary = {
        "total_events": len(events),
        "earliest_date": min(all_dates).isoformat() if all_dates else None,
        "latest_date": max(all_dates).isoformat() if all_dates else None,
        "severity_counts": dict(severity_counter),
        "avg_ndvi": mean(all_ndvi) if all_ndvi else None,
        "avg_evi": mean(all_evi) if all_evi else None,
        "top_hotspots": hotspots_sorted,
    }
    return summary


def build_prompt(summary: dict, sample_hotspots: list, question: str):
    """
    Build a short but information-dense prompt for the model.
    We ask the model to return JSON only (and include a fallback plain-text explanation).
    """
    prompt_lines = []

    prompt_lines.append("You are an expert marine ecologist and remote-sensing analyst. The user provided bloom events for a coastal region.")
    prompt_lines.append(f"User question: {question}\n")
    prompt_lines.append("SUMMARY METRICS:")
    prompt_lines.append(f"- Total distinct events: {summary.get('total_events')}")
    prompt_lines.append(f"- Date range: {summary.get('earliest_date')} to {summary.get('latest_date')}")
    prompt_lines.append(f"- Average NDVI: {summary.get('avg_ndvi')}")
    prompt_lines.append(f"- Average EVI: {summary.get('avg_evi')}")
    prompt_lines.append(f"- Severity counts: {json.dumps(summary.get('severity_counts', {}))}\n")

    prompt_lines.append("TOP HOTSPOTS (sample):")
    for h in sample_hotspots:
        prompt_lines.append(f"- coords: {h.get('coordinates')}, max_ndvi: {h.get('max_ndvi')}, last_date: {h.get('last_date')}, trend_ndvi_slope/day: {h.get('trend_ndvi_slope_per_day'):.6f}")

    prompt_lines.append("\nTASK: Provide a thorough analysis for the user including (1) brief executive summary (3-5 sentences), (2) key observations/patterns, (3) likely causes, (4) monitoring & mitigation recommendations (specific, prioritized), (5) short forecast/risk assessment for the next 30 days, and (6) concise actionable next steps.\n")
    prompt_lines.append("OUTPUT FORMAT: Return JSON only. The JSON object MUST have these keys: `summary` (string), `insights` (array of strings), `recommendations` (array of strings), `hotspots` (array of hotspot objects with keys: coordinates, max_ndvi, last_date, trend_ndvi_slope_per_day, severity), `time_series_notes` (string). Do not output any extra commentary outside the JSON. If you cannot fully respond, return best-effort JSON.\n")
    prompt_lines.append("Now, produce the JSON response.")

    return "\n".join(prompt_lines)


def extract_json_block(text: str):
    """
    Try to extract JSON object from model output.
    Looks for ```json { ... } ``` first, then { ... } block.
    Returns parsed JSON or None.
    """
    # prefer explicit ```json ... ```
    m = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if not m:
        # fallback to first {...} block
        m = re.search(r"(\{[^}]*\})", text, re.DOTALL)
    if not m:
        return None

    json_text = m.group(1)
    # attempt tidy and parse
    try:
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        # try to fix common issues (trailing commas)
        cleaned = re.sub(r",\s*([}\]])", r"\1", json_text)
        try:
            return json.loads(cleaned)
        except Exception:
            return None


def extract_text_from_gemini_content(content):
    """Safely extract text from Gemini Content object (single or list)."""
    output_text = ""

    # single object with text
    if hasattr(content, "text"):
        output_text = content.text

    # list of objects (legacy behavior)
    elif isinstance(content, list):
        for block in content:
            if hasattr(block, "text"):
                output_text += block.text

    else:
        # unknown content type
        return None

    return output_text


# --- Safe Gemini call function ---
def call_gemini_sdk(prompt: str, model_name: str = "gemini-2.5-flash",
                    max_output_tokens: int = 800, temperature: float = 0.2):
    """
    Calls Gemini safely. Extracts text from the response.
    If text cannot be extracted, it simply skips it.
    Returns the concatenated text or empty string if nothing could be extracted.
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": 0.9,
                "max_output_tokens": max_output_tokens
            }
        )

        # no candidates -> empty
        if not hasattr(response, "candidates") or len(response.candidates) == 0:
            return ""

        first_candidate = response.candidates[0]
        content = getattr(first_candidate, "content", None)
        output_text = ""

        if content is None:
            return ""

        # Check if content has .parts (Gemini v2.5 behavior)
        if hasattr(content, "parts") and content.parts:
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    output_text += part.text

        # fallback: single .text attribute
        elif hasattr(content, "text") and content.text:
            output_text = content.text

        # fallback: content is a list of parts
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, "text") and block.text:
                    output_text += block.text

        print("OUTPUT", output_text)
        return output_text

    except Exception as e:
        print(f"call_gemini_sdk warning: {e}")
        return ""







app = FastAPI()
# CORS settings
origins = [
    "http://localhost:5173",  # your frontend URL
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    # allow only frontend origin
    allow_credentials=True,
    allow_methods=["*"],      # allow all methods (POST, GET, etc.)
    allow_headers=["*"],      # allow all headers
)

# ✅ Signup
@app.post("/addUser")
def add_user(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        uid = user['localId']

        # Save extra details in Firestore
        dbf.collection("users").document(uid).set({
            "username": username,
            "email": email
        })

        return {"message": "User created successfully", "uid": uid}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# ✅ Login
@app.post("/login")
async def login_user(email: str = Form(...), password: str = Form(...)):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        token = user["idToken"]
        uid = user["localId"]

        # Fetch user profile from Firestore
        user_doc = dbf.collection("users").document(uid).get()
        profile = user_doc.to_dict() if user_doc.exists else {}
        print(profile)
        return {"message": "Login successful", "uid": uid, "profile": profile}
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    

@app.post("/addRegion")
async def addRegion(uid: str = Form(...),lat_1: str = Form(...),lat_2: str=Form(...),lan_1: str=Form(...),lan_2: str=Form(...) ):

    try:
        dbf.collection("users").document(uid).set({
            "region":{
                "lat_1":lat_1,
                "lat_2":lat_2,
                "lan_1":lan_1,
                "lan_2":lan_2
            }

        },merge=True)

        # predict_bounds(lat_1,lan_1, lat_2, lan_2)



        return{"message" :"Regions Added sussfully"}
        
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    

# Pydantic model
class DataModel(BaseModel):
    regional_data: Dict[str, Any]  # accept nested JSON

@app.post("/store-data")
async def store_data(data: DataModel):
    try:
        new_data= format_and_save_data(data)
        # Store in Firebase
        
        return {"status": "success", "message": "Data stored in Firebase", "data": new_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""@app.get("/getRegionData")
async def getRegionData(uid: str = Form(...)):
    user_doc = dbf.collection("users").document(uid).get()
    lat_1= user_doc["region"]["lat_1"]
    lan_1= user_doc["region"]["lan_1"]
    lat_2= user_doc["region"]["lat_2"]
    lan_2= user_doc["region"]["lan_2"]

    data = {
        "lat_1": lat_1,
        "lon_1": lan_1,
        "lat_2": lat_2,
        "lon_2": lan_2
    }
    return data"""

@app.get("/getData")
async def get_data(uid: str = Query(..., description="User ID")):
    try:
        user_doc = dbf.collection("users").document(uid).get()
        if not user_doc.exists:
            return {"error": "User not found"}

        # ✅ Convert DocumentSnapshot to dict
        user_data = user_doc.to_dict()
        if "region" not in user_data:
            return {"error": "Region data not found"}

        lat_1 = float(user_data["region"]["lat_1"])
        lon_1 = float(user_data["region"]["lan_1"])
        lat_2 = float(user_data["region"]["lat_2"])
        lon_2 = float(user_data["region"]["lan_2"])

        print(lat_1, lon_1, lat_2, lon_2)

        min_lat = min(lat_1, lat_2)
        max_lat = max(lat_1, lat_2)
        min_lon = min(lon_1, lon_2)
        max_lon = max(lon_1, lon_2)

        region_ref = db.reference('/region_data/')
        region_data = region_ref.get()
        if not region_data:
            return {"error": "No regional data found"}

        filtered_data = {}
        for lat_key, lon_dict in region_data.items():
            lat = float(lat_key.replace('_', '.'))
            if min_lat <= lat <= max_lat:
                for lon_key, data in lon_dict.items():
                    lon = float(lon_key.replace('_', '.'))
                    if min_lon <= lon <= max_lon:
                        if lat_key not in filtered_data:
                            filtered_data[lat_key] = {}
                        filtered_data[lat_key][lon_key] = data
        
        transformed_data = transform_region_data(filtered_data)

        return {"region_data": transformed_data}

    except Exception as e:
        return {"error": str(e)}













# ------------------ /chatbot-analysis endpoint ------------------
# --- /chatbot-analysis endpoint ---
@app.post("/chatbot-analysis")
async def chatbot_analysis(
    payload: dict = Body(..., example={
        "uid": "user-id",
        "region_data": {},           # optional raw region data
        "question": "Please analyze and recommend next steps",
        "top_n": 5
    })
):
    try:
        print(payload)
        uid = payload.get("uid")
        region_data_in = payload.get("region_data")
        question = payload.get("question", "Analyze the user's region data and provide insights and recommendations.")
        top_n = int(payload.get("top_n", 5))

        # 1) Acquire region data
        if region_data_in:
            raw_region_data = region_data_in
        else:
            if not uid:
                raise HTTPException(status_code=400, detail="Either uid or region_data must be provided")
            user_doc = dbf.collection("users").document(uid).get()
            if not user_doc.exists:
                raise HTTPException(status_code=404, detail="User not found")
            user_obj = user_doc.to_dict()
            if "region" not in user_obj:
                raise HTTPException(status_code=400, detail="User has no saved region")

            lat_1 = float(user_obj["region"]["lat_1"])
            lon_1 = float(user_obj["region"]["lan_1"])
            lat_2 = float(user_obj["region"]["lat_2"])
            lon_2 = float(user_obj["region"]["lan_2"])
            min_lat, max_lat = min(lat_1, lat_2), max(lat_1, lat_2)
            min_lon, max_lon = min(lon_1, lon_2), max(lon_1, lon_2)

            region_ref = db.reference('/region_data/')
            all_region_data = region_ref.get() or {}
            raw_region_data = {}
            for lat_key, lon_dict in all_region_data.items():
                lat = float(lat_key.replace('_', '.'))
                if min_lat <= lat <= max_lat:
                    for lon_key, data in lon_dict.items():
                        lon = float(lon_key.replace('_', '.'))
                        if min_lon <= lon <= max_lon:
                            raw_region_data.setdefault(lat_key, {})[lon_key] = data

        # 2) Transform data to events
          # log some keys
        events = raw_region_data if isinstance(raw_region_data, list) else transform_region_data(raw_region_data)
        if not events:
            return {"reply": "No bloom events available for analysis", "insights": None, "parsed": None}

        # 3) Compute metrics & top hotspots
        summary = compute_event_metrics(events, top_n=top_n)
        
        # 4) Build prompt
        prompt = build_prompt(summary, summary.get("top_hotspots", [])[:top_n], question)
        # 5) Call Gemini safely
        model_output = call_gemini_sdk(prompt, model_name="gemini-2.5-flash", max_output_tokens=8000, temperature=0.2)
        # 6) Parse JSON from output if possible
        parsed_json = extract_json_block(model_output)

        print(model_output)  # log full output for debugging
        return JSONResponse({
            "reply_text": model_output,
            "parsed": parsed_json,
            "summary_metrics": summary
        })

    except HTTPException:
        raise
    except Exception as e:
        print("chatbot-analysis error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# --- /bloom-analysis endpoint ---
@app.post("/bloom-analysis")
async def bloom_analysis(payload: dict = Body(..., example={
    "ndvi": 0.65,
    "evi": 0.85,
    "probability": 0.9,
    "severity": "high",
    "flag": "critical"
})):
    try:
        print(payload)
        ndvi = payload.get("ndvi")
        evi = payload.get("evi")
        probability = payload.get("probability")
        severity = payload.get("severity")
        flag = payload.get("flag")

        if None in [ndvi, evi, probability, severity, flag]:
            raise HTTPException(status_code=400, detail="All parameters (ndvi, evi, probability, severity, flag) are required.")

        # Build prompt
        prompt = f"""
        You are an environmental analysis expert specialized in bloom monitoring.
        Based on the following satellite observation parameters, provide a detailed
        analysis of bloom conditions, potential ecological impacts, and management recommendations.

        Parameters:
        - NDVI: {ndvi}
        - EVI: {evi}
        - Bloom Probability: {probability}
        - Severity Level: {severity}
        - Alert Flag: {flag}

        Include:
        1. Interpretation of vegetation and water health based on NDVI and EVI.
        2. Risk assessment of bloom formation.
        3. Ecological implications.
        4. Recommended next steps or mitigation strategies.
        """

        # Call Gemini safely
        response_text = call_gemini_sdk(prompt, model_name="gemini-2.5-flash", max_output_tokens=800, temperature=0.2)
        print(response_text)  # log full output for debugging
        return {"analysis": response_text}

    except HTTPException:
        raise
    except Exception as e:
        print("bloom-analysis error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
def health():
    return {"status": "ok"}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
