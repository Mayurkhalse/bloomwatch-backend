# main.py
import os
import json
import re
import math
from collections import Counter, defaultdict
from statistics import mean
from datetime import datetime, date

from typing import Dict, Any

from fastapi import FastAPI, Form, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Firebase / Pyrebase / Google GenAI
import firebase_admin
from firebase_admin import credentials, firestore, db as firebase_db
import pyrebase
import requests
from dotenv import load_dotenv

# NOTE: The google generativeai SDK/package name may vary in your environment.
# Keep your existing import if it works in your environment.
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content as genai_content

load_dotenv()  # safe for local dev, Render will use env vars

# --- Environment / Secrets (set these in Render Dashboard) ---
# - FIREBASE_CRED_JSON: (optional) full service account JSON string
# - FIREBASE_CRED_FILE: (optional) path to uploaded secret file on Render (preferred)
# - PYREBASE_CONFIG_JSON: JSON string for pyrebase config OR set individual keys
# - GEMINI_API_KEY: generative model API key
# - GEN_MODEL: optional model name

# --- ENVIRONMENT VARIABLES ---
FIREBASE_CRED_FILE = os.getenv("FIREBASE_CRED_FILE")
FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-1.5-pro")

# --- INITIALIZE FIREBASE ---
def init_firebase():
    if not firebase_admin._apps:
        if FIREBASE_CRED_FILE and os.path.exists(FIREBASE_CRED_FILE):
            cred = credentials.Certificate(FIREBASE_CRED_FILE)
        elif FIREBASE_CRED_JSON:
            cred = credentials.Certificate(json.loads(FIREBASE_CRED_JSON))
        else:
            raise RuntimeError("Firebase credentials not provided.")
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.getenv("FIREBASE_DATABASE_URL", "")
        })
    return firestore.client()

try:
    dbf = init_firebase()
except Exception as e:
    print("Firebase init failed:", e)
    dbf = None

# --- GEMINI SETUP ---
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("genai.configure failed:", e)

# --- Utility functions (cleaned/fixed) ---
def extract_text_from_content_block(content_block):
    if isinstance(content_block, genai_content.Content):
        if content_block.WhichOneof("content") == "text":
            return content_block.text
    return ""

def transform_region_data(region_data: dict):
    today = datetime.today().date()
    min_year = 2023
    events = []
    event_id = 1

    for lat_key, lon_dict in region_data.items():
        # Convert keys like "45_099985251651" -> 45.099985251651
        try:
            lat = float(lat_key.replace("_", ".", 1).replace("_", ""))
        except Exception:
            continue

        for lon_key, data in lon_dict.items():
            try:
                lon = float(lon_key.replace("_", ".", 1).replace("_", ""))
            except Exception:
                continue

            historical = []
            predicted = []
            for date_str, values in data.items():
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                except Exception:
                    continue

                if date_obj.year < min_year:
                    continue

                trend = {
                    "date": date_str,
                    "severity": (
                        1 if values.get("severity") == "Low"
                        else 2 if values.get("severity") == "Moderate"
                        else 3
                    ),
                    "severityLabel": values.get("severity", "Unknown"),
                    "evi": values.get("evi", 0),
                    "ndvi": values.get("ndvi", 0)
                }

                if date_obj < today:
                    historical.append(trend)
                else:
                    predicted.append(trend)

            if not (historical or predicted):
                continue

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

def convert_daily_to_monthly(daily_data: dict) -> dict:
    monthly_data = {}
    for key, region in daily_data.items():
        lat = region["lat"]
        lon = region["lon"]
        lat_key = f"{round(lat, 1)}".replace(".", "_")
        lon_key = f"{round(lon, 1)}".replace(".", "_")

        month_group = defaultdict(list)
        for date_str, values in region.items():
            if date_str in ["lat", "lon"]:
                continue
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue
            month_key = date_obj.strftime("%Y-%m")
            month_group[month_key].append(values)

        monthly_region_data = {}
        for month, entries in month_group.items():
            max_entry = max(entries, key=lambda x: (x.get("ndvi", 0), x.get("evi", 0)))
            monthly_region_data[f"{month}-01"] = {
                "ndvi": max_entry.get("ndvi", 0),
                "evi": max_entry.get("evi", 0),
                "flag": max_entry.get("flag", 0),
                "severity": max_entry.get("severity", "Low")
            }

        monthly_data.setdefault(lat_key, {})[lon_key] = monthly_region_data

    return monthly_data

def format_and_save_data(data):
    if firebase_db is None:
        raise RuntimeError("Realtime DB is not initialized.")
    ref = firebase_db.reference('/region_data/')
    daily_data = data.regional_data
    monthly_data = convert_daily_to_monthly(daily_data)
    ref.update(monthly_data)
    with open("output.json", "w") as f:
        json.dump(monthly_data, f, indent=2)
    return monthly_data

# Analysis helpers
def linear_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den != 0 else 0.0

def compute_event_metrics(events, top_n=5):
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
                d = datetime.strptime(ds, "%Y-%m-%d").date()
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
            sev = t.get("severityLabel") or t.get("severity") or "Unknown"
            severity_counter[str(sev)] += 1

        if not dates:
            continue

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
    m = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{(?:.|\n)*\})", text, re.DOTALL)
    if not m:
        return None
    json_text = m.group(1)
    try:
        return json.loads(json_text)
    except Exception:
        cleaned = re.sub(r",\s*([}\]])", r"\1", json_text)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

def extract_text_from_gemini_content(content):
    output_text = ""
    if hasattr(content, "text"):
        output_text = content.text
    elif isinstance(content, list):
        for block in content:
            if hasattr(block, "text"):
                output_text += block.text
    else:
        return None
    return output_text

def call_gemini_sdk(prompt: str, model_name: str = "gemini-2.5-flash",
                    max_output_tokens: int = 800, temperature: float = 0.2):
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY missing; skipping Gemini call.")
        return ""
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
        if not hasattr(response, "candidates") or len(response.candidates) == 0:
            return ""
        first_candidate = response.candidates[0]
        content = getattr(first_candidate, "content", None)
        output_text = ""
        if content is None:
            return ""
        if hasattr(content, "parts") and content.parts:
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    output_text += part.text
        elif hasattr(content, "text") and content.text:
            output_text = content.text
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, "text") and block.text:
                    output_text += block.text
        print("Gemini output length:", len(output_text))
        return output_text
    except Exception as e:
        print(f"call_gemini_sdk warning: {e}")
        return ""

# --- FastAPI app ---
app = FastAPI()

origins = [
    os.getenv("FRONTEND_URL", "http://localhost:5173"),
    "http://127.0.0.1:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/addUser")
def add_user(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    try:
        if auth is None:
            raise HTTPException(status_code=500, detail="Firebase auth not initialized.")
        user = auth.create_user_with_email_and_password(email, password)
        uid = user['localId']
        dbf.collection("users").document(uid).set({"username": username, "email": email})
        return {"message": "User created successfully", "uid": uid}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/login")
async def login_user(email: str = Form(...), password: str = Form(...)):
    try:
        if auth is None:
            raise HTTPException(status_code=500, detail="Firebase auth not initialized.")
        user = auth.sign_in_with_email_and_password(email, password)
        token = user["idToken"]
        uid = user["localId"]
        user_doc = dbf.collection("users").document(uid).get()
        profile = user_doc.to_dict() if user_doc.exists else {}
        return {"message": "Login successful", "uid": uid, "profile": profile}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/addRegion")
async def addRegion(uid: str = Form(...), lat_1: str = Form(...), lat_2: str = Form(...), lan_1: str = Form(...), lan_2: str = Form(...)):
    try:
        dbf.collection("users").document(uid).set({
            "region": {
                "lat_1": lat_1,
                "lat_2": lat_2,
                "lan_1": lan_1,
                "lan_2": lan_2
            }
        }, merge=True)
        return {"message": "Regions Added successfully"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

class DataModel(BaseModel):
    regional_data: Dict[str, Any]

@app.post("/store-data")
async def store_data(data: DataModel):
    try:
        new_data = format_and_save_data(data)
        return {"status": "success", "message": "Data stored in Firebase", "data": new_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getData")
async def get_data(uid: str = Query(..., description="User ID")):
    try:
        user_doc = dbf.collection("users").document(uid).get()
        if not user_doc.exists:
            return {"error": "User not found"}
        user_data = user_doc.to_dict()
        if "region" not in user_data:
            return {"error": "Region data not found"}

        lat_1 = float(user_data["region"]["lat_1"])
        lon_1 = float(user_data["region"]["lan_1"])
        lat_2 = float(user_data["region"]["lat_2"])
        lon_2 = float(user_data["region"]["lan_2"])

        min_lat = min(lat_1, lat_2)
        max_lat = max(lat_1, lat_2)
        min_lon = min(lon_1, lon_2)
        max_lon = max(lon_1, lon_2)

        region_ref = firebase_db.reference('/region_data/')
        region_data = region_ref.get() or {}
        if not region_data:
            return {"error": "No regional data found"}

        filtered_data = {}
        for lat_key, lon_dict in region_data.items():
            try:
                lat = float(lat_key.replace('_', '.'))
            except Exception:
                continue
            if min_lat <= lat <= max_lat:
                for lon_key, data in lon_dict.items():
                    try:
                        lon = float(lon_key.replace('_', '.'))
                    except Exception:
                        continue
                    if min_lon <= lon <= max_lon:
                        filtered_data.setdefault(lat_key, {})[lon_key] = data

        transformed_data = transform_region_data(filtered_data)
        return {"region_data": transformed_data}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chatbot-analysis")
async def chatbot_analysis(payload: dict = Body(...)):
    try:
        uid = payload.get("uid")
        region_data_in = payload.get("region_data")
        question = payload.get("question", "Analyze the user's region data and provide insights and recommendations.")
        top_n = int(payload.get("top_n", 5))

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
            region_ref = firebase_db.reference('/region_data/')
            all_region_data = region_ref.get() or {}
            raw_region_data = {}
            for lat_key, lon_dict in all_region_data.items():
                try:
                    lat = float(lat_key.replace('_', '.'))
                except Exception:
                    continue
                if min_lat <= lat <= max_lat:
                    for lon_key, data in lon_dict.items():
                        try:
                            lon = float(lon_key.replace('_', '.'))
                        except Exception:
                            continue
                        if min_lon <= lon <= max_lon:
                            raw_region_data.setdefault(lat_key, {})[lon_key] = data

        events = raw_region_data if isinstance(raw_region_data, list) else transform_region_data(raw_region_data)
        if not events:
            return {"reply": "No bloom events available for analysis", "insights": None, "parsed": None}

        summary = compute_event_metrics(events, top_n=top_n)
        prompt = build_prompt(summary, summary.get("top_hotspots", [])[:top_n], question)
        model_output = call_gemini_sdk(prompt, model_name=os.getenv("GEN_MODEL", "gemini-2.5-flash"), max_output_tokens=8000, temperature=0.2)
        parsed_json = extract_json_block(model_output)

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

@app.post("/bloom-analysis")
async def bloom_analysis(payload: dict = Body(...)):
    try:
        ndvi = payload.get("ndvi")
        evi = payload.get("evi")
        probability = payload.get("probability")
        severity = payload.get("severity")
        flag = payload.get("flag")

        if None in [ndvi, evi, probability, severity, flag]:
            raise HTTPException(status_code=400, detail="All parameters (ndvi, evi, probability, severity, flag) are required.")

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
        """

        response_text = call_gemini_sdk(prompt, model_name=os.getenv("GEN_MODEL", "gemini-2.5-flash"), max_output_tokens=800, temperature=0.2)
        return {"analysis": response_text}
    except HTTPException:
        raise
    except Exception as e:
        print("bloom-analysis error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

# Start block for local dev only. Render uses start command (uvicorn main:app --host 0.0.0.0 --port $PORT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
