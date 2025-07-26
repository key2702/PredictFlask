from flask import Flask, request, jsonify
import numpy as np
from scipy.spatial.distance import euclidean
import joblib
import pandas as pd
from flask_cors import CORS
from sqlalchemy import create_engine
import urllib.parse
import os
import json
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import time
from math import radians, sin, cos, sqrt, atan2
from twilio.rest import Client
from dotenv import load_dotenv

# Charger les variables d’environnement
load_dotenv()

# ========== CONFIG FLASK ==========
app = Flask(__name__)
# Configuration CORS plus permissive
CORS(app, resources={
    r"/predict": {
        "origins": ["https://zonesafe.somee.com","http://localhost:5077", "http://127.0.0.1:5077"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ========== CHARGEMENT DU MODELE ==========
try:
    model = joblib.load("risk_classification_model.pkl")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {str(e)}")
    model = None

# ========== TWILIO ==========
def send_whatsapp_alert(to_number, message_body):
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_SANDBOX_NUMBER")
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            from_=from_number,
            to=f"whatsapp:{to_number}",
            body=message_body
        )
        print(f"Message envoyé à {to_number}, SID: {message.sid}")
    except Exception as e:
        print(f"Erreur Twilio: {str(e)}")

# ========== OUTILS ==========
def haversine_distance(coord1, coord2):
    R = 6371.0
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = radians(coord2[1]) - radians(coord1[1])
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_sqlalchemy_engine():
    USER = "ngombe3_SQLLogin_1"
    PASSWORD = "i6azcp949a"
    HOST = "ZonesafeBD.mssql.somee.com"
    PORT = "1433"  
    DATABASE = "ZonesafeDB"
    conn_str = f"mssql+pymssql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    return create_engine(conn_str)

def fetch_user_infrastructures(user_id):
    engine = get_sqlalchemy_engine()
    query = """
    SELECT  
        esa.typezone AS Type_zone,
        osm.hauteur AS hauteur,
        mnt.pente AS pente,
        mnt.altitude AS altitude,
        osm.typemateriel AS type_materiaux,
        osm.latitude AS latitude,
        osm.longitude AS longitude,
        osm.type_infra AS type_infra,
        osm.nom_infra AS nom_infra
    FROM 
        OSM osm
        INNER JOIN ESA esa ON osm.id = esa.id_osm  
        INNER JOIN MNT mnt ON osm.id = mnt.id_osm  
    WHERE 
        osm.id_user = ?  
    """
    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection, params=(user_id,))
            return df.to_dict('records')
    except Exception as e:
        print(f"Erreur BDD: {str(e)}")
        return None
    finally:
        engine.dispose()        
def predict_user(user_id, user_lat=0.0, user_lng=0.0, user_phone=None):
    try:
        objects = fetch_user_infrastructures(user_id)
        if not objects:
            return None, "Aucune infrastructure trouvée pour cet utilisateur"
    except Exception as e:
        return None, f"Erreur récupération données: {str(e)}"

    user_coords = np.array([user_lat, user_lng])
    results = []
    notification_sent = False
    risque_proche = None
    distance_min_risque = float("inf")
    
    safe_proche = None
    distance_min_safe = float("inf")
    for obj in objects:
        try:
            hauteur = 0.0 if obj["hauteur"] is None else float(obj["hauteur"])
            pente = 0.0 if obj["pente"] is None else float(obj["pente"])
            altitude = 0.0 if obj["altitude"] is None else float(obj["altitude"])
            latitude = float(obj['latitude'])  
            longitude = float(obj['longitude'])  
            
            features = pd.DataFrame([{
                "Type_zone": obj["Type_zone"],
                "hauteur": hauteur,
                "pente": pente,
                "altitude": altitude,
                "type_materiaux": obj["type_materiaux"]
            }])
            
            risque = model.predict(features)[0]
            obj_coords = np.array([latitude, longitude])
            distance = haversine_distance(user_coords, obj_coords)
            poids = 1 / (distance + 1e-5)
            
            output_obj = {
                **obj,
                "risque": int(risque),
                "distance": float(distance),
                "poids": float(poids)
            }

            if int(risque) in [2, 3] and distance <= 30:
                if distance < distance_min_risque:
                    risque_proche = output_obj
                    distance_min_risque = distance
            else:
                if distance < distance_min_safe:
                    safe_proche = output_obj
                    distance_min_safe = distance
          

        except Exception as e:
            print(f"Erreur objet {obj.get('nom_infra')}: {str(e)}")
            continue
      # 🔔 Envoi WhatsApp si risque détecté
    if risque_proche and user_phone:
        msg = f"⚠️ Alerte ! Risque niveau {risque_proche['risque']} détecté à {risque_proche['distance']}m de {risque_proche['type_infra']}."
        send_whatsapp_alert(user_phone, msg)
        return [risque_proche], None

   
    if safe_proche:
        return [safe_proche], None
    return None, "Aucune infrastructure exploitable détectée"

# ========== ENDPOINTS ==========
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'user_id' not in data:
        return jsonify({"error": "Champ requis: user_id"}), 400

    try:
        user_id = int(data['user_id'])
        user_lat = float(data.get('user_lat', 0))
        user_lng = float(data.get('user_lng', 0))
        user_phone = "+243974902107"  # numéro du user envoyé dans la requête
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Format invalide: {str(e)}"}), 400

    result, error = predict_user(user_id, user_lat, user_lng, user_phone)
    if error:
        return jsonify({"error": error}), 404

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)