# -*- coding: utf-8 -*-
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import time
import os
from PIL import Image
# Import specifico per EXIF da Pillow
from PIL.ExifTags import TAGS, GPSTAGS
import io
import yt_dlp
import traceback
import ssl
import certifi
import base64
import json
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime
import urllib.parse
import re
# Import per le feature AI
import openai
import folium
from folium.plugins import MarkerCluster
from dotenv import load_dotenv
import markdown2
try:
    import pdfkit
except ImportError:
    pdfkit = None
    print("WARNING: 'pdfkit' not installed. PDF export disabled.")
import subprocess

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY" and OPENAI_API_KEY is not None:
    openai.api_key = OPENAI_API_KEY # Imposta la chiave per la libreria openai
    print("INFO: OpenAI API Key loaded.")
else:
    print("WARNING: OpenAI API Key not found or invalid. AI features will be disabled.")
    OPENAI_API_KEY = None

# --- Parameters & Keywords ---
FISH_DETECTION_THRESHOLD = 0.05
MAX_VIDEO_FRAMES_TO_PROCESS = 150
FRAME_SAMPLE_RATE = 5
PREDICTION_CLASSES_TO_SHOW = 5
FISH_KEYWORDS = [ # Lista parole chiave completa
    'fish', 'pesce', 'trout', 'trota', 'oncorhynchus mykiss', 'marble trout',
    'trota marmorata', 'brown trout', 'trota fario', 'rainbow trout', 'trota iridea',
    'lake trout', 'trota lacustre', 'char', 'salmerino', 'salvelinus', 'arctic char',
    'salmerino alpino', 'brook trout', 'salmerino di fonte', 'grayling', 'temolo',
    'thymallus', 'pike', 'luccio', 'esox lucius', 'perch', 'persico',
    'perca fluviatilis', 'european perch', 'persico reale', 'zander', 'luccioperca',
    'sandre', 'sander lucioperca', 'whitefish', 'coregone', 'lavarello', 'coregonus',
    'carp', 'carpa', 'cyprinus carpio', 'tench', 'tinca', 'tinca tinca', 'barbel',
    'barbo', 'barbus barbus', 'chub', 'cavedano', 'squalius cephalus', 'rudd',
    'scardola', 'scardinius erythrophthalmus', 'eel', 'anguilla', 'anguilla anguilla',
    'burbot', 'bottatrice', 'lota lota', 'gudgeon', 'gobione', 'gobio gobio',
    'minnow', 'vairone', 'sanguinerola', 'phoxinus phoxinus', 'bleak', 'alborella',
    'alburnus', 'roach', 'pigo', 'rutilus', 'bream', 'abramide', 'abramis',
    'crucian carp', 'carassio', 'carassius carassius', 'common nase', 'catfish',
    'pesce gatto', 'ameiurus melas', 'ictalurus', 'wels catfish', 'siluro',
    'silurus', 'sturgeon', 'storione', 'bass', 'largemouth bass', 'persico trota',
    'boccalone', 'micropterus salmoides', 'pumpkinseed', 'sunfish', 'persico sole',
    'lepomis gibbosus', 'spined loach', 'cobite comune', 'cobitis taenia',
    'balkan loach', 'cobite mascherata', 'sabanejewia', 'three-spined stickleback',
    'spinnarello', 'gasterosteus aculeatus', 'prussian carp', 'carassio dorato',
    'gibelio', 'gobi', 'ghiozzo', 'padanian goby', 'ghiozzo padano',
    'padogobius martensii', 'bullhead', 'scazzone', 'cottus gobio', 'grass carp',
    'carpa erbivora', 'amur', 'ctenopharyngodon idella', 'topmouth gudgeon',
    'pseudorasbora', 'pseudorasbora parva', 'tuna', 'tonno', 'salmon', 'salmone',
    'cod', 'merluzzo', 'haddock', 'eglefino', 'mackerel', 'sgombro', 'sardine',
    'sardina', 'anchovy', 'acciuga', 'sea bream', 'orata', 'sea bass', 'spigola',
    'branzino', 'mullet', 'cefalo', 'muggine', 'swordfish', 'pesce spada', 'shark',
    'squalo', 'grouper', 'cernia', 'snapper', 'dentice', 'sole', 'sogliola',
    'goldfish', 'pesce rosso', 'koi', 'carpa koi', 'clownfish', 'pesce pagliaccio',
    'great white shark', 'tiger shark', 'hammerhead shark', 'electric eel',
    'coelacanth', 'rockfish', 'lionfish', 'puffer', 'stingray', 'starfish', 'anemone fish'
]


# --- Global Variables ---
fish_model = None; fish_model_loading_error = None; openai_api_error = None; MODEL_INPUT_SIZE = (224, 224)
UPLOADS_DIR = "uploads" # Cartella per immagini caricate nell'editor report
if not os.path.exists(UPLOADS_DIR):
    try: os.makedirs(UPLOADS_DIR); print(f"INFO: Created directory '{UPLOADS_DIR}'.")
    except OSError as e: print(f"ERROR: Cannot create '{UPLOADS_DIR}': {e}"); UPLOADS_DIR = None

# --- SSL Handling ---
try: cafile = certifi.where(); ssl_context = ssl.create_default_context(cafile=cafile); ssl._create_default_https_context = lambda: ssl_context; print(f"INFO: Using SSL bundle: {cafile}")
except Exception as e: print(f"WARNING: SSL context setup failed: {e}")

# --- Fish Detection Model Loading ---
def load_fish_detection_model():
    global fish_model, fish_model_loading_error;
    if fish_model is not None: return True
    if fish_model_loading_error is not None: return False
    try:
        print("Loading Fish Detection model (MobileNetV2)...")
        fish_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
        _ = fish_model.predict(tf.keras.applications.mobilenet_v2.preprocess_input(np.zeros((1, *MODEL_INPUT_SIZE, 3))), verbose=0)
        print("Fish Detection model loaded successfully.")
        return True
    except Exception as e:
        fish_model_loading_error = f"Fatal Error loading Fish Model: {e}\n{traceback.format_exc()}"
        print(f"\n*** FISH MODEL LOADING FAILED ***\n{fish_model_loading_error}\n******\n"); fish_model = None; return False

# --- OpenAI Client Initialization Check ---
def initialize_chatbot_client():
    global openai_api_error;
    if openai_api_error is None:
        if not OPENAI_API_KEY: openai_api_error = "OpenAI API Key not configured."; print(f"FATAL ERROR: {openai_api_error}"); return False
        try:
            if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY":
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                client.models.list()
                print("INFO: OpenAI API client appears valid.")
        except openai.AuthenticationError as auth_err: openai_api_error = f"OpenAI API Key Invalid: {auth_err}"; print(f"FATAL ERROR: {openai_api_error}"); return False
        except Exception as api_conn_err: openai_api_error = f"OpenAI API Connection Error: {api_conn_err}"; print(f"ERROR: {openai_api_error}"); return False
        return True
    return not openai_api_error

def encode_image_to_base64(image_pil):
    if not isinstance(image_pil, Image.Image): return None;
    try: buf=io.BytesIO(); image_pil.save(buf,format="JPEG"); return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e: print(f"Err b64: {e}"); return None

def create_matplotlib_chart(data_dict):
    print(f"Creating chart: {data_dict.get('title', 'N/A')}, type: {data_dict.get('type', 'N/A')}")
    if not data_dict or not isinstance(data_dict, dict): return None
    chart_type = data_dict.get("type", "bar").lower()
    labels = data_dict.get("labels"); data = data_dict.get("data")
    title = data_dict.get("title", "Grafico"); xlabel = data_dict.get("xlabel"); ylabel = data_dict.get("ylabel")
    if not labels or not isinstance(labels, list) or not data or not isinstance(data, list): print(f"Invalid keys/types: labels={type(labels)}, data={type(data)}"); return None
    valid=False
    if chart_type in ['line', 'bar', 'pie']:
        if len(labels) == len(data) and all(isinstance(d, (int, float)) for d in data): valid = True
    elif chart_type == 'scatter':
        if all(isinstance(p, (list, tuple)) and len(p) >= 2 and all(isinstance(v, (int, float)) for v in p[:2]) for p in data): valid = True
    elif chart_type == 'heatmap':
        if isinstance(labels, list) and len(labels) == 2 and isinstance(labels[0], list) and isinstance(labels[1], list) and isinstance(data, list) and all(isinstance(row, list) for row in data):
            if data and data[0] and len(data) == len(labels[1]) and len(data[0]) == len(labels[0]): valid = True
            elif not data and len(labels[1]) == 0 and len(labels[0]) > 0 : valid = True
    if not valid: print("Chart data validation failed."); return None
    plt.switch_backend('agg'); fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if chart_type == 'line': ax.plot(labels, data, marker='o', linestyle='-', color='#3b82f6')
        elif chart_type == 'bar': ax.bar(labels, data, color='#60a5fa'); ax.set_ylim(0, max(data) * 1.15 if data and max(data) > 0 else 1)
        elif chart_type == 'pie':
            pos_d=[d for d in data if d > 0]; pos_l=[labels[i] for i, d in enumerate(data) if d > 0]
            if pos_d: ax.pie(pos_d, labels=pos_l, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(range(len(pos_d)))); ax.axis('equal'); xlabel, ylabel = None, None
            else: plt.close(fig); return None
        elif chart_type == 'scatter':
             x_vals=[p[0] for p in data]; y_vals=[p[1] for p in data]; ax.scatter(x_vals, y_vals, color='#60a5fa', alpha=0.7)
             if labels and isinstance(labels, list) and len(labels) == len(data):
                for i, txt in enumerate(labels): ax.annotate(txt, (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(0,5), ha='center')
        elif chart_type == 'heatmap':
            if not data: ax.text(0.5, 0.5, 'Nessun dato per heatmap', ha='center', va='center', transform=ax.transAxes)
            else:
                im = ax.imshow(data, cmap='viridis', aspect='auto')
                if labels and len(labels) == 2 and isinstance(labels[0], list) and isinstance(labels[1], list):
                    x_labels_hm, y_labels_hm = labels
                    if data and data[0] and len(x_labels_hm) == len(data[0]) and len(y_labels_hm) == len(data):
                        ax.set_xticks(np.arange(len(x_labels_hm))); ax.set_yticks(np.arange(len(y_labels_hm)))
                        ax.set_xticklabels(x_labels_hm); ax.set_yticklabels(y_labels_hm)
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                plt.colorbar(im)
        else: plt.close(fig); return None
        ax.set_title(title);
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if chart_type in ['bar', 'line', 'scatter'] and labels and (len(labels) > 8 or any(len(str(l)) > 8 for l in labels)): ax.tick_params(axis='x', rotation=45, labelsize=9)
        elif chart_type in ['bar', 'line', 'scatter']: ax.tick_params(axis='x', labelsize=10)
        fig.tight_layout();
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="chart_") as tmp_file: fig.savefig(tmp_file, format='png'); temp_path = tmp_file.name
        print(f"Matplotlib chart saved to: {temp_path}"); return temp_path
    except Exception as e: print(f"Error generating Matplotlib chart plot: {e}\n{traceback.format_exc()}"); plt.close(fig); return None
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

def preprocess_image_for_mobilenet(img_data):
    if isinstance(img_data, np.ndarray): img = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
    elif isinstance(img_data, Image.Image): img = img_data.convert('RGB')
    else: raise TypeError("Input must be PIL Image or OpenCV BGR NumPy array")
    img = img.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_array, axis=0))

def run_fish_prediction(processed_image_tensor):
    if fish_model is None: raise RuntimeError("Fish model not loaded")
    preds = fish_model.predict(processed_image_tensor, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=PREDICTION_CLASSES_TO_SHOW)[0]
    return {label.replace('_', ' ').title(): float(conf) for _, label, conf in decoded}

# --- NUOVA FUNZIONE GPS EXIF (DALLA GUIDA) ---
def get_exif_gps(pil_image):
    lat, lon = None, None # Inizializza lat e lon a None
    try:
        exif_data = pil_image.getexif() # Metodo più recente
        if not exif_data: return None, None

        gps_info_raw = exif_data.get_ifd(0x8825) # Tag per GPS IFD
        if not gps_info_raw: return None, None

        gps_info = {}
        for tag_id, value in gps_info_raw.items():
            gps_tag_name = GPSTAGS.get(tag_id, tag_id)
            gps_info[gps_tag_name] = value

        def convert_to_degrees(value):
            # Gestisce sia tuple di Rational che tuple di float/int
            d = float(value[0].numerator / value[0].denominator if hasattr(value[0], 'numerator') else value[0])
            m = float(value[1].numerator / value[1].denominator if hasattr(value[1], 'numerator') else value[1])
            s = float(value[2].numerator / value[2].denominator if hasattr(value[2], 'numerator') else value[2])
            return d + (m / 60.0) + (s / 3600.0)

        if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info and \
           'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:

            lat_dms = gps_info['GPSLatitude']
            lon_dms = gps_info['GPSLongitude']

            # Verifica che DMS siano tuple di 3 elementi
            if not (isinstance(lat_dms, tuple) and len(lat_dms) == 3 and
                    isinstance(lon_dms, tuple) and len(lon_dms) == 3):
                print("Debug GPS: GPSLatitude/Longitude non sono tuple di 3 elementi come atteso.")
                return None, None

            lat = convert_to_degrees(lat_dms)
            if gps_info['GPSLatitudeRef'] == 'S': # Se 'S' (Sud)
                lat = -lat

            lon = convert_to_degrees(lon_dms)
            if gps_info['GPSLongitudeRef'] == 'W': # Se 'W' (Ovest)
                lon = -lon

            if -90 <= lat <= 90 and -180 <= lon <= 180:
                 print(f"INFO: EXIF GPS found and parsed: Lat={lat:.6f}, Lon={lon:.6f}")
                 return lat, lon
            else:
                print(f"Warn: Calculated GPS coords out of valid range: Lat={lat}, Lon={lon}")
                return None, None
        else:
            # print("Debug GPS: GPSLatitude/Longitude or their Refs are missing.")
            return None, None

    except AttributeError: # pil_image potrebbe non avere getexif()
        # print("Debug GPS: Image object has no getexif method.")
        return None, None
    except Exception as e: # Catch any other unexpected errors during EXIF processing
        print(f"EXIF extraction error: {e}")
        traceback.print_exc()
        return None, None # Ritorna None, None in caso di qualsiasi errore


# --- Fish Detector Analysis Function (MODIFICATA per logging e GPS) ---
def analyze_media(input_image, input_video_path, input_stream_url, current_report_content, current_chart_data, progress=gr.Progress(track_tqdm=True)):
    start_time = time.time(); status_message = "Inizializzazione..."; results_dict = {}; output_media = None
    fish_log_entry_report = ""; fish_log_entry_chart = ""; best_fish_label = None; best_prob = 0.0
    lat, lon = None, None # Variabili per le coordinate

    if not load_fish_detection_model(): error_msg = f"❌ Errore Modello: {fish_model_loading_error or 'Impossibile caricare.'}" ; return None, {}, error_msg, current_report_content, current_chart_data

    current_report_content_str = current_report_content if isinstance(current_report_content, str) else ""
    current_chart_data_str = current_chart_data if isinstance(current_chart_data, str) else ""
    updated_report_content = current_report_content_str
    updated_chart_data = current_chart_data_str

    try:
        media_type = None
        if input_image is not None: # input_image è già un oggetto PIL
            media_type = "Immagine"; print("Analyzing Image..."); output_media = input_image
            processed_tensor = preprocess_image_for_mobilenet(input_image); results_dict = run_fish_prediction(processed_tensor)
            for label, prob_val in results_dict.items():
                if any(k in label.lower() for k in FISH_KEYWORDS) and prob_val >= FISH_DETECTION_THRESHOLD and prob_val > best_prob:
                    best_prob = prob_val; best_fish_label = label
            # Estrai GPS
            lat, lon = get_exif_gps(input_image)
            status_message = f"Analisi immagine completata." + (f" GPS: {lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else " (No GPS EXIF)")

        elif input_video_path is not None:
            media_type = "Video"; print(f"Analyzing Video: {input_video_path}")
            cap = cv2.VideoCapture(input_video_path);
            if not cap.isOpened(): raise ValueError(f"Cannot open video: {input_video_path}")
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
            actual_frame_indices=list(range(0,frame_count,FRAME_SAMPLE_RATE))[:MAX_VIDEO_FRAMES_TO_PROCESS]
            last_results={}; last_bgr=None; temp_best_prob=0.0; temp_best_label=None
            progress(0, desc="Analisi video...")
            for i, idx in enumerate(actual_frame_indices):
                 cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, frame = cap.read()
                 if not ret: break
                 progress((i+1)/len(actual_frame_indices), f"Frame {idx}/{frame_count}..."); last_bgr = frame
                 res = run_fish_prediction(preprocess_image_for_mobilenet(frame)); last_results = res
                 for lbl, p_val in res.items():
                     if any(k in lbl.lower() for k in FISH_KEYWORDS) and p_val>=FISH_DETECTION_THRESHOLD and p_val>temp_best_prob:
                         temp_best_prob=p_val; temp_best_label=lbl
            cap.release(); results_dict = last_results
            if last_bgr is not None: output_media = Image.fromarray(cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB))
            best_fish_label = temp_best_label; best_prob = temp_best_prob
            status_message = f"Analisi video completata ({len(actual_frame_indices)} frames)."
            # GPS non applicabile direttamente a video file in questo workflow
        elif input_stream_url:
            media_type = "Stream"; print(f"Processing Stream: {input_stream_url}"); actual_url = input_stream_url
            if "youtube" in input_stream_url or "youtu.be" in input_stream_url:
                ydl_opts = {'format': 'best[protocol^=http][height<=720]/best', 'quiet': True, 'noplaylist': True, 'simulate': True}
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl: info=ydl.extract_info(input_stream_url, download=False); url=info.get('url')
                    if url: actual_url = url
                except Exception as e: print(f"yt-dlp failed: {e}.")
            cap = cv2.VideoCapture(actual_url, cv2.CAP_FFMPEG); ret, frame = cap.read(); cap.release()
            if not ret or frame is None: raise ValueError("Cannot read stream.")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); output_media = Image.fromarray(frame_rgb)
            results_dict = run_fish_prediction(preprocess_image_for_mobilenet(frame_rgb))
            for lbl, p_val in results_dict.items():
                if any(k in lbl.lower() for k in FISH_KEYWORDS) and p_val >= FISH_DETECTION_THRESHOLD and p_val > best_prob:
                    best_prob = p_val; best_fish_label = lbl
            status_message = "Analisi stream completata."
            # GPS non applicabile a stream
        else: status_message = "ℹ️ Nessun input fornito."; return None, {}, status_message, current_report_content_str, current_chart_data_str

        if best_fish_label:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gps_string_report = f"\nGPS: {lat:.6f}, {lon:.6f}" if lat is not None and lon is not None else ""
            fish_log_entry_report = f"--- Rilevamento {media_type} ({timestamp}) ---\nSpecie: {best_fish_label}\nProb: {best_prob * 100:.1f}%{gps_string_report}\n---"
            chart_label = best_fish_label.split(',')[0].strip()

            # --- CORREZIONE LOGGING GPS E FORMATTAZIONE ---
            lat_str = f"{lat:.6f}" if lat is not None else ""
            lon_str = f"{lon:.6f}" if lon is not None else ""
            fish_log_entry_chart = f"{timestamp},{chart_label},{best_prob * 100:.1f},{lat_str},{lon_str}".strip() # Rimuovi newline qui

            if fish_log_entry_report:
                 updated_report_content = (current_report_content_str.strip() + "\n\n" + fish_log_entry_report.strip()).strip()
            else:
                 updated_report_content = current_report_content_str # Mantieni il vecchio se non c'è entry report

            if fish_log_entry_chart: # Solo se c'è contenuto effettivo nel log chart
                current_chart_data_stripped = updated_chart_data.strip()
                new_entry_stripped = fish_log_entry_chart.strip() # Strip di nuovo per sicurezza
                if current_chart_data_stripped:
                    updated_chart_data = current_chart_data_stripped + "\n" + new_entry_stripped
                else:
                    updated_chart_data = new_entry_stripped
            else:
                 updated_chart_data = current_chart_data_str # Mantieni il vecchio se non c'è entry chart

    except Exception as e:
        status_message = f"❌ Errore analisi: {e}"; traceback.print_exc(); output_media=None; results_dict={};
        # In caso di errore nell'analisi, non modificare i log esistenti
        updated_report_content = current_report_content_str
        updated_chart_data = current_chart_data_str

    processing_time = time.time() - start_time; status_message += f" (Tempo: {processing_time:.2f}s)"
    return output_media, results_dict, status_message, updated_report_content, updated_chart_data


def analyze_webcam_frame(frame_np):
    if frame_np is None: return None, {}
    if not initial_fish_model_status:
         try: frame_pil = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
         except: frame_pil = None
         return frame_pil, {"❌ ERRORE": 1.0, "Modello non caricato": 0.0}
    try:
        processed_tensor = preprocess_image_for_mobilenet(frame_np)
        results_dict = run_fish_prediction(processed_tensor)
        frame_pil_out = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
        return frame_pil_out, results_dict
    except Exception as e:
        print(f"Error webcam frame: {e}\n{traceback.format_exc()}")
        try: frame_pil_err = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
        except: frame_pil_err = None
        return frame_pil_err, {"❌ ERRORE Analisi": 1.0, str(e): 0.0}

def chatbot_response(message_input, chat_history, img_input_pil):
    if not initialize_chatbot_client():
        error_msg=f"❌ Errore API: {openai_api_error or 'Sconosciuto'}"; user_content_display=[{"type":"text","text":message_input}] if message_input else []
        chat_history.append({'role':'user','content':user_content_display}); chat_history.append({'role':'assistant','content':error_msg}); return chat_history,"",None
    user_img_temp_path = None; user_turn_content_display = []
    if message_input and message_input.strip(): user_turn_content_display.append({"type": "text", "text": message_input})
    if img_input_pil:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="chat_user_") as tmp:
                img_input_pil.save(tmp, format="PNG"); user_img_temp_path = tmp.name
                user_turn_content_display.append({"type": "image_url", "image_url": {"url": user_img_temp_path}})
        except Exception as e: print(f"Error saving user input image: {e}")
    if not user_turn_content_display: return chat_history, "", None
    chat_history.append({"role": "user", "content": user_turn_content_display})
    api_messages = [{"role": "system", "content": "Sei Ocean AI, assistente marino e visivo. Rispondi in italiano..."}]
    for message in chat_history[:-1]:
        if message["role"] == "user":
            api_user_content = []
            if isinstance(message["content"], list):
                 for part in message["content"]:
                    if part["type"] == "text": api_user_content.append(part)
                    elif part["type"] == "image_url":
                        img_path = part["image_url"]["url"]
                        if img_path and os.path.exists(img_path):
                            try:
                                img_pil_hist = Image.open(img_path); b64 = encode_image_to_base64(img_pil_hist)
                                if b64: api_user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                            except Exception as e: print(f"Error processing historical image {img_path}: {e}")
            if api_user_content: api_messages.append({"role": "user", "content": api_user_content})
        elif message["role"] == "assistant" and isinstance(message["content"], str): api_messages.append(message)
    current_turn_api_content = []
    if message_input: current_turn_api_content.append({"type": "text", "text": message_input})
    if img_input_pil:
        b64_current = encode_image_to_base64(img_input_pil)
        if b64_current: current_turn_api_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_current}"}})
    if current_turn_api_content: api_messages.append({"role": "user", "content": current_turn_api_content})
    else: chat_history.pop(); return chat_history, "", None
    ai_response_text = "Errore API."
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=api_messages, max_tokens=1500, temperature=0.7)
        ai_response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "L'AI non ha fornito risposta."
    except Exception as e: ai_response_text = f"❌ Errore API: {e}"; print(f"OpenAI Error: {traceback.format_exc()}")
    chat_history.append({"role": "assistant", "content": ai_response_text})
    return chat_history, "", None

def ai_interact_report(current_report_content, user_prompt):
    if not initialize_chatbot_client(): return f"❌ Errore API Report: {openai_api_error}\n\n{current_report_content}"
    if not user_prompt: gr.Warning("Fornisci istruzione."); return current_report_content
    messages=[{"role": "system", "content": "Sei un AI per report marini..."}, {"role": "user", "content": f"Report:\n---\n{current_report_content}\n---\n\nIstruzione: {user_prompt}"}]
    ai_modified_content = current_report_content
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=4000, temperature=0.5)
        if response.choices and response.choices[0].message.content:
            ai_modified_content = response.choices[0].message.content.strip() or current_report_content
            if ai_modified_content == current_report_content.strip(): gr.Info("AI no changes.")
        else: gr.Warning("AI no response.")
    except Exception as e: print(f"Error OpenAI report: {traceback.format_exc()}"); gr.Error(f"❌ Errore API Report: {e}")
    return ai_modified_content

def download_report(report_content):
    if not report_content or not report_content.strip(): gr.Warning("Relazione vuota."); return None
    output_path = None
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S"); fn = f"ocean_report_{ts}.md"
        td = tempfile.mkdtemp(prefix="gradio_report_"); output_path = os.path.join(td, fn)
        with open(output_path, "w", encoding='utf-8') as f: f.write(report_content)
        print(f"Report saved for download: {output_path}"); return output_path
    except Exception as e:
        print(f"Error creating report download: {e}\n{traceback.format_exc()}"); gr.Error(f"Errore creazione file: {e}")
        if output_path and os.path.exists(output_path):
            try: os.remove(output_path)
            except Exception: pass
        return None

def send_report_email_mailto(recipient_email, subject, report_body):
    if not recipient_email or "@" not in recipient_email: return '<span style="color: #f87171;">Email non valida.</span>'
    if not subject: subject = "Report Ocean AI"
    subject_enc = urllib.parse.quote(subject); max_len = 8000; display_body = report_body
    if len(report_body) > max_len: display_body = report_body[:max_len] + "\n\n[...Report troncato...]\n"; gr.Warning(f"Report troncato mailto.")
    body_enc = urllib.parse.quote(display_body); params = urllib.parse.urlencode({'subject': subject_enc, 'body': body_enc}, safe='')
    mailto = f"mailto:{recipient_email}?" + params
    if len(mailto) > 2000: gr.Warning(f"Link mailto lungo.")
    return f'<a href="{mailto}" target="_blank" style="color:#60a5fa;font-weight:bold;">Apri client email</a>'

# --- AI Chart Generation ---
def ai_parse_and_generate_chart(user_chart_prompt, chart_data_log):
    if not initialize_chatbot_client(): error_msg=f"API error: {openai_api_error}"; gr.Error(error_msg); return None, error_msg, None
    if not user_chart_prompt: gr.Warning("Descrivi grafico."); return None,"Inserisci descrizione.",None
    if not chart_data_log: gr.Warning("Log dati vuoto."); return None,"Log dati vuoto.", None
    system_message_content = ("Sei un AI per grafici Matplotlib. Input: log 'timestamp,specie,prob[,lat,lon]' e richiesta. "
                              "Output: JSON tra [CHART_PARAMS][/CHART_PARAMS] con type, title, labels, data, xlabel?, ylabel?. "
                              "Per heatmap giorno/ora: analizza timestamp (giorno 0-6 Lun-Dom, ora 0-23), type='heatmap', "
                              "labels=[['00'...'23'], ['Lunedì'...'Domenica']], data=matrice 7x24 con conteggi. "
                              "Se dati insuff./impossibile, usa type='text', data='messaggio'. Formato JSON rigoroso.")
    messages = [{"role":"system", "content":system_message_content}, {"role":"user", "content": f"Log:\n{chart_data_log}\n\nRichiesta: {user_chart_prompt}"}]
    chart_image_path = None; status_message = "Errore AI."
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=1500, temperature=0.1)
        if response.choices and response.choices[0].message.content:
            ai_response_text = response.choices[0].message.content.strip(); print(f"Chart params response:\n{ai_response_text}")
            match = re.search(r"\[CHART_PARAMS\]\s*(\{.*?\})\s*\[/CHART_PARAMS\]", ai_response_text, re.DOTALL | re.IGNORECASE)
            if match:
                json_string = match.group(1).strip()
                try:
                    params = json.loads(json_string); print(f"Parsed chart params: {params}")
                    if params.get("type") == "text": status_message = params.get("data", "Messaggio AI."); chart_image_path = None
                    else:
                        if params.get("type") == "heatmap":
                             data_matrix = params.get("data"); labels_list = params.get("labels")
                             if not (isinstance(data_matrix, list) and all(isinstance(row, list) for row in data_matrix) and isinstance(labels_list, list) and len(labels_list) == 2 and isinstance(labels_list[0], list) and isinstance(labels_list[1], list) and (not data_matrix or (data_matrix and data_matrix[0] and len(data_matrix) == len(labels_list[1]) and len(data_matrix[0]) == len(labels_list[0])) ) ):
                                 status_message = "❌ Formato/Dimensioni heatmap JSON errato."; print(status_message); return None, status_message, None
                        chart_image_path = create_matplotlib_chart(params)
                        if chart_image_path: status_message = f"Grafico '{params.get('title', 'Generato')}' creato."; return chart_image_path, status_message, chart_image_path
                        else: status_message = "❌ Errore Matplotlib."; chart_image_path = None
                except json.JSONDecodeError as e: status_message = f"❌ AI JSON non valido: {e}."; print(f"JSON Decode Error: {e}"); chart_image_path = None
                except Exception as e: status_message = f"❌ Errore creazione grafico: {e}"; print(f"Err creating chart: {e}\n{traceback.format_exc()}"); chart_image_path = None
            else: status_message = "❌ AI non ha restituito [CHART_PARAMS]."; print(f"Markers not found: {ai_response_text}"); chart_image_path = None
        else: print("Warn: OpenAI empty response."); status_message = "❌ AI non ha fornito parametri."; chart_image_path = None
    except Exception as e: status_message = f"❌ Errore API chart params: {e}"; print(f"Error OpenAI chart: {traceback.format_exc()}"); gr.Error(status_message); chart_image_path = None
    return chart_image_path, status_message, None

def download_chart_file(chart_file_path):
    if chart_file_path and os.path.exists(chart_file_path): return chart_file_path
    else: gr.Warning("Grafico non disponibile."); return None

def clear_chat(): return [], "", None
def clear_chart_data(): return ""
def clear_other_fd_inputs(input_type): return (None, None, "", {}, "")

# --- Habitat Map Functions ---
def create_map_html(locations): # locations è lista di tuple (nome, lat, lon)
    if not locations: m = folium.Map(location=[46,10],zoom_start=4,tiles='CartoDB positron'); folium.Html('<h3>Nessun dato habitat.</h3>',script=True).add_to(m); return m._repr_html_()
    lats=[float(loc[1]) for loc in locations if loc[1] is not None]; lons=[float(loc[2]) for loc in locations if loc[2] is not None]
    if not lats or not lons: m = folium.Map(location=[46,10],zoom_start=4); folium.Html('<h3 style="color:orange;">AI no coords.</h3>',script=True).add_to(m); return m._repr_html_()
    m = folium.Map(location=[sum(lats)/len(lats), sum(lons)/len(lons)], zoom_start=5, tiles='CartoDB positron')
    mc_ai = MarkerCluster(name="Habitat Stimati (AI)").add_to(m) # Solo cluster AI per ora
    for name, lat, lon in locations: # Non c'è 'source' qui
        if lat is not None and lon is not None:
            try: folium.Marker([lat, lon], popup=f"<b>{name}</b><br>Habitat stimato (AI)", tooltip=name, icon=folium.Icon(color='blue', icon='info-sign')).add_to(mc_ai)
            except Exception as e: print(f"Err marker {name}: {e}")
    folium.LayerControl(collapsed=False).add_to(m); return m._repr_html_()

# --- generate_habitat_map (Utilizza solo stima AI per ora, basata su log senza GPS) ---
def generate_habitat_map(fish_log_data):
    locations_on_map = [] # Sarà una lista di tuple (nome, lat, lon)
    species_to_ask_ai = set()
    print("Parsing fish log for species (no GPS extraction in this version)...")
    if fish_log_data and fish_log_data.strip():
        lines = fish_log_data.strip().split('\n')
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 2: # Formato: Timestamp,Specie,Prob
                species_name = parts[1].strip().title()
                if species_name: species_to_ask_ai.add(species_name)

    if not species_to_ask_ai: return create_map_html([]), "ℹ️ Nessuna specie valida nel log per stima AI."

    species_list_str = "\n".join(sorted(list(species_to_ask_ai)))
    status_message = ""; ai_locations_found = 0
    print(f"Requesting AI habitat estimation for: {species_list_str}")
    if not initialize_chatbot_client(): return create_map_html([]), f"❌ API Error: {openai_api_error}"
    map_prompt = f"Specie:\n{species_list_str}\nPer ogni specie, fornisci UNA posizione stimata habitat primario (lat, lon). Rispondi SOLO 'Nome Specie, Latitudine, Longitudine', una per riga."
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": "Fornisci 'Nome, Lat, Lon' per habitat."}, {"role": "user", "content": map_prompt}], temperature=0.3, max_tokens=1024)
        ai_raw_response = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
        print(f"AI map response: {ai_raw_response if ai_raw_response else 'None'}")
        for line in ai_raw_response.split('\n'):
            if not line.strip(): continue; parts = line.split(',')
            if len(parts) >= 3:
                name_ai = parts[0].strip().title()
                if name_ai in species_to_ask_ai:
                    try:
                        lat_str = re.sub(r"[^0-9.-]", "", parts[1].strip()); lon_str = re.sub(r"[^0-9.-]", "", parts[2].strip())
                        if not lat_str or not lon_str: continue
                        lat = float(lat_str); lon = float(lon_str)
                        if -90 <= lat <= 90 and -180 <= lon <= 180: locations_on_map.append((name_ai, lat, lon)); ai_locations_found +=1
                    except ValueError: print(f"Warn: Cannot parse AI coords: '{line}'")
                    except Exception as e_parse: print(f"Warn: Error parsing AI line '{line}': {e_parse}")
        status_message = f"Mappa generata con {ai_locations_found} habitat stimati da AI." if ai_locations_found > 0 else "❌ AI non ha fornito coordinate valide."
    except openai.APIError as e_api: print(f"Errore API OpenAI per mappa: {e_api}"); status_message = f"❌ Errore API OpenAI: {e_api}"
    except Exception as e_general: print(f"Errore gen. mappa: {e_general}"); status_message = f"❌ Errore generazione mappa: {e_general}"
    return create_map_html(locations_on_map), status_message # locations_on_map è una lista di tuple

def map_chat_response(message_input, chat_history):
    if not initialize_chatbot_client(): error_msg=f"API error."; chat_history.append({'role':'user','content':message_input}); chat_history.append({'role':'assistant','content':f"❌ {error_msg}"}); return chat_history, ""
    chat_history.append({"role": "user", "content": message_input})
    api_messages = [{"role": "system", "content": "Sei Ocean AI. Stai aiutando un utente con una mappa di habitat ittici stimati. Rispondi in italiano."}]
    api_messages.extend(chat_history[max(0, len(chat_history)-10):])
    ai_response_text = "Errore AI."
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=api_messages, max_tokens=1000, temperature=0.7)
        ai_response_text = response.choices[0].message.content.strip() if response.choices else "Nessuna risposta."
    except Exception as e: ai_response_text = f"❌ Errore API: {e}"
    chat_history.append({"role": "assistant", "content": ai_response_text})
    return chat_history, ""

def inserisci_grafico_ai(prompt_grafico, chart_data_log):
    if not initialize_chatbot_client(): error_msg=f"API error: {openai_api_error}"; gr.Error(error_msg); return None, error_msg, None
    if not prompt_grafico: return None, "⚠️ Inserisci descrizione.", None
    if not chart_data_log: return None, "⚠️ Log dati vuoto.", None
    chart_path, status, _ = ai_parse_and_generate_chart(prompt_grafico, chart_data_log)
    if chart_path and os.path.exists(chart_path):
        markdown_link = f"![Grafico: {prompt_grafico.replace(']','').replace('[',' ')}]({chart_path})"
        return chart_path, f"✅ Grafico generato.", markdown_link
    else: return None, f"❌ Errore: {status}", None


def gestisci_immagine_upload(file_obj):
    if file_obj is None: return "⚠️ Nessuna immagine.", None
    if not UPLOADS_DIR: return "❌ Errore: Cartella Uploads non accessibile.", None
    try:
        original_filename = os.path.basename(file_obj.name); safe_filename = re.sub(r'[^\w\-.]', '_', original_filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S"); unique_filename = f"{timestamp}_{safe_filename}"
        save_path = os.path.join(UPLOADS_DIR, unique_filename)
        with open(file_obj.name, "rb") as f_in, open(save_path, "wb") as f_out: f_out.write(f_in.read())
        print(f"Image saved to: {save_path}"); markdown_link = f"![{safe_filename}]({save_path})"
        return f"✅ Immagine salvata.", markdown_link
    except Exception as e: print(f"Err saving img: {e}"); return f"❌ Errore: {e}", None

def esporta_report(report_md, formato):
    if not report_md: gr.Warning("Report vuoto."); return None
    print(f"Exporting report to: {formato}"); html_content = markdown2.markdown(report_md, extras=["tables", "fenced-code-blocks", "strike", "markdown-in-html", "code-friendly"])
    output_path = None; styled_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Report</title><style>body{{font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto;}}table{{border-collapse: collapse; width: 100%; margin-bottom: 1em;}}th,td{{border: 1px solid #ddd; padding: 8px; text-align: left;}}th{{background-color: #f2f2f2;}}img{{max-width: 90%; height: auto; display: block; margin: 1em auto; border: 1px solid #eee;}}pre{{background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd; border-radius: 5px; overflow-x: auto;}}code{{font-family: monospace;}}h1,h2,h3{{border-bottom: 1px solid #eee; padding-bottom: 0.3em;}}</style></head><body>{html_content}</body></html>"""
    try:
        if formato == "HTML":
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding='utf-8', prefix="report_") as f: f.write(styled_html); output_path = f.name
        elif formato == "PDF":
            if pdfkit is None: gr.Error("Install 'pdfkit' and 'wkhtmltopdf' for PDF export."); return None
            options = {'encoding': "UTF-8", 'enable-local-file-access': None, 'quiet': ''}
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="report_") as f: pdfkit.from_string(styled_html, f.name, options=options); output_path = f.name
        elif formato == "Word":
            html_input_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding='utf-8', prefix="temp_html_") as f_html: f_html.write(styled_html); html_input_path = f_html.name
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False, prefix="report_") as f_docx: output_path = f_docx.name
                command = ["pandoc", html_input_path, "-f", "html", "-t", "docx", "-o", output_path]
                print(f"Running Pandoc: {' '.join(command)}");
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                print(f"Pandoc DOCX created: {output_path}")
            except FileNotFoundError:
                 gr.Error("Install 'pandoc' for Word export."); return None
            except subprocess.CalledProcessError as e:
                 gr.Error(f"Pandoc Error: {e.stderr or 'Unknown Pandoc error'}")
                 print(f"Pandoc STDERR: {e.stderr}\nPandoc STDOUT: {e.stdout}")
                 if output_path and os.path.exists(output_path): os.remove(output_path)
                 return None
            except Exception as e_pandoc:
                 gr.Error(f"Word Export Error: {e_pandoc}")
                 if output_path and os.path.exists(output_path): os.remove(output_path)
                 return None
            finally:
                if html_input_path and os.path.exists(html_input_path): os.remove(html_input_path)
        else: gr.Warning(f"Formato '{formato}' non supportato."); return None
        print(f"Export successful: {output_path}"); return output_path
    except Exception as e: print(f"Error exporting: {e}\n{traceback.format_exc()}"); gr.Error(f"Errore esportazione: {e}"); return None


# --- Gradio Interface Definition ---
initial_fish_model_status = load_fish_detection_model()
initial_chatbot_status = initialize_chatbot_client()

theme = gr.themes.Soft(primary_hue="sky", secondary_hue="cyan").set(
    body_background_fill="linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%)",
    body_text_color="#f8fafc", block_background_fill="rgba(255, 255, 255, 0.07)",
    block_border_width="1px", block_border_color="rgba(255, 255, 255, 0.1)",
    block_label_text_color="#cbd5e1", block_title_text_color="#e2e8f0",
    input_background_fill="rgba(255, 255, 255, 0.05)", input_border_color="rgba(255, 255, 255, 0.2)",
    button_primary_background_fill="linear-gradient(135deg, #2563eb, #3b82f6)", button_primary_text_color="white",
    button_secondary_background_fill="rgba(255, 255, 255, 0.1)", button_secondary_text_color="#e2e8f0",
    border_color_accent="rgba(59, 130, 246, 0.5)", background_fill_secondary="#1e293b",
    shadow_drop="rgba(0, 0, 0, 0.2) 0px 4px 12px"
)
css = """footer{display:none !important}.gr-label>.label-wrap>span{color:#cbd5e1 !important}#results_label .confidences{background-color:rgba(0,0,0,.2)!important;border-radius:5px;padding:5px}#results_label .confidence-set .primary{background:linear-gradient(to right,#3b82f6,#60a5fa);background-color:unset!important;border-radius:5px}#main_title_block{text-align:center;padding-bottom:20px;margin-bottom:20px;border-bottom:1px solid rgba(255,255,255,.1)}#main_title_block h1{color:#e0f2fe;font-size:2.5em;font-weight:700;display:flex;align-items:center;justify-content:center;gap:15px;margin-bottom:10px}#main_title_block p{color:#cbd5e1;font-size:1.1em;max-width:700px;margin:auto}@import url(https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css);#chatbot_display,#map_chatbot_display{height:500px;overflow-y:auto}#map_display iframe{width:100%;height:500px;border-radius:8px;border:1px solid rgba(255,255,255,.1)}#map_status_label,#chart_status_label{margin-top:10px;font-style:italic;color:#cbd5e1;min-height:1.5em}#chatbot_tab .gr-panel,#report_editor_tab .gr-panel,#charting_tab .gr-panel,#fish_detector_tab .gr-panel,#map_tab .gr-panel{padding:15px}#chat_input_area_row>div{display:flex;align-items:stretch;padding:0!important;height:auto}#chat_input_area_row{padding:0 10px 10px}#chat_message_input textarea,#map_chat_input textarea{width:100%;height:40px;box-sizing:border-box}#chatbot_input_image{height:40px;width:60px!important;min-width:60px!important;max-width:60px!important}#chatbot_input_image .upload_btn{padding:0!important;min-width:unset!important;flex-grow:0!important;height:100%;background:rgba(255,255,255,.1);border-radius:8px;border:none;cursor:pointer;display:flex!important;align-items:center;justify-content:center}#chatbot_input_image label{height:100%;width:100%;display:flex;align-items:center;justify-content:center;cursor:pointer;padding:0 5px}#chatbot_input_image img{max-width:100%;max-height:100%;object-fit:contain;border-radius:4px}#chatbot_input_image input[type=file]{display:none}#chat_send_button,#chat_clear_button,#chat_clear_image_button,#map_chat_send_button{height:40px}#chat_send_button,#map_chat_send_button{flex-grow:0;min-width:100px}#chatbot_display .message-container[data-testid=user] .message-body,#map_chatbot_display .message-container[data-testid=user] .message-body{background-color:rgba(59,130,246,.15)!important;border-top-left-radius:var(--radius-lg)!important;border-top-right-radius:var(--radius-xl)!important;border-bottom-left-radius:var(--radius-lg)!important;border-bottom-right-radius:0!important;color:#fff}#chatbot_display .message-container[data-testid=bot] .message-body,#map_chatbot_display .message-container[data-testid=bot] .message-body{background-color:rgba(100,116,139,.15)!important;border-top-left-radius:var(--radius-xl)!important;border-top-right-radius:var(--radius-lg)!important;border-bottom-left-radius:0!important;border-bottom-right-radius:var(--radius-lg)!important;color:#fff}#chatbot_display .message-container[data-testid=user] .message-body img,#map_chatbot_display .message-container[data-testid=user] .message-body img{background-color:rgba(59,130,246,.15)!important;border-radius:var(--radius-lg);padding:5px}#chatbot_display .message-container[data-testid=bot] .message-body img,#map_chatbot_display .message-container[data-testid=bot] .message-body img{background-color:rgba(100,116,139,.15)!important;border-radius:var(--radius-lg);padding:5px}.gradio-container .message{max-width:75%!important}#report_editor_textbox textarea,#chart_data_editor textarea{min-height:300px;font-family:monospace;font-size:.9rem;line-height:1.4;resize:vertical}#report_preview>div{background-color:rgba(255,255,255,.07);padding:15px;border-radius:8px;min-height:100px;margin-top:10px;overflow-x:auto}#report_controls .gr-button,#charting_tab .gr-button,#map_tab .gr-button,#report_editor_tab .gr-button{min-width:120px}#chart_display_image img,#report_chart_preview img{background-color:#fff;padding:5px;border-radius:5px;object-fit:contain;max-height:400px}#charting_controls,#report_editor_tab .gr-group{margin-top:15px}#charting_controls>div{flex-wrap:wrap;gap:10px}.gradio-container .hidden{display:none!important}.gr-box.gr-text-input[label="Stato Corrente"] textarea{color:#60a5fa}.gr-box.gr-text-input[label^=❌] textarea{color:#f87171}
"""

with gr.Blocks(theme=theme, css=css, title="Ocean AI: Detector, Chatbot & Report Pro") as demo:
    with gr.Column(elem_id="main_title_block"):
        gr.HTML("<h1><i class='fas fa-fish'></i> Ocean AI <i class='fas fa-robot'></i> <i class='fas fa-chart-line'></i> <i class='fas fa-map-marked-alt'></i> <i class='fas fa-file-alt'></i></h1>")
        gr.Markdown("Rilevamento Pesci, Chatbot, Report Pro, Grafici e Mappa Habitat")
    with gr.Row():
        with gr.Column(): gr.Markdown(f"✅ Fish Detector OK." if initial_fish_model_status else f"❌ ERRORE Fish Detector: {fish_model_loading_error}")
        with gr.Column(): gr.Markdown(f"✅ OpenAI API OK." if initial_chatbot_status else f"❌ ERRORE OpenAI API: {openai_api_error or 'Non disponibile.'}")

    with gr.Tabs() as main_tabs:
        with gr.TabItem("🐟 Fish Detector", id="fish_detector_tab"):
             with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Input Media")
                    with gr.Tabs():
                        with gr.TabItem("🖼️ Immagine"): fd_input_image = gr.Image(type="pil", label="Carica", height=240, sources=["upload", "clipboard"])
                        with gr.TabItem("🎬 Video"): fd_input_video = gr.Video(label="Carica", height=240, sources=["upload"])
                        with gr.TabItem("📡 Stream URL"): fd_input_stream_url = gr.Textbox(label="URL", placeholder="Incolla URL...", lines=1)
                        with gr.TabItem("📷 Webcam Live"):
                            fd_input_webcam = gr.Image(label="Input Webcam", sources=["webcam"], streaming=True, height=240)
                            gr.Markdown("<small><i>L'analisi si avvia automaticamente.</i></small>")
                    gr.Markdown("### 2. Analisi (File/URL)")
                    fd_analyze_button = gr.Button("🔬 Analizza File/URL", variant="primary", interactive=initial_fish_model_status)
                    gr.Markdown("### 3. Stato (File/URL)")
                    fd_status_textbox = gr.Textbox(label="Stato", interactive=False, lines=2)
                with gr.Column(scale=1):
                    gr.Markdown("### Anteprima & Risultati")
                    with gr.Tabs():
                         with gr.TabItem("Risultati File/Stream URL"):
                             gr.Markdown("#### Anteprima"); fd_output_media_preview = gr.Image(label="Anteprima File/URL", height=240, interactive=False)
                             gr.Markdown(f"#### Classificazione"); fd_results_label = gr.Label(label="Classificazione", num_top_classes=PREDICTION_CLASSES_TO_SHOW)
                         with gr.TabItem("Risultati Webcam Live"):
                              gr.Markdown("#### Frame Webcam"); fd_webcam_output_preview = gr.Image(label="Frame Webcam", height=240, interactive=False)
                              gr.Markdown(f"#### Classificazione Live"); fd_webcam_results_label = gr.Label(label="Class. Live", num_top_classes=PREDICTION_CLASSES_TO_SHOW)

        with gr.TabItem("💬 Chatbot", id="chatbot_tab"):
             with gr.Column(variant="panel"):
                chatbot_display = gr.Chatbot([], type='messages', elem_id="chatbot_display", label="Conversazione", layout="bubble", avatar_images=(None, "https://img.icons8.com/color/48/bot.png"), height=550)
                with gr.Row(elem_id="chat_input_area_row", equal_height=True):
                    with gr.Column(scale=4): chat_message_input = gr.Textbox(show_label=False, placeholder="Ciao! Chiedi o carica immagine...", lines=1, elem_id="chat_message_input")
                    with gr.Column(min_width=50, scale=1): chat_image_input = gr.Image(type="pil", label="Img", sources=["upload", "clipboard"], interactive=True, height=40, width=60, show_label=False, elem_id="chatbot_input_image")
                    with gr.Column(min_width=100, scale=1): chat_send_button = gr.Button("✉️ Invia", variant="primary", interactive=initial_chatbot_status, elem_id="chat_send_button")
                with gr.Row(): chat_clear_button = gr.Button("🗑️ Cancella Chat", variant="secondary"); chat_clear_image_button = gr.Button("❌ Cancella Img", variant="secondary")

        with gr.TabItem("📝 Report Editor Pro", id="report_editor_tab"):
            with gr.Row(variant="panel"):
                with gr.Column(scale=2):
                    gr.Markdown("### Editor Relazioni (Markdown)")
                    report_editor = gr.Textbox(lines=25, label="Contenuto Relazione", placeholder="## Titolo...\nScrivi qui o usa gli strumenti...", interactive=True, elem_id="report_editor_textbox", show_copy_button=True)
                    with gr.Accordion("🛠️ Strumenti Editor", open=True):
                        with gr.Group():
                             gr.Markdown("#### 📊 Inserisci Grafico AI")
                             report_chart_prompt = gr.Textbox(label="Descrivi il grafico:", placeholder="Es: 'heatmap giorno/ora', 'barre conteggio specie'", lines=1, interactive=initial_chatbot_status)
                             report_insert_chart_btn = gr.Button("Genera Link Grafico", icon="📈", interactive=initial_chatbot_status)
                             report_chart_preview = gr.Image(label="Anteprima Grafico", type="filepath", interactive=False, show_download_button=False, height=200)
                             report_chart_status = gr.Textbox(label="Stato Grafico:", value="", interactive=False, lines=1)
                             report_chart_link_output = gr.Textbox(label="Link MD (copia nel report):", interactive=True, show_copy_button=True)
                        with gr.Group():
                             gr.Markdown("#### 🖼️ Aggiungi Immagine")
                             report_image_upload = gr.File(label="Trascina Immagine Qui o Clicca", type="filepath", file_types=["image"], interactive=UPLOADS_DIR is not None)
                             report_image_status = gr.Textbox(label="Stato Immagine:", value="", interactive=False, lines=1)
                             report_image_link_output = gr.Textbox(label="Link MD (copia nel report):", interactive=True, show_copy_button=True)
                             if not UPLOADS_DIR: gr.Markdown("⚠️ *Upload immagini disabilitato*")
                        with gr.Group():
                            gr.Markdown("#### 📤 Esporta Report")
                            report_export_format = gr.Dropdown(["HTML", "PDF", "Word"], label="Scegli Formato", value="HTML")
                            report_export_button = gr.Button("Esporta", icon="📄")
                            report_export_file = gr.File(label="Scarica Report Esportato", interactive=False)
                    with gr.Accordion("🤖 Assistenza AI Testo", open=False):
                        report_ai_prompt = gr.Textbox(label="Istruzioni AI per il testo:", placeholder="Es: 'Riassumi...', 'Correggi'...", lines=2, interactive=initial_chatbot_status)
                        report_ai_assist_button = gr.Button("💡 Chiedi Assistenza Testo", interactive=initial_chatbot_status)
                    with gr.Accordion("📧 Invia via Email Client", open=False):
                         email_recipient = gr.Textbox(label="A:", placeholder="dest@example.com"); email_subject = gr.Textbox(label="Oggetto:", value="Report Ocean AI")
                         send_email_button = gr.Button("✉️ Apri Email", variant="primary"); email_mailto_link = gr.HTML()
                with gr.Column(scale=1):
                    gr.Markdown("### Anteprima Relazione"); report_preview = gr.Markdown(elem_id="report_preview", height=600)

        with gr.TabItem("📊 Grafici Standalone", id="charting_tab"):
             with gr.Column(variant="panel"):
                gr.Markdown("### Creazione Grafici Assistita da AI"); gr.Markdown("Genera grafici dai dati del log per visualizzazione o download separato.")
                gr.Markdown("#### Log Dati Rilevamenti"); gr.Markdown("<small><i>Formato: <code>AAAA-MM-GG HH:MM:SS,Specie,Prob%[,Lat,Lon]</code></i></small>")
                chart_data_editor = gr.Textbox(label="Log Dati", lines=10, interactive=True, placeholder="...", elem_id="chart_data_editor", show_copy_button=True)
                chart_ai_prompt = gr.Textbox(label="Richiesta Grafico AI:", placeholder="Es: 'heatmap giorno/ora'", lines=2, interactive=initial_chatbot_status)
                chart_ai_button = gr.Button("📈 Genera Grafico", variant="primary", interactive=initial_chatbot_status)
                gr.Markdown("#### Grafico Generato"); chart_display_image = gr.Image(label="Grafico", elem_id="chart_display_image", type="filepath", interactive=False, show_download_button=False)
                chart_status_label = gr.Label(label="Stato Generazione", value="")
                chart_download_path = gr.Textbox(visible=False); chart_download_file = gr.File(label="DL", visible=False)
                with gr.Row(elem_id="charting_controls"): chart_download_button = gr.Button("💾 Scarica Grafico"); chart_clear_data_button = gr.Button("🗑️ Cancella Log")

        with gr.TabItem("🗺️ Mappa Habitat & AI", id="map_tab"):
             with gr.Column(variant="panel"):
                gr.Markdown("### Mappa Habitat Stimati & Chat AI"); gr.Markdown("Visualizza posizioni stimate habitat e chiedi info all'AI.")
                with gr.Row(): map_generate_button = gr.Button("📍 Posiziona Specie su Mappa", variant="primary", interactive=initial_chatbot_status)
                with gr.Row(): map_display = gr.HTML(label="Mappa Habitat Stimato", elem_id="map_display")
                map_status_label = gr.Label(label="Stato Mappa", value="Mappa non generata.")
                gr.Markdown("---"); gr.Markdown("### Chat sulla Mappa")
                map_chatbot_display = gr.Chatbot([], type='messages', elem_id="map_chatbot_display", label="Conversazione Mappa", layout="bubble", avatar_images=(None, "https://img.icons8.com/color/48/bot.png"), height=450)
                with gr.Row(equal_height=True): map_chat_input = gr.Textbox(show_label=False, placeholder="Chiedi sulla mappa...", lines=1, scale=4); map_chat_send_button = gr.Button("✉️ Invia", variant="primary", scale=1, interactive=initial_chatbot_status)

    # --- Interactions ---
    fd_analyze_button.click(analyze_media, [fd_input_image, fd_input_video, fd_input_stream_url, report_editor, chart_data_editor], [fd_output_media_preview, fd_results_label, fd_status_textbox, report_editor, chart_data_editor], queue=True)
    fd_input_webcam.change(analyze_webcam_frame, fd_input_webcam, [fd_webcam_output_preview, fd_webcam_results_label], queue=True)
    fd_input_image.change(lambda: clear_other_fd_inputs('image'), None, [fd_input_video, fd_input_stream_url, fd_output_media_preview, fd_results_label, fd_status_textbox], queue=False)
    fd_input_video.change(lambda: clear_other_fd_inputs('video'), None, [fd_input_image, fd_input_stream_url, fd_output_media_preview, fd_results_label, fd_status_textbox], queue=False)
    fd_input_stream_url.change(lambda: clear_other_fd_inputs('stream'), None, [fd_input_image, fd_input_video, fd_output_media_preview, fd_results_label, fd_status_textbox], queue=False)

    clear_chat_outputs = [chat_message_input, chat_image_input]
    chat_inputs = [chat_message_input, chatbot_display, chat_image_input]
    chat_send_button.click(chatbot_response, chat_inputs, [chatbot_display] + clear_chat_outputs, queue=True)
    chat_message_input.submit(chatbot_response, chat_inputs, [chatbot_display] + clear_chat_outputs, queue=True)
    chat_clear_button.click(clear_chat, None, [chatbot_display, chat_message_input, chat_image_input], queue=False)
    chat_clear_image_button.click(lambda: None, None, [chat_image_input], queue=False)

    report_editor.change(lambda x: x, report_editor, report_preview, queue=False)
    report_ai_assist_button.click(ai_interact_report, [report_editor, report_ai_prompt], report_editor, queue=True)
    send_email_button.click(send_report_email_mailto, [email_recipient, email_subject, report_editor], email_mailto_link, queue=False)
    report_insert_chart_btn.click(fn=ai_parse_and_generate_chart, inputs=[report_chart_prompt, chart_data_editor], outputs=[report_chart_preview, report_chart_status, report_chart_link_output], queue=True)
    report_image_upload.upload(fn=gestisci_immagine_upload, inputs=[report_image_upload], outputs=[report_image_status, report_image_link_output], queue=True)
    report_export_button.click(fn=esporta_report, inputs=[report_editor, report_export_format], outputs=[report_export_file], queue=True)

    chart_ai_button.click(fn=ai_parse_and_generate_chart, inputs=[chart_ai_prompt, chart_data_editor], outputs=[chart_display_image, chart_status_label, chart_download_path], queue=True)
    chart_download_button.click(fn=download_chart_file, inputs=[chart_download_path], outputs=[chart_download_file], queue=False)
    chart_clear_data_button.click(clear_chart_data, None, [chart_data_editor], queue=False)

    map_generate_button.click(fn=generate_habitat_map, inputs=[chart_data_editor], outputs=[map_display, map_status_label], queue=True)
    map_chat_send_button.click(fn=map_chat_response, inputs=[map_chat_input, map_chatbot_display], outputs=[map_chatbot_display, map_chat_input], queue=True)
    map_chat_input.submit(fn=map_chat_response, inputs=[map_chat_input, map_chatbot_display], outputs=[map_chatbot_display, map_chat_input], queue=True)

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Avvio App Gradio 'Ocean AI Pro'...")
    if not initial_fish_model_status: print(f"❌ ERRORE Modello Pesci: {fish_model_loading_error}")
    else: print("✅ Modello Rilevamento Pesci caricato.")
    if not initial_chatbot_status: print(f"❌ ERRORE Chatbot (OpenAI): {openai_api_error or 'Non disponibile.'}")
    if not OPENAI_API_KEY: print("👉 ATTENZIONE: Chiave API OpenAI non configurata.")
    else: print("✅ OpenAI API client pronto.")
    if pdfkit is None: print("⚠️ ATTENZIONE: Esportazione PDF disabilitata (installare pdfkit e wkhtmltopdf).")
    try: subprocess.run(["pandoc", "--version"],capture_output=True,check=True, text=True, timeout=2); print("✅ Pandoc trovato.")
    except: print("⚠️ ATTENZIONE: Esportazione Word disabilitata (installare pandoc).")
    # Note: UPLOADS_DIR creation checked during setup, warning already printed if failed
    print("\nInformazioni:"); print(" - Dipendenze: gradio tensorflow opencv-python openai matplotlib folium yt-dlp python-dotenv certifi markdown2 pdfkit Pillow") # Pillow implicita, pillow-heif per HEIC
    print(" - Per PDF/Word Export: installare wkhtmltopdf e/o pandoc."); print("=" * 70 + "\n")
    demo.launch(share=False, debug=True)
