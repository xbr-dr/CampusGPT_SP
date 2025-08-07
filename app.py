import streamlit as st
import os
import pickle
import faiss
import nltk
import re
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from difflib import get_close_matches
import folium
from streamlit_folium import st_folium
from groq import Groq
import shutil
from langdetect import detect, LangDetectException
import tempfile
from openpyxl import load_workbook

nltk.download('punkt')

# --------------- Configuration ---------------
DOCUMENTS_DIR = "data/documents"
STORAGE_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.faiss")
CORPUS_PATH = os.path.join(STORAGE_DIR, "corpus.pkl")
LOCATION_DATA_PATH = os.path.join(STORAGE_DIR, "locations.pkl")
ADMIN_PASSWORD = "1234"  # In production, use environment variables

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# --------------- Initialization & Caching ---------------
@st.cache_resource
def load_models_and_groq():
    """Load sentence transformer model and initialize Groq client."""
    try:
        embed_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.warning("⚠️ Groq API key not found. Please add it to your Streamlit secrets.", icon="🔒")
            return embed_model, None
        groq_client = Groq(api_key=api_key)
        return embed_model, groq_client
    except Exception as e:
        st.error(f"❌ Error loading models or initializing Groq: {e}")
        return None, None

embed_model, client = load_models_and_groq()

@st.cache_resource
def load_nltk_data():
    """Download and verify NLTK 'punkt' data for sentence tokenization."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            nltk.data.find('tokenizers/punkt')
        except Exception as e:
            st.error(f"❌ Failed to download NLTK 'punkt' data: {e}")
            return False
    return True

nltk_loaded = load_nltk_data()

# --------------- Data Processing Functions ---------------
def process_uploaded_files(uploaded_files):
    file_data, locations_from_csv = [], {}
    for uploaded_file in uploaded_files:
        try:
            file_name = uploaded_file.name
            file_path = os.path.join(DOCUMENTS_DIR, file_name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            text = ""
            if file_path.lower().endswith('.pdf'):
                reader = PdfReader(file_path)
                text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
            elif file_path.lower().endswith('.txt'):
                text = uploaded_file.read().decode("utf-8")
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                if file_path.lower().endswith('.xlsx'):
                    df = pd.read_excel(tmp_path, engine='openpyxl')
                else:
                    df = pd.read_excel(tmp_path)
                
                df.columns = [col.lower().strip() for col in df.columns]
                name_col = next((c for c in ['name', 'location', 'place', 'department', 'building'] if c in df.columns), None)
                lat_col = next((c for c in ['lat', 'latitude', 'y'] if c in df.columns), None)
                lon_col = next((c for c in ['lon', 'longitude', 'x'] if c in df.columns), None)
                desc_col = next((c for c in ['description', 'desc', 'details', 'floor', 'level'] if c in df.columns), 'name')
                
                if name_col and lat_col and lon_col:
                    for _, row in df.iterrows():
                        try:
                            name = str(row[name_col]).strip().lower()
                            lat, lon = float(row[lat_col]), float(row[lon_col])
                            desc_parts = []
                            for col in ['building', 'floor', 'department', 'description']:
                                if col in df.columns and pd.notna(row[col]):
                                    desc_parts.append(f"{col.title()}: {row[col]}")
                            desc = " | ".join(desc_parts) if desc_parts else f"Location: {name}"
                            
                            if name and -90 <= lat <= 90 and -180 <= lon <= 180:
                                locations_from_csv[name] = {
                                    'name': name, 
                                    'lat': lat, 
                                    'lon': lon, 
                                    'desc': desc,
                                    'original_name': str(row[name_col]).strip()
                                }
                        except (ValueError, TypeError): 
                            continue
                
                os.unlink(tmp_path)
                text = df.to_string()
            
            if text: 
                file_data.append({'text': text, 'source': file_name})
            
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
                df.columns = [col.lower().strip() for col in df.columns]
                name_col = next((c for c in ['name', 'location', 'place', 'department', 'building'] if c in df.columns), None)
                lat_col = next((c for c in ['lat', 'latitude', 'y'] if c in df.columns), None)
                lon_col = next((c for c in ['lon', 'longitude', 'x'] if c in df.columns), None)
                desc_col = next((c for c in ['description', 'desc', 'details', 'floor', 'level'] if c in df.columns), 'name')
                
                if name_col and lat_col and lon_col:
                    for _, row in df.iterrows():
                        try:
                            name = str(row[name_col]).strip().lower()
                            lat, lon = float(row[lat_col]), float(row[lon_col])
                            desc_parts = []
                            for col in ['building', 'floor', 'department', 'description']:
                                if col in df.columns and pd.notna(row[col]):
                                    desc_parts.append(f"{col.title()}: {row[col]}")
                            desc = " | ".join(desc_parts) if desc_parts else f"Location: {name}"
                            
                            if name and -90 <= lat <= 90 and -180 <= lon <= 180:
                                locations_from_csv[name] = {
                                    'name': name, 
                                    'lat': lat, 
                                    'lon': lon, 
                                    'desc': desc,
                                    'original_name': str(row[name_col]).strip()
                                }
                        except (ValueError, TypeError): 
                            continue
        except Exception as e: 
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
    return file_data, locations_from_csv

def extract_sentences(text_data):
    all_sentences = []
    for data in text_data:
        text, source = data['text'], data['source']
        if not text: continue
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        sentences = nltk.sent_tokenize(text) if nltk_loaded else re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 25:
                s_clean = re.sub(r'\s+', ' ', s_clean)
                s_clean = s_clean.replace('\n', ' ')
                all_sentences.append({'sentence': s_clean, 'source': source})
    return all_sentences

def extract_locations_from_text(text):
    patterns = [
        r'([\w\s]{3,50}?)\s*-\s*Lat:\s*([-+]?\d{1,3}\.?\d+),?\s*Lon:\s*([-+]?\d{1,3}\.?\d+)',
        r'([\w\s]{3,50}?)\s+Latitude:\s*([-+]?\d{1,3}\.?\d+),?\s*Longitude:\s*([-+]?\d{1,3}\.?\d+)',
        r'([\w\s]{3,50}?)\s*\(\s*([-+]?\d{1,3}\.?\d+),\s*([-+]?\d{1,3}\.?\d+)\s*\)'
    ]
    locations = {}
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                name, lat, lon = match.groups()
                name = name.strip().lower()
                if name not in locations:
                    locations[name] = {
                        'name': name, 
                        'lat': float(lat), 
                        'lon': float(lon), 
                        'desc': f"Found in document at coordinates {lat}, {lon}.",
                        'original_name': name
                    }
            except (ValueError, IndexError): 
                continue
    return locations

def build_and_save_data(corpus, locations):
    saved_sentences, saved_locations = 0, 0
    try:
        if corpus and embed_model:
            unique_sentences = []
            seen_sentences = set()
            for item in corpus:
                if item['sentence'] not in seen_sentences:
                    seen_sentences.add(item['sentence'])
                    unique_sentences.append(item)
            
            embeddings = embed_model.encode([item['sentence'] for item in unique_sentences], show_progress_bar=True)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings, dtype="float32"))
            faiss.write_index(index, FAISS_INDEX_PATH)
            
            with open(CORPUS_PATH, "wb") as f: 
                pickle.dump(unique_sentences, f)
            saved_sentences = len(unique_sentences)
        else:
            if os.path.exists(FAISS_INDEX_PATH): 
                os.remove(FAISS_INDEX_PATH)
            if os.path.exists(CORPUS_PATH): 
                os.remove(CORPUS_PATH)
        
        if locations:
            unique_locations = {}
            for name, loc in locations.items():
                unique_locations[name] = loc
            
            with open(LOCATION_DATA_PATH, "wb") as f: 
                pickle.dump(unique_locations, f)
            saved_locations = len(unique_locations)
        else:
            if os.path.exists(LOCATION_DATA_PATH): 
                os.remove(LOCATION_DATA_PATH)
        
        return True, saved_sentences, saved_locations
    except Exception as e:
        st.error(f"❌ Error building/saving data: {e}")
        return False, 0, 0

def load_system_data():
    index, corpus, location_map = None, [], {}
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CORPUS_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CORPUS_PATH, "rb") as f: 
                corpus = pickle.load(f)
        
        if os.path.exists(LOCATION_DATA_PATH):
            with open(LOCATION_DATA_PATH, "rb") as f: 
                location_map = pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error loading system data: {e}")
        return None, [], {}
    
    return index, corpus, location_map

# --------------- RAG & Chat Functions ---------------
def retrieve_chunks(query, corpus, index, top_k=5):
    if not all([query, corpus, index, embed_model]): 
        return []
    
    try:
        query_embedding = embed_model.encode([query])
        _, I = index.search(np.array(query_embedding, dtype="float32"), top_k)
        return [corpus[i] for i in I[0] if i < len(corpus)]
    except Exception as e:
        st.warning(f"⚠️ Retrieval error: {e}")
        return []

def match_locations(query, location_map):
    if not location_map: 
        return []
    
    query_lower = query.lower()
    found = []
    
    # First try exact matches
    for name, loc in location_map.items():
        if name in query_lower:
            found.append(loc)
    
    # Then try partial matches with similarity threshold
    if not found:
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in query_words:
            matches = get_close_matches(word, list(location_map.keys()), n=3, cutoff=0.6)
            if matches:
                for match in matches:
                    if match not in [f['name'] for f in found]:
                        found.append(location_map[match])
    
    # If still not found, try to find the most similar name
    if not found and len(query_lower) > 3:
        all_names = list(location_map.keys())
        matches = get_close_matches(query_lower, all_names, n=1, cutoff=0.5)
        if matches:
            found.append(location_map[matches[0]])
    
    # Remove duplicates
    unique_found = {}
    for loc in found:
        unique_found[loc['name']] = loc
    
    return list(unique_found.values())

def compute_distance_info(locations):
    if len(locations) == 2:
        try:
            coord1 = (locations[0]["lat"], locations[0]["lon"])
            coord2 = (locations[1]["lat"], locations[1]["lon"])
            dist = geodesic(coord1, coord2)
            
            if dist.kilometers >= 1:
                return f"The distance between {locations[0]['original_name']} and {locations[1]['original_name']} is approximately {dist.kilometers:.1f} kilometers."
            else:
                return f"The distance between {locations[0]['original_name']} and {locations[1]['original_name']} is approximately {dist.meters:.0f} meters."
        except Exception:
            return ""
    return ""

def ask_chatbot(query, context_chunks, geo_context, distance_info):
    if not client: 
        return "The AI assistant is currently offline. Please try again later."
    
    try:
        lang_code = detect(query)
        lang_map = {'en': 'English', 'ur': 'Urdu', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French'}
        language = lang_map.get(lang_code, 'English')
    except LangDetectException: 
        language = "English"

    # Prepare context from documents
    context = "\n".join([chunk['sentence'] for chunk in context_chunks])
    
    # Enhanced system prompt for precise responses
    system_prompt = f"""
    You are CampusGPT, an efficient university campus assistant. Follow these guidelines strictly:
    
    1. **Greetings**: If greeted, respond with a brief, friendly greeting like "Hello! How can I assist you with campus information today?"
    
    2. **Location Queries**:
       - Always include exact coordinates in format: (Lat: XX.XX, Lon: YY.YY)
       - Provide building, floor, and department if available
       - Example: "The Computer Science department is in Engineering Building, 3rd Floor (Lat: 40.7128, Lon: -74.0060)"
    
    3. **General Responses**:
       - Be concise (1-3 sentences max)
       - Only answer what was asked
       - Never say "according to documents" or similar
       - If unsure, say "I don't have that information"
    
    4. **Formatting**:
       - Use bullet points for lists
       - Bold important names/places
       - Use the detected language: {language}
    
    Current Context:
    {geo_context if geo_context else 'No specific locations identified'}
    {distance_info if distance_info else ''}
    """
    
    prompt = f"""
    {system_prompt}
    
    === RELEVANT CAMPUS INFORMATION ===
    {context if context else 'No specific information found'}
    
    === USER'S QUESTION ===
    {query}
    
    Provide a concise response:
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=300    # Limit response length
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error. Please try again."

# --------------- UI Components ---------------
def create_map(locations):
    if not locations: 
        return None
    
    try:
        lats = [loc['lat'] for loc in locations]
        lons = [loc['lon'] for loc in locations]
        map_center = [np.mean(lats), np.mean(lons)]
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        max_range = max(lat_range, lon_range)
        
        if max_range < 0.01:
            zoom_start = 16
        elif max_range < 0.1:
            zoom_start = 14
        else:
            zoom_start = 12
        
        m = folium.Map(
            location=map_center, 
            zoom_start=zoom_start, 
            tiles='CartoDB positron'
        )
        
        for loc in locations:
            maps_url = f"https://www.google.com/maps/search/?api=1&query={loc['lat']},{loc['lon']}"
            
            # Simplified popup with only essential info
            popup_html = f"""
            <div style="width: 200px; font-family: Arial;">
                <h4 style="margin: 0 0 5px 0; font-size: 14px;">
                    {loc.get('original_name', loc['name'].title())}
                </h4>
                <p style="margin: 0 0 3px 0; font-size: 12px;">
                    {loc.get('desc', '').split('|')[0].strip()}
                </p>
                <p style="margin: 0 0 8px 0; font-size: 11px; color: #555;">
                    Coordinates: {loc['lat']:.6f}, {loc['lon']:.6f}
                </p>
                <a href="{maps_url}" target="_blank" 
                   style="display: inline-block; padding: 5px 10px; background-color: #3949ab; 
                          color: white; text-decoration: none; border-radius: 3px; font-size: 12px;">
                    Navigate
                </a>
            </div>
            """
            
            folium.Marker(
                [loc['lat'], loc['lon']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=loc.get('original_name', loc['name'].title()),
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Map creation failed: {e}")
        return None

def display_welcome_message():
    st.markdown("""
    <div style="background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
        <h2 style="color: #2c3e50; margin-top: 0;">🏫 Campus Assistant</h2>
        <p style="color: #34495e;">
            Ask about campus locations, services, or facilities. I can show you exact locations on a map.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------- Admin Page ---------------
def admin_page():
    st.title("🔧 Admin Portal")
    
    if not st.session_state.get("authenticated", False):
        st.subheader("Admin Login")
        password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        return
    
    st.success("✅ Admin access granted")
    
    tab1, tab2 = st.tabs(["Upload Documents", "System Status"])
    
    with tab1:
        st.subheader("Upload Campus Data")
        uploaded_files = st.file_uploader(
            "Select files", 
            type=['pdf', 'txt', 'csv', 'xlsx', 'xls'], 
            accept_multiple_files=True
        )
        
        if st.button("Process Files"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    file_data, csv_locs = process_uploaded_files(uploaded_files)
                    full_text = " ".join([d['text'] for d in file_data])
                    text_locs = extract_locations_from_text(full_text)
                    all_locations = {**csv_locs, **text_locs}
                    corpus_sentences = extract_sentences(file_data)
                    
                    success, num_sentences, num_locations = build_and_save_data(corpus_sentences, all_locations)
                    
                    if success:
                        st.success(f"Processed {num_sentences} sentences and {num_locations} locations")
                    else:
                        st.error("Processing failed")
            else:
                st.warning("Please upload files")
    
    with tab2:
        st.subheader("Current System Status")
        index, corpus, location_map = load_system_data()
        
        col1, col2 = st.columns(2)
        col1.metric("Knowledge Items", len(corpus))
        col2.metric("Locations", len(location_map))
        
        if location_map:
            st.subheader("Location Preview")
            loc_name = st.selectbox("Select location", options=list(location_map.keys()))
            loc_data = location_map[loc_name]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {loc_data.get('original_name', loc_name)}")
                st.write(f"**Coordinates:** {loc_data['lat']:.6f}, {loc_data['lon']:.6f}")
                st.write(f"**Details:** {loc_data.get('desc', '')}")
            
            with col2:
                preview_map = folium.Map(location=[loc_data['lat'], loc_data['lon']], zoom_start=16)
                folium.Marker(
                    [loc_data['lat'], loc_data['lon']],
                    tooltip=loc_data.get('original_name', loc_name)
                ).add_to(preview_map)
                st_folium(preview_map, width=300, height=200)

# --------------- User Page ---------------
def user_page():
    st.title("🏫 Campus Assistant")
    
    index, corpus, location_map = load_system_data()
    system_ready = (index is not None and corpus) or bool(location_map)
    
    if not system_ready:
        display_welcome_message()
        st.info("System is not yet configured. Please contact admin.")
        return
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        is_user = msg["role"] == "user"
        if is_user:
            st.markdown(f"""
            <div style="text-align: right; margin: 10px 0; padding: 10px 15px; 
                        background: #e3f2fd; border-radius: 15px 15px 0 15px;
                        display: inline-block; max-width: 80%; float: right; clear: both;">
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: left; margin: 10px 0; padding: 10px 15px; 
                        background: #f5f5f5; border-radius: 15px 15px 15px 0;
                        display: inline-block; max-width: 80%; float: left; clear: both;">
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            if "locations" in msg and msg["locations"]:
                with st.expander("📍 View Map"):
                    map_obj = create_map(msg["locations"])
                    if map_obj:
                        st_folium(map_obj, width=700, height=400)
    
    if prompt := st.chat_input("Ask about campus..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()
    
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.spinner("Searching..."):
            last_prompt = st.session_state.chat_history[-1]["content"]
            
            chunks = retrieve_chunks(last_prompt, corpus, index)
            locs = match_locations(last_prompt, location_map)
            loc_info = "\n".join([f"{l.get('original_name', l['name'].title())}: (Lat: {l['lat']}, Lon: {l['lon']})" for l in locs])
            dist_info = compute_distance_info(locs)
            
            response = ask_chatbot(last_prompt, chunks, loc_info, dist_info)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response, 
                "locations": locs if locs else None
            })
            st.rerun()

# --------------- Main App ---------------
def main():
    st.set_page_config(page_title="CampusGPT", page_icon="🏫", layout="centered")
    
    st.markdown("""
    <style>
        /* Main container */
        .stApp {
            max-width: 900px;
            margin: 0 auto;
        }
        
        /* Chat bubbles */
        [data-testid="stMarkdownContainer"] p {
            margin: 0;
        }
        
        /* Input box */
        [data-testid="stChatInput"] {
            position: fixed;
            bottom: 20px;
            width: calc(100% - 40px);
            max-width: 860px;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #4b6cb7 0%, #182848 100%);
            color: white;
        }
        
        /* Map container */
        .folium-map {
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## Navigation")
        app_mode = st.radio(
            "Select mode",
            ["👤 User Chat", "🔧 Admin Portal"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        if st.button("Clear Chat"):
            if "chat_history" in st.session_state:
                st.session_state.chat_history = []
                st.rerun()
    
    if app_mode == "👤 User Chat":
        user_page()
    else:
        admin_page()

if __name__ == "__main__":
    main()
