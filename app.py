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

nltk.download('punkt_tab')

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
            
            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            text = ""
            if file_path.lower().endswith('.pdf'):
                reader = PdfReader(file_path)
                text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
            elif file_path.lower().endswith('.txt'):
                text = uploaded_file.read().decode("utf-8")
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                # Handle Excel files
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                if file_path.lower().endswith('.xlsx'):
                    df = pd.read_excel(tmp_path, engine='openpyxl')
                else:
                    df = pd.read_excel(tmp_path)
                
                # Process as CSV would be processed
                df.columns = [col.lower().strip() for col in df.columns]
                name_col = next((c for c in ['name', 'location', 'place'] if c in df.columns), None)
                lat_col = next((c for c in ['lat', 'latitude', 'y'] if c in df.columns), None)
                lon_col = next((c for c in ['lon', 'longitude', 'x'] if c in df.columns), None)
                desc_col = next((c for c in ['description', 'desc', 'details'] if c in df.columns), 'name')
                
                if name_col and lat_col and lon_col:
                    for _, row in df.iterrows():
                        try:
                            name, lat, lon = str(row[name_col]).strip().lower(), float(row[lat_col]), float(row[lon_col])
                            desc = str(row.get(desc_col, f"Location: {name}"))
                            if name and -90 <= lat <= 90 and -180 <= lon <= 180:
                                locations_from_csv[name] = {'name': name, 'lat': lat, 'lon': lon, 'desc': desc}
                        except (ValueError, TypeError): continue
                
                os.unlink(tmp_path)
                text = df.to_string()  # Fallback text representation
            
            if text: 
                file_data.append({'text': text, 'source': file_name})
            
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
                df.columns = [col.lower().strip() for col in df.columns]
                name_col = next((c for c in ['name', 'location', 'place'] if c in df.columns), None)
                lat_col = next((c for c in ['lat', 'latitude', 'y'] if c in df.columns), None)
                lon_col = next((c for c in ['lon', 'longitude', 'x'] if c in df.columns), None)
                desc_col = next((c for c in ['description', 'desc', 'details'] if c in df.columns), 'name')
                
                if name_col and lat_col and lon_col:
                    for _, row in df.iterrows():
                        try:
                            name, lat, lon = str(row[name_col]).strip().lower(), float(row[lat_col]), float(row[lon_col])
                            desc = str(row.get(desc_col, f"Location: {name}"))
                            if name and -90 <= lat <= 90 and -180 <= lon <= 180:
                                locations_from_csv[name] = {'name': name, 'lat': lat, 'lon': lon, 'desc': desc}
                        except (ValueError, TypeError): continue
        except Exception as e: 
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
    return file_data, locations_from_csv

def extract_sentences(text_data):
    all_sentences = []
    for data in text_data:
        text, source = data['text'], data['source']
        if not text: continue
        
        # Clean text before processing
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = re.sub(r'\[.*?\]', '', text)  # Remove citations like [1], [2]
        
        sentences = nltk.sent_tokenize(text) if nltk_loaded else re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 25:  # Only include meaningful sentences
                # Additional cleaning
                s_clean = re.sub(r'\s+', ' ', s_clean)  # Remove extra spaces
                s_clean = s_clean.replace('\n', ' ')  # Remove newlines
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
                        'desc': f"Found in document at coordinates {lat}, {lon}."
                    }
            except (ValueError, IndexError): 
                continue
    return locations

def build_and_save_data(corpus, locations):
    saved_sentences, saved_locations = 0, 0
    try:
        if corpus and embed_model:
            # Filter out very similar sentences to reduce redundancy
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
            # Deduplicate locations
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
    
    # Then try partial matches
    if not found:
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in query_words:
            matches = get_close_matches(word, list(location_map.keys()), n=1, cutoff=0.7)
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
                return f"The distance between {locations[0]['name'].title()} and {locations[1]['name'].title()} is approximately {dist.kilometers:.1f} kilometers."
            else:
                return f"The distance between {locations[0]['name'].title()} and {locations[1]['name'].title()} is approximately {dist.meters:.0f} meters."
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
    
    # Enhanced system prompt for better responses
    system_prompt = """
    You are CampusGPT, a knowledgeable and friendly campus assistant at a university. Your role is to:
    1. Provide accurate information about campus facilities, departments, and locations
    2. Answer questions concisely and conversationally
    3. When mentioning locations, include coordinates in this format: (Lat: XX.XX, Lon: YY.YY)
    4. Never mention that you're referring to documents - just provide the information naturally
    5. If you don't know something, say so rather than guessing
    6. For location-related questions, provide helpful details like building numbers or landmarks
    
    Important guidelines:
    - Use the context below only if relevant to the question
    - For location questions, always confirm if you've found the correct place
    - Keep answers brief but informative
    - Use bullet points for lists of items
    - Format your response appropriately for the detected language
    """
    
    prompt = f"""
    {system_prompt}
    
    === RELEVANT CAMPUS INFORMATION ===
    {context if context else 'No specific information found in campus documents.'}
    
    === LOCATIONS MENTIONED ===
    {geo_context if geo_context else 'No specific locations identified in the query.'}
    
    === DISTANCE CALCULATIONS ===
    {distance_info if distance_info else 'No distance calculations needed.'}
    
    === USER'S QUESTION ===
    {query}
    
    Please provide a helpful response in {language} based on the above information:
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # Lower temperature for more factual responses
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error processing your request. Please try again later."

# --------------- UI Components ---------------
def create_map(locations):
    if not locations: 
        return None
    
    try:
        # Calculate map center and appropriate zoom level
        lats = [loc['lat'] for loc in locations]
        lons = [loc['lon'] for loc in locations]
        map_center = [np.mean(lats), np.mean(lons)]
        
        # Determine zoom level based on spread of locations
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        max_range = max(lat_range, lon_range)
        
        if max_range < 0.01:
            zoom_start = 16  # Very close locations
        elif max_range < 0.1:
            zoom_start = 14  # Campus-scale
        else:
            zoom_start = 12  # City-scale
        
        m = folium.Map(
            location=map_center, 
            zoom_start=zoom_start, 
            tiles='CartoDB positron', 
            attr='CampusGPT Map',
            control_scale=True
        )
        
        # Add markers for each location
        for loc in locations:
            maps_url = f"https://www.google.com/maps/search/?api=1&query={loc['lat']},{loc['lon']}"
            
            popup_html = f"""
            <div style="width: 250px; font-family: Arial, sans-serif;">
                <h4 style="margin: 0 0 10px 0; color: #1a237e; font-size: 16px;">
                    {loc['name'].title()}
                </h4>
                <p style="margin: 0 0 8px 0; color: #424242; font-size: 14px;">
                    {loc.get('desc', 'Location details')}
                </p>
                <p style="margin: 0 0 12px 0; color: #616161; font-size: 13px;">
                    Coordinates: {loc['lat']:.6f}, {loc['lon']:.6f}
                </p>
                <a href="{maps_url}" target="_blank" 
                   style="display: inline-block; padding: 8px 12px; background-color: #3949ab; 
                          color: white; text-decoration: none; border-radius: 4px; font-size: 14px;">
                    Open in Google Maps
                </a>
            </div>
            """
            
            folium.Marker(
                [loc['lat'], loc['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=loc['name'].title(),
                icon=folium.Icon(color="blue", icon="university", prefix="fa")
            ).add_to(m)
        
        # Add a circle around each marker for better visibility
        for loc in locations:
            folium.Circle(
                location=[loc['lat'], loc['lon']],
                radius=50,
                color='#3186cc',
                fill=True,
                fill_color='#3186cc',
                fill_opacity=0.2
            ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"🗺️ Map creation failed: {e}")
        return None

def display_welcome_message():
    st.markdown("""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 25px; 
                border-left: 5px solid #4b6cb7; margin-bottom: 30px;">
        <h2 style="color: #2c3e50; margin-top: 0;">🏫 Welcome to CampusGPT</h2>
        <p style="font-size: 16px; color: #34495e;">
            Your intelligent campus assistant is ready to help with information about university facilities, 
            departments, and locations. Ask about anything from building locations to campus services.
        </p>
        <div style="background-color: #e8f4fd; border-radius: 8px; padding: 15px; margin-top: 15px;">
            <h4 style="color: #2c3e50; margin-top: 0;">Try asking:</h4>
            <ul style="margin-bottom: 0;">
                <li>Where is the Computer Science department?</li>
                <li>What are the library opening hours?</li>
                <li>How far is it from the Student Center to the Main Library?</li>
                <li>Tell me about campus parking facilities</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------- Admin Page ---------------
def admin_page():
    st.title("🔧 CampusGPT Admin Portal")
    st.markdown("---")
    
    if not st.session_state.get("authenticated", False):
        st.subheader("🔐 Admin Login")
        password = st.text_input("Enter Admin Password", type="password", key="admin_pass")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        return
    
    st.success("✅ You are logged in as administrator")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "📊 System Status", "⚙️ Advanced Settings"])
    
    with tab1:
        st.subheader("Upload Campus Documents")
        st.info("""
        Upload documents to build the knowledge base:
        - **PDF/TXT**: For general campus information (brochures, guides, etc.)
        - **CSV/Excel**: For location data (must contain name, latitude, longitude)
        """, icon="ℹ️")
        
        uploaded_files = st.file_uploader(
            "Select files", 
            type=['pdf', 'txt', 'csv', 'xlsx', 'xls'], 
            accept_multiple_files=True,
            help="You can select multiple files at once"
        )
        
        if st.button("🔄 Process & Build Knowledge Base", type="primary"):
            if uploaded_files:
                with st.spinner("Processing files..."):
                    file_data, csv_locs = process_uploaded_files(uploaded_files)
                    full_text = " ".join([d['text'] for d in file_data])
                    text_locs = extract_locations_from_text(full_text)
                    all_locations = {**csv_locs, **text_locs}
                    corpus_sentences = extract_sentences(file_data)
                    
                    success, num_sentences, num_locations = build_and_save_data(corpus_sentences, all_locations)
                    
                    if success:
                        if num_sentences > 0 or num_locations > 0:
                            st.success(f"""
                            ✅ Processing complete!
                            - Indexed {num_sentences} knowledge sentences
                            - Added {num_locations} campus locations
                            """)
                            st.balloons()
                        else:
                            st.warning("Processed files but found no usable data. Check file contents.")
                    else:
                        st.error("Processing failed. Check error messages above.")
            else:
                st.warning("Please upload at least one file.", icon="⚠️")
    
    with tab2:
        st.subheader("System Status")
        
        index, corpus, location_map = load_system_data()
        system_ready = (index is not None and corpus) or bool(location_map)
        
        col1, col2 = st.columns(2)
        col1.metric("System Status", "✅ Ready" if system_ready else "❌ Not Ready")
        col2.metric("Knowledge Base", f"{len(corpus) if corpus else 0} facts")
        
        st.markdown("---")
        st.subheader("Location Database")
        
        if location_map:
            st.success(f"📌 {len(location_map)} locations available")
            
            # Location search and preview
            search_term = st.text_input("Search locations", key="loc_search")
            filtered_locs = [
                loc for name, loc in location_map.items() 
                if not search_term or search_term.lower() in name.lower()
            ][:20]  # Limit to 20 results
            
            if filtered_locs:
                selected_loc = st.selectbox(
                    "Preview location",
                    options=[loc['name'].title() for loc in filtered_locs],
                    index=0
                )
                
                selected_data = next(loc for loc in filtered_locs if loc['name'].title() == selected_loc)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**Name:** {selected_data['name'].title()}")
                    st.write(f"**Latitude:** {selected_data['lat']:.6f}")
                    st.write(f"**Longitude:** {selected_data['lon']:.6f}")
                    st.write(f"**Description:** {selected_data.get('desc', 'No description')}")
                
                with col2:
                    preview_map = folium.Map(
                        location=[selected_data['lat'], selected_data['lon']], 
                        zoom_start=16
                    )
                    folium.Marker(
                        [selected_data['lat'], selected_data['lon']],
                        tooltip=selected_data['name'].title()
                    ).add_to(preview_map)
                    st_folium(preview_map, width=400, height=300)
        else:
            st.warning("No location data available. Upload CSV/Excel files with location data.")
    
    with tab3:
        st.subheader("System Management")
        st.warning("These actions are irreversible. Proceed with caution.", icon="⚠️")
        
        if st.button("🗑️ Clear Chat History", key="clear_chat_hist"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        st.markdown("---")
        st.subheader("Reset System")
        
        if st.button("🔄 Reset Knowledge Base", type="secondary"):
            st.session_state.confirm_reset = True
        
        if st.session_state.get("confirm_reset", False):
            st.error("""
            ❗ This will delete ALL indexed data:
            - All document knowledge
            - All location data
            - Cannot be undone
            """)
            
            col1, col2 = st.columns(2)
            if col1.button("✅ Confirm Reset", type="primary"):
                try:
                    if os.path.exists(STORAGE_DIR):
                        shutil.rmtree(STORAGE_DIR)
                    os.makedirs(STORAGE_DIR, exist_ok=True)
                    st.success("Knowledge base reset complete!")
                    st.session_state.confirm_reset = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Reset failed: {e}")
            
            if col2.button("❌ Cancel"):
                st.session_state.confirm_reset = False
                st.rerun()

# --------------- User Page ---------------
def user_page():
    st.title("🏫 CampusGPT Assistant")
    st.markdown("---")
    
    # Load system data
    index, corpus, location_map = load_system_data()
    system_ready = (index is not None and corpus) or bool(location_map)
    
    if not system_ready:
        display_welcome_message()
        st.info("""
        ℹ️ The system is not yet ready. An administrator needs to upload campus documents 
        and build the knowledge base before you can ask questions.
        """)
        return
    
    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for i, msg in enumerate(st.session_state.chat_history):
        is_user = msg["role"] == "user"
        
        # User message
        if is_user:
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
                <div style="background: #4b6cb7; color: white; padding: 12px 16px; 
                            border-radius: 18px 18px 0 18px; max-width: 80%;">
                    {msg["content"]}
                </div>
                <div style="margin-left: 10px; font-size: 24px; align-self: center;">👤</div>
            </div>
            """, unsafe_allow_html=True)
        # Assistant message
        else:
            st.markdown(f"""
            <div style="display: flex; margin-bottom: 20px;">
                <div style="margin-right: 10px; font-size: 24px; align-self: center;">🏫</div>
                <div style="background: #f0f2f6; color: #333; padding: 12px 16px; 
                            border-radius: 18px 18px 18px 0; max-width: 80%;">
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show map if locations are mentioned
            if "locations" in msg and msg["locations"]:
                with st.expander("📍 View on Map"):
                    map_obj = create_map(msg["locations"])
                    if map_obj:
                        st_folium(map_obj, width=700, height=400, key=f"map_{i}")
    
    # Chat input
    if prompt := st.chat_input("Ask about campus locations, services, or facilities..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()
    
    # Generate assistant response if last message was from user
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.spinner("Searching campus information..."):
            last_prompt = st.session_state.chat_history[-1]["content"]
            
            # Retrieve relevant information
            chunks = retrieve_chunks(last_prompt, corpus, index)
            locs = match_locations(last_prompt, location_map)
            loc_info = "\n".join([f"{l['name'].title()}: (Lat: {l['lat']}, Lon: {l['lon']})" for l in locs])
            dist_info = compute_distance_info(locs)
            
            # Get chatbot response
            response = ask_chatbot(last_prompt, chunks, loc_info, dist_info)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response, 
                "locations": locs if locs else None
            })
            st.rerun()

# --------------- Main App ---------------
def main():
    # Custom CSS for the entire app
    st.markdown("""
    <style>
        /* Main styling */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #4b6cb7 0%, #182848 100%);
            color: white;
        }
        
        [data-testid="stSidebar"] .stRadio label {
            color: white !important;
        }
        
        [data-testid="stSidebar"] .stButton button {
            background-color: white;
            color: #4b6cb7;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            margin-top: 20px;
        }
        
        /* Chat bubbles */
        .user-bubble {
            background: #4b6cb7;
            color: white;
            border-radius: 18px 18px 0 18px;
            padding: 12px 16px;
            margin-bottom: 10px;
            max-width: 80%;
            margin-left: auto;
        }
        
        .assistant-bubble {
            background: #f0f2f6;
            color: #333;
            border-radius: 18px 18px 18px 0;
            padding: 12px 16px;
            margin-bottom: 10px;
            max-width: 80%;
        }
        
        /* Input styling */
        .stTextInput input {
            border-radius: 20px;
            padding: 10px 15px;
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Tab styling */
        [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        [data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 8px 16px !important;
            background: #f0f2f6 !important;
            transition: all 0.2s !important;
        }
        
        [aria-selected="true"] {
            background: #4b6cb7 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=CampusGPT", width=150)
        st.markdown("---")
        
        app_mode = st.radio(
            "Navigation",
            ["👤 User Chat", "🔧 Admin Portal"],
            index=0,
            key="app_mode"
        )
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", key="sidebar_clear_chat"):
            if "chat_history" in st.session_state:
                st.session_state.chat_history = []
                st.toast("Chat history cleared!", icon="🔄")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #aaa; font-size: 12px; margin-top: 20px;">
            CampusGPT v1.0<br>© 2023 University Assistant
        </div>
        """, unsafe_allow_html=True)
    
    # Page routing
    if app_mode == "👤 User Chat":
        user_page()
    else:
        admin_page()

if __name__ == "__main__":
    main()
