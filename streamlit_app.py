import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import gspread
import io
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Commodity Bill Processor",
    page_icon="📄",
    layout="centered"
)

# --- Google API Authentication Function (Defined First) ---
@st.cache_resource
def get_google_clients():
    """Initializes and returns authenticated Google API clients."""
    try:
        # Convert the secrets object into a real dictionary
        creds_info = dict(st.secrets["gcp_creds"])
        
        # Manually replace the literal '\\n' with actual newline characters
        creds_info['private_key'] = creds_info['private_key'].replace('\\n', '\n')
        
        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/cloud-platform'
            ]
        )
        
        # Import and initialize Vertex AI
        from google.cloud import aiplatform
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part

        aiplatform.init(project=st.secrets["GCP_PROJECT_ID"], location='us-central1', credentials=creds)
        model = GenerativeModel("gemini-1.5-flash-001")
        
        # Build other Google services
        drive_service = build('drive', 'v3', credentials=creds)
        gspread_client = gspread.authorize(creds)
        
        return drive_service, gspread_client, model

    except Exception as e:
        st.error(f"Failed to initialize Google clients: {e}")
        st.exception(e)
        return None, None, None


# --- Initialize Clients and Load Global Variables ---
try:
    drive_service, gspread_client, gemini_model = get_google_clients()
    GOOGLE_SHEET_ID = st.secrets["GOOGLE_SHEET_ID"]
    GOOGLE_DRIVE_FOLDER_ID = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]

    # Check if the initialization was successful
    if not all([drive_service, gspread_client, gemini_model]):
        st.error("Could not initialize one or more Google services. The app cannot continue.")
        st.stop()

except KeyError as e:
    st.error(f"A required key is missing from your .streamlit/secrets.toml file: {e}")
    st.info("Please ensure GCP_PROJECT_ID, GOOGLE_SHEET_ID, GOOGLE_DRIVE_FOLDER_ID, and the [gcp_creds] section are all present.")
    st.stop()


# --- Core Processing Functions ---
def analyze_bill_type_and_party(image_bytes):
    """Uses Gemini to identify if it's a Loading/Unloading bill and find the primary party name."""
    if not gemini_model: return None, None
    from vertexai.generative_models import Part
    prompt = """Analyze this image of a bill. Your task is to determine two things: 1. Bill Type: Is this a "Loading Bill" or an "Unloading Bill"? - A Loading Bill usually has the seller's name prominently at the top. - An Unloading Bill usually has the buyer's name prominently at the top. 2. Party Name: Extract the full name of this primary party. Provide the output in a clean JSON format with keys "bill_type" and "party_name"."""
    image_part = Part.from_data(image_bytes, mime_type="image/jpeg")
    response = gemini_model.generate_content([prompt, image_part])
    try:
        cleaned_response = response.text.strip().replace("`", "").replace("json", "")
        data = json.loads(cleaned_response)
        return data.get("bill_type"), data.get("party_name")
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        st.error(f"Error parsing Gemini's first response: {e}")
        st.text_area("Gemini Raw Response", response.text)
        return None, None

def extract_bill_details(image_bytes):
    """Uses Gemini to extract specific fields from the bill."""
    if not gemini_model: return None
    from vertexai.generative_models import Part
    prompt = """You are an expert OCR data extractor for agricultural commodity bills. From the provided image, extract the following fields. If a field is not present, use "N/A". - Contract No: (P.O. No. or Contract No.), Bill No:, Date:, Lorry No: (Vehicle No. or Truck/Gadi Number), Party Name: (Buyer/Seller Name), Weight: (Total weight or 'Vajan' in kg), Rate: (Rate or 'Bhav'), Bags: (Total bags/Katte/Bore), Quality: (The type of commodity, e.g., Paddy, IR धान, Rice, etc.). Provide the output as a single, clean JSON object."""
    image_part = Part.from_data(image_bytes, mime_type="image/jpeg")
    response = gemini_model.generate_content([prompt, image_part])
    try:
        cleaned_response = response.text.strip().replace("`", "").replace("json", "")
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, AttributeError) as e:
        st.error(f"Error parsing Gemini's second response: {e}")
        st.text_area("Gemini Raw Response", response.text)
        return None

def get_or_create_drive_folder(party_name):
    """Finds a subfolder by name in the main Drive folder, or creates it if not found."""
    if not drive_service: return None
    query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and name = '{party_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    response = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = response.get('files', [])
    if items:
        return items[0]['id']
    else:
        file_metadata = {'name': party_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [GOOGLE_DRIVE_FOLDER_ID]}
        folder = drive_service.files().create(body=file_metadata, fields='id').execute()
        return folder.get('id')

def upload_to_drive(folder_id, file_name, image_bytes):
    """Uploads the image file to the specified Google Drive folder."""
    if not drive_service: return None
    media = MediaIoBaseUpload(io.BytesIO(image_bytes), mimetype='image/jpeg', resumable=True)
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
    return file.get('webViewLink')

def get_or_create_worksheet(party_name):
    """Finds a worksheet by name, or creates it if not found (case-insensitive)."""
    if not gspread_client: return None
    spreadsheet = gspread_client.open_by_key(GOOGLE_SHEET_ID)
    try:
        return spreadsheet.worksheet(party_name)
    except gspread.WorksheetNotFound:
        for sheet in spreadsheet.worksheets():
            if sheet.title.lower() == party_name.lower():
                return sheet
        return spreadsheet.add_worksheet(title=party_name, rows="100", cols="20")

def update_google_sheet(worksheet, data_dict):
    """Appends data to the Google Sheet, adding headers if needed."""
    if not worksheet: return
    headers = worksheet.row_values(1)
    new_headers = [h for h in data_dict.keys() if h not in headers]
    if new_headers:
        if not headers:
             worksheet.update('A1', [list(data_dict.keys())])
        else:
            start_col = gspread.utils.rowcol_to_a1(1, len(headers) + 1)
            worksheet.update(start_col, [new_headers])
    all_headers = worksheet.row_values(1)
    new_row = [data_dict.get(h, "") for h in all_headers]
    worksheet.append_row(new_row, value_input_option='USER_ENTERED')

# --- Streamlit UI ---
st.title("Rishabh Corporation Loading/Unloading Bill AI Processor")
st.markdown("Upload a Loading or Unloading Bill to automatically extract data, file the image, and update your Google Sheet.")
st.markdown("डेटा निकालने, छवि फ़ाइल करने और अपनी Google शीट को अपडेट करने के लिए लोडिंग या अनलोडिंग बिल अपलोड करें।")

uploaded_file = st.file_uploader("Choose a bill image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Uploaded Bill", width=300)

    with st.spinner("Analyzing bill type and party name..."):
        bill_type, party_name = analyze_bill_type_and_party(image_bytes)
    
    if bill_type and party_name:
        st.info(f"Detected **{bill_type}** for party: **{party_name}**")
        
        with st.spinner(f"Processing... Creating folder and sheet for {party_name}..."):
            drive_folder_id = get_or_create_drive_folder(party_name)
            worksheet = get_or_create_worksheet(party_name)
        
        with st.spinner("Uploading image to Google Drive..."):
            file_name = uploaded_file.name
            drive_link = upload_to_drive(drive_folder_id, file_name, image_bytes)
            
        with st.spinner("Extracting detailed data from the bill..."):
            extracted_data = extract_bill_details(image_bytes)

        if extracted_data:
            with st.spinner("Updating Google Sheet..."):
                update_google_sheet(worksheet, extracted_data)
            
            st.success("🎉 Process Complete!")
            st.markdown(f"**Image successfully filed in Google Drive.** [View File]({drive_link})")
            st.write("Extracted Data:")
            st.json(extracted_data)
        else:
            st.error("Could not extract detailed data from the bill. Please check the image quality.")
    else:
        st.error("Could not determine the bill type or party name. The AI might have had trouble reading the image.")
