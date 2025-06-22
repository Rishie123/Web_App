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

# --- Load Secrets (The Secure Way) ---
# When deployed on Streamlit Cloud, it will get these from the Secrets manager.
# For local testing, you would need a secrets.toml file, but we'll focus on deployment.
try:
    gcp_creds_dict = st.secrets["gcp_creds"]
    GCP_PROJECT_ID = st.secrets["GCP_PROJECT_ID"]
    GOOGLE_SHEET_ID = st.secrets["GOOGLE_SHEET_ID"]
    GOOGLE_DRIVE_FOLDER_ID = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
    GCP_LOCATION = 'us-central1'
    MODEL_NAME = 'gemini-1.5-flash-001'
except FileNotFoundError:
    st.error("Secrets not found! Please ensure you have added the secrets in Streamlit Cloud settings.")
    st.stop()
except KeyError as e:
    st.error(f"Missing secret: {e}. Please check your Streamlit Cloud secrets configuration.")
    st.stop()


# --- Google API Authentication ---
@st.cache_resource
def get_google_clients():
    """Initializes and returns authenticated Google API clients."""
    try:
        creds = service_account.Credentials.from_service_account_info(
            gcp_creds_dict,
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/cloud-platform'
            ]
        )
        drive_service = build('drive', 'v3', credentials=creds)
        sheets_service = build('sheets', 'v4', credentials=creds)
        gspread_client = gspread.authorize(creds)

        # Import Vertex AI only when needed and configured
        from google.cloud import aiplatform
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part

        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, credentials=creds)
        model = GenerativeModel(MODEL_NAME)

        return drive_service, sheets_service, gspread_client, model
    except Exception as e:
        st.error(f"Failed to initialize Google clients: {e}")
        return None, None, None, None

drive_service, sheets_service, gspread_client, gemini_model = get_google_clients()


# --- Core Functions ---

def analyze_bill_type_and_party(image_bytes):
    """Uses Gemini to identify if it's a Loading/Unloading bill and find the primary party name."""
    if not gemini_model: return None, None
    
    prompt = """
    Analyze this image of a bill. Your task is to determine two things:
    1.  Bill Type: Is this a "Loading Bill" or an "Unloading Bill"?
        - A Loading Bill usually has the seller's name prominently at the top (e.g., a company name like BHARTI AGRO).
        - An Unloading Bill usually has the buyer's name prominently at the top (e.g., a shop name like लिल्हारे अनाज भण्डार).
    2.  Party Name: Extract the full name of this primary party (the seller for Loading, the buyer for Unloading).

    Provide the output in a clean JSON format with keys "bill_type" and "party_name".
    Example for a loading bill: {"bill_type": "Loading Bill", "party_name": "BHARTI AGRO IMPEX PRIVATE LIMITED"}
    Example for an unloading bill: {"bill_type": "Unloading Bill", "party_name": "लिल्हारे अनाज भण्डार"}
    """
    image_part = Part.from_data(image_bytes, mime_type="image/jpeg")
    response = gemini_model.generate_content([prompt, image_part])
    try:
        # Clean up the response to be valid JSON
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

    prompt = """
    You are an expert OCR data extractor for agricultural commodity bills. From the provided image, extract the following fields. If a field is not present, use "N/A".
    - Contract No: (P.O. No. or Contract No.)
    - Bill No: (Bill No.)
    - Date: (Date)
    - Lorry No: (Vehicle No. or Truck/Gadi Number)
    - Party Name: (Buyer/Seller Name, depending on the bill type)
    - Weight: (Total weight or 'Vajan' in kg. Look for a large number like 18800)
    - Rate: (Rate or 'Bhav')
    - Bags: (Total bags/Katte/Bore. It might be in a column 'nag' or derived from 'Qty Bag')
    - Quality: (The type of commodity, e.g., Paddy, IR धान, Rice, etc.)

    Provide the output as a single, clean JSON object.
    Example:
    {
        "Contract No": "8045",
        "Bill No": "1160",
        "Date": "14-06-2025",
        "Lorry No": "UP70GT2223",
        "Party Name": "RUSHABH CORPORATION",
        "Weight": "27540",
        "Rate": "2020.00",
        "Bags": "340",
        "Quality": "Paddy P"
    }
    """
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
        file_metadata = {
            'name': party_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [GOOGLE_DRIVE_FOLDER_ID]
        }
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
        worksheet = spreadsheet.worksheet(party_name)
        return worksheet
    except gspread.WorksheetNotFound:
        # Let's do a case-insensitive check
        for sheet in spreadsheet.worksheets():
            if sheet.title.lower() == party_name.lower():
                return sheet
        # If still not found, create it
        worksheet = spreadsheet.add_worksheet(title=party_name, rows="100", cols="20")
        return worksheet


def update_google_sheet(worksheet, data_dict):
    """Appends data to the Google Sheet, adding headers if needed."""
    if not worksheet: return

    headers = worksheet.row_values(1)
    new_headers = [h for h in data_dict.keys() if h not in headers]

    # If there are new headers to add, append them to the first row
    if new_headers:
        # If the sheet is completely empty, set all headers
        if not headers:
             worksheet.update('A1', [list(data_dict.keys())])
        else: # Append new headers to the existing ones
            start_col = gspread.utils.rowcol_to_a1(1, len(headers) + 1)
            worksheet.update(start_col, [new_headers])
    
    # Get all headers again to ensure correct order for the new row
    all_headers = worksheet.row_values(1)
    
    # Create the row in the correct order based on headers
    new_row = [data_dict.get(h, "") for h in all_headers]
    worksheet.append_row(new_row, value_input_option='USER_ENTERED')


# --- Streamlit UI ---
st.title("📄 Commodity Bill AI Processor")
st.markdown("Upload a Loading or Unloading Bill to automatically extract data, file the image, and update your Google Sheet.")

uploaded_file = st.file_uploader("Choose a bill image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and gemini_model:
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

elif uploaded_file:
    st.error("Google services are not available. Check secrets and API configurations.")