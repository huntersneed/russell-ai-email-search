import os
from msal import PublicClientApplication
import requests
from email import message_from_bytes
import base64
from dotenv import load_dotenv
import webbrowser
import time
import json
import random

# Load environment variables
load_dotenv()

# Function to clean up the folder names and save emails
def save_email(email_message, folder, email_id):
    try:
        # Clean up the folder name
        folder = ''.join(c for c in folder if c.isalnum() or c in (' ', '_', '-'))
        folder = folder.strip()

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Save the email
        filename = os.path.join(folder, f"{email_id.decode()}.eml")
        with open(filename, 'wb') as f:
            f.write(email_message.as_bytes())
        print(f"Saved email with ID: {email_id.decode()} to {filename}")
        return True
    except Exception as e:
        print(f"Error saving email with ID {email_id.decode()}: {str(e)}")
        return False

# Azure AD app registration details
CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")

# Microsoft Graph API endpoints
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/Mail.Read"]
ENDPOINT = "https://graph.microsoft.com/v1.0"

# Create a public client application
app = PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

# Acquire a token
result = None
accounts = app.get_accounts()
if accounts:
    result = app.acquire_token_silent(SCOPE, account=accounts[0])

if not result:
    flow = app.initiate_device_flow(scopes=SCOPE)
    if "user_code" not in flow:
        raise ValueError("Failed to create device flow")

    print(flow["message"])
    webbrowser.open(flow["verification_uri"])

    result = app.acquire_token_by_device_flow(flow)

if "access_token" in result:
    access_token = result["access_token"]
    print("Successfully acquired access token")
else:
    print(f"Error acquiring token: {result.get('error')}")
    print(f"Error description: {result.get('error_description')}")
    exit()

def download_emails(folder_id, local_folder):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    # Ensure the directory exists
    os.makedirs(local_folder, exist_ok=True)

    # Get messages from the folder
    messages_url = f"{ENDPOINT}/me/mailFolders/{folder_id}/messages?$top=300"
    all_messages = []
    page_count = 0

    print(f"Fetching messages from {folder_id}...")
    while messages_url:
        page_count += 1
        print(f"Fetching page {page_count}...")
        
        max_retries = 5
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                response = requests.get(messages_url, headers=headers, timeout=30)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Failed to fetch messages after {max_retries} attempts: {str(e)}")
                    return
                print(f"Error fetching messages (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                retry_delay += random.uniform(0, 1)  # Add jitter

        data = response.json()
        messages = data.get('value', [])
        all_messages.extend(messages)
        print(f"Retrieved {len(messages)} messages from page {page_count}. Total messages so far: {len(all_messages)}")
        
        # Check if there are more pages
        messages_url = data.get('@odata.nextLink')
        if messages_url:
            print("More pages available. Continuing to next page...")
            time.sleep(1)  # Add a small delay to avoid hitting rate limits

    print(f"Found a total of {len(all_messages)} messages in {folder_id}")

    if not all_messages:
        print(f"No messages found in {folder_id}. This could be due to an empty folder or insufficient permissions.")
        return

    print(f"Starting to download emails from {folder_id} in batches...")
    batch_size = 20  # Adjust this value based on your needs and API limits
    saved_count = 0
    for i in range(0, len(all_messages), batch_size):
        batch = all_messages[i:i+batch_size]
        batch_requests = []
        for msg in batch:
            batch_requests.append({
                "id": msg['id'],
                "method": "GET",
                "url": f"/me/messages/{msg['id']}/$value"
            })

        batch_body = {
            "requests": batch_requests
        }

        batch_response = requests.post(f"{ENDPOINT}/$batch", headers=headers, json=batch_body)
        if batch_response.status_code == 200:
            batch_data = batch_response.json()
            for response in batch_data['responses']:
                if response['status'] == 200:
                    try:
                        email_content = base64.b64decode(response['body'])
                        email_message = message_from_bytes(email_content)
                        if save_email(email_message, local_folder, response['id'].encode()):
                            saved_count += 1
                    except Exception as e:
                        print(f"Error processing email {response['id']}: {str(e)}")
                else:
                    print(f"Failed to retrieve email {response['id']}: {response['status']}")
        else:
            print(f"Batch request failed: {batch_response.status_code}")
            print(f"Response content: {batch_response.text}")

        print(f"Downloaded and saved {saved_count} out of {len(all_messages)} emails...")
        time.sleep(1)  # Add a small delay between batches to avoid hitting rate limits

    print(f"Finished downloading all emails from {folder_id}. Total saved: {saved_count}")

print("Starting email download process...")

# Download emails from the inbox
print("\nDownloading inbox emails...")
download_emails("inbox", "Emails/Inbox")

# Download emails from the sent folder
print("\nDownloading sent emails...")
download_emails("sentitems", "Emails/Sent")

print("\nEmail download process complete.")
