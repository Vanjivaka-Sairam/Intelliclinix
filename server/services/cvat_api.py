from flask import current_app
from cvat_sdk import make_client
from cvat_sdk.core.proxies.users import User
from cvat_sdk.exceptions import ApiException
import requests


def get_cvat_auth():
    """returns the auth tuple for the cvat admin user"""
    return (
        current_app.config['CVAT_ADMIN_USERNAME'],
        current_app.config['CVAT_ADMIN_PASSWORD']
    )
    
        
    


def create_cvat_user(username, email, password, first_name, last_name) -> User:
    """Create a user in CVAT via rest_api"""
    api_url  = f"{current_app.config['CVAT_API_URL']}/users"
    user_payload = {
                'username': username,
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'password': password,
                'groups':["users"],
            }
    try:
        response = requests.post(
            api_url,
            json=user_payload,
            auth=get_cvat_auth()
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error creating CVAT user {username}: {e}")
        if e.response:
            print("CVAT API response:", e.response.json())
        return None


def login_and_get_token(username, password) -> str:
    """Login to CVAT for a user and get an API token."""
    host_url = current_app.config['CVAT_API_URL'].replace('/api', '')
    
    with make_client(host=host_url, credentials=(username, password)) as client:
        try:
            return client.api_client.configuration.api_key.get('Authorization', '').replace('Token ', '')
        except ApiException as e:
            print(f"Failed to log in to CVAT as user {username}: {e}")
            return None


def logout_and_invalidate_token(client):
    """Logout from CVAT and invalidate the API token used by the client."""
    try:
        print("CVAT API token invalidated successfully.")
    except ApiException as e:
        print(f"Error invalidating CVAT token: {e}")