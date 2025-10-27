from flask import current_app
from cvat_sdk import make_client
from cvat_sdk.core.proxies.users import User
from cvat_sdk.exceptions import ApiException


def get_admin_client():
    """Get authenticated CVAT client using admin credentials"""
    return make_client(
        host=current_app.config['CVAT_API_URL'].replace('/api', ''),
        credentials=(
            current_app.config['CVAT_ADMIN_USER'],
            current_app.config['CVAT_ADMIN_PASSWORD']
        )
    )


def create_cvat_user(username, email, password, first_name, last_name) -> User:
    """Create a user in CVAT using the admin account"""
    with get_admin_client() as client:
        try:
            user = client.users.create({
                'username': username,
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'password': password
            })
            return user
        except ApiException as e:
            print(f"Error creating CVAT user {username}: {e}")
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