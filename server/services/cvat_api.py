import os
from pprint import pprint
from flask import Flask, request, jsonify
from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.api_client.models import RegisterSerializerExRequest, LoginSerializerExRequest

CVAT_HOST = os.getenv('CVAT_API_URL')
CVAT_ADMIN_USERNAME = os.getenv('CVAT_API_USER') 
CVAT_ADMIN_PASSWORD = os.getenv('CVAT_API_PASSWORD') 

def get_cvat_user_id():
    from cvat_sdk.core.helpers import make_client
    configuration = Configuration(
        host=CVAT_HOST,
        username=CVAT_ADMIN_USERNAME,
        password=CVAT_ADMIN_PASSWORD,
    )
    with ApiClient(configuration) as api_client:
        with make_client(current_app.config["CVAT_API_URL"]) as client:
            client.login(
                CVAT_ADMIN_USERNAME,
                CVAT_ADMIN_PASSWORD
            )
            users = client.users.list()
            for user in users:
                if user.username == username:
                    return user.id
    return None

def create_cvat_user(username: str, email: str, password: str, first_name: str, last_name: str):
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    required_fields = ["username", "email", "password", "first_name", "last_name"]
    for f in required_fields:
        if f not in data:
            return jsonify({"error": f"Missing field: {f}"}), 400
    configuration = Configuration(
        host=CVAT_HOST,
        username=CVAT_ADMIN_USERNAME,
        password=CVAT_ADMIN_PASSWORD,
    )

    try:
        with ApiClient(configuration) as api_client:
            register_request = RegisterSerializerExRequest(
                username=username,
                email=email,
                password1=password,
                password2=password,
                first_name=first_name,
                last_name=last_name,
            )

            (created_user, response) = api_client.auth_api.create_register(register_request)
            print(created_user)
            print(response)
            return jsonify({"message": "User registered successfully", "data": created_user.to_dict()}), 201

    except exceptions.ApiException as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

def cvat_login(host: str, username: str, password: str, email: str = None):

    configuration = Configuration(
        host=CVAT_HOST,
        username=CVAT_ADMIN_USERNAME,
        password=CVAT_ADMIN_PASSWORD,
    )

    with ApiClient(configuration) as api_client:
        login_request = LoginSerializerExRequest(
            username=username,
            email=email or "",
            password=password,
        )

        try:
            (data, response) = api_client.auth_api.create_login(login_request)
            pprint(data)
            return data.to_dict()
        except exceptions.ApiException as e:
            print(f"Exception when calling AuthApi.create_login(): {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Unexpected error during login: {e}")
            return {"error": str(e)}

def cvat_logout():
    configuration = Configuration(
        host=CVAT_HOST,
        username=CVAT_ADMIN_USERNAME,
        password=CVAT_ADMIN_PASSWORD,
    )

    with ApiClient(configuration) as api_client:
        try:
            (data, response) = api_client.auth_api.create_logout()
            pprint(data)
            return {"message": "User logged out successfully", "data": data.to_dict() if data else {}}
        except exceptions.ApiException as e:
            print(f"Exception when calling AuthApi.create_logout(): {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Unexpected error during logout: {e}")
            return {"error": str(e)}
