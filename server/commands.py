import click
from flask.cli import with_appcontext
from db import get_db

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Drops existing collections and creates them with validators."""
    db = get_db()
    
    click.echo("Applying database validators...")

    # --- Schemas ---
    collections_to_validate = {
        'users': {
            '$jsonSchema': {
                'bsonType': 'object',
                'required': ['username', 'email', 'password_hash', 'role', 'created_at', 'first_name', 'last_name'],
                'properties': {
                    'username': {'bsonType': 'string'},
                    'email': {'bsonType': 'string'},
                    'first_name': {'bsonType': 'string'},
                    'last_name': {'bsonType': 'string'},
                    'password_hash': {'bsonType': 'string'},
                    'role': {'enum': ['admin', 'annotator', 'researcher']},
                    'cvat_user_id': {'bsonType': ['int', 'string', 'null']},
                    'created_at': {'bsonType': 'date'},
                    'last_login': {'bsonType': 'date'}
                }
            }
        },
        'datasets': {
            '$jsonSchema': {
                'bsonType': 'object',
                'required': ['name', 'owner_id', 'created_at', 'files'],
                'properties': {
                    'name': {'bsonType': 'string'},
                    'description': {'bsonType': 'string'},
                    'owner_id': {'bsonType': 'objectId'},
                    'created_at': {'bsonType': 'date'},
                    'files': {
                        'bsonType': 'array',
                        'items': {
                            'bsonType': 'object',
                            'required': ['gridfs_id', 'filename', 'type'],
                            'properties': {
                                'gridfs_id': {'bsonType': 'objectId'},
                                'filename': {'bsonType': 'string'},
                                'type': {'enum': ['image', 'mask', 'patch']},
                                'width': {'bsonType': 'int'},
                                'height': {'bsonType': 'int'}
                            }
                        }
                    }
                }
            }
        },
        'inferences': {
            '$jsonSchema': {
                'bsonType': 'object',
                'required': ['dataset_id', 'requested_by', 'status', 'created_at'],
                'properties': {
                    'dataset_id': {'bsonType': 'objectId'},
                    'requested_by': {'bsonType': 'objectId'},
                    'params': {'bsonType': 'object'},
                    'status': {'enum': ['queued', 'running', 'completed', 'failed']},
                    'notes': {'bsonType': 'string'},
                    # Results schema is flexible, no change needed here.
                    # It will store objects like:
                    # { source_filename: "...", class_mask_id: "...", instance_mask_id: "..." }
                    'results': {'bsonType': 'array', 'items': {'bsonType': 'object'}},
                    'created_at': {'bsonType': 'date'},
                    'finished_at': {'bsonType': 'date'}
                }
            }
        }
    }
    
    for name, validator in collections_to_validate.items():
        if name in db.list_collection_names():
            db.drop_collection(name)
            click.echo(f"Dropped collection: {name}")
        
        try:
            db.create_collection(name, validator=validator)
            click.echo(f"Created '{name}' collection with validator.")
        except Exception as e:
            click.echo(f"Failed to create collection {name}: {e}")

    click.echo("Database initialization complete.")

def register_commands(app):
    app.cli.add_command(init_db_command)