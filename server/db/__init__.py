from flask import current_app
from flask_pymongo import PyMongo
from gridfs import GridFS

mongo = PyMongo()
fs = None


def init_db(app):
    mongo.init_app(app)
    global fs
    fs = GridFS(mongo.db) #for storing models and file(since they are large)

def get_db():
    return mongo.db 

def get_fs():
    if fs is None:
        raise RuntimeError("GridFS has not been initialized. Call init_db(app) first")
    return fs