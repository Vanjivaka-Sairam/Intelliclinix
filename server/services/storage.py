from db import get_fs

def save_file_to_gridfs(file_storage, metadata = None): #save a file like images to the gridfs
    fs = get_fs()
    return fs.put(
        file_storage.read(),
        filename=file_storage.filename,
        content_type = file_storage.content_type,
        metadata=metadata
    )

def save_bytes_to_gridfs(data : bytes, filename : str, metadata : dict = None): #saves raw bytes like generated masks to the gridfs
    fs = get_fs()
    return fs.put(data, filename = filename, metadata = metadata)

def get_file_from_gridfs(file_id):
    fs = get_fs()
    return fs.get(file_id)

