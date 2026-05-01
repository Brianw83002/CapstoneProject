import sqlite3
import os


###################################
#       Connect To Database
###################################
def connectDatabase():
    # Get current file directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Full database path
    DB_PATH = os.path.join(BASE_DIR, "users.db")

    return sqlite3.connect(DB_PATH)


###################################
#       Check If Table Exists
###################################
def tableExists(cursor, tableName):

    cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name=?
    """, (tableName,))

    return cursor.fetchone() is not None


###################################
#       Create Users Table
###################################
def createUsersTable(cursor):

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (

        username TEXT PRIMARY KEY,

        password TEXT NOT NULL
    )
    """)

    print("Users table created successfully.")


###################################
#       Create Videos Table
###################################
def createVideosTable(cursor):

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (

        video_name TEXT PRIMARY KEY,

        username TEXT NOT NULL,

        original_video_path TEXT NOT NULL,

        processed_video_path TEXT NOT NULL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (username) REFERENCES users(username)
    )
    """)

    print("Videos table created successfully.")


###################################
#       Create Video Photos Table
###################################
def createVideoPhotosTable(cursor):

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_photos (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        video_name TEXT NOT NULL,

        photo_path TEXT NOT NULL,

        FOREIGN KEY (video_name) REFERENCES videos(video_name)
    )
    """)

    print("Video photos table created successfully.")


###################################
#               Main
###################################
def main():

    # Connect to database & Create cursor
    connection = connectDatabase()
    cursor = connection.cursor()

    ###################################
    #       Users Table
    ###################################
    if tableExists(cursor, "users"):
        print("Users table already exists.")
    else:
        createUsersTable(cursor)
        

    ###################################
    #       Videos Table
    ###################################
    if tableExists(cursor, "videos"):
        print("Videos table already exists.")
    else:
        createVideosTable(cursor)


    ###################################
    #       Video Photos Table
    ###################################
    if tableExists(cursor, "video_photos"):
        print("Video photos table already exists.")
    else:
        createVideoPhotosTable(cursor)


    ###################################
    #       Save Changes
    ###################################
    connection.commit()
    print("Changes saved successfully.")

    ###################################
    #       Close Database
    ###################################
    connection.close()


###################################
#           Run Program
###################################
if __name__ == "__main__":
    main()