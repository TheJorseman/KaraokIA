import sqlite3

class DBManager():
    def __init__(self, name="karaokia.db") -> None:
        self.connection = sqlite3.connect(name)
        try:
            self.connection.execute("""CREATE TABLE karaoke (
                                    id integer PRIMARY KEY AUTOINCREMENT,
                                    yt_id text
                                )""")                     
        except sqlite3.OperationalError:
            print("Reading Table Karaoke")                    
        #self.connection.close()

    def insert(self, yt_id):
        self.connection.execute("INSERT INTO karaoke(yt_id) VALUES (?)", (yt_id,))
        self.connection.commit()
        #self.connection.close()

    def select(self):
        cursor = self.connection.execute("SELECT yt_id FROM karaoke")
        result = [fila for fila in cursor]
        #self.connection.close()
        return result

    def select_by_ytid(self, _yt_id):
        cursor = self.connection.execute("SELECT yt_id FROM karaoke WHERE yt_id==?", (_yt_id,))
        result = [fila for fila in cursor]
        #self.connection.close()
        return result
    
    def yt_id_exist(self, yt_id):
        cursor = self.connection.execute("SELECT yt_id FROM karaoke WHERE yt_id==?", (yt_id,))
        fila = cursor.fetchone()
        exist = fila != None
        #self.connection.close()
        return exist
        