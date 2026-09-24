from mysql.connector import Error, connect


class DatabaseManager:

  def __init__(self, db_config: dict):
    self.db_config = db_config

  def _get_connection(self):
    return connect(**self.db_config)

  def fetch_by_id(self, file_id: str):
    """Fetches the complete record from MySQL using the primary key."""
    try:
      conn = self._get_connection()
      cursor = conn.cursor(dictionary=True)
      # Notice 'semantic_sumary' matches your SQL table column definition
      cursor.execute(
          """
                SELECT file_id, file_name, content, semantic_sumary, grade, comments 
                FROM file 
                WHERE file_id = %s
            """,
          (file_id,),
      )
      row = cursor.fetchone()
      cursor.close()
      conn.close()
      return row
    except Error as e:
      print(f"MySQL Fetch Error: {e}")
      return None