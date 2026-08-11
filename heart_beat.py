"""
This class is the heart beat of RI-RAG
Here we parse every file from the Dataset used.
These files will constitute
the cyber intelligence foundation
and populate the database accordingly.
Note: Writen with Gemini.

@author Laurie MAV    CEO of JK AI    lmavoungou@outlook.be
"""

from database import Database
import os
import json
import yaml
from scapy.all import rdpcap

class HeartBeat():
    def __init__(self, database):
        self.db = database

    def in_the_database(self, log_entry, file_origin):
        # Optional keys to explicitly check and set to NULL if absent
        OPTIONAL_KEYS = ["duration", "orig_bytes", "service", "resp_bytes"]

        # Standardize metadata keys (e.g., convert @system -> system_name, @proc -> proc)
        cleaned_entry = {}
        for k, v in log_entry.items():
            clean_key = k.lstrip("@")
            if clean_key == "system":
                clean_key = "system_name"
            if clean_key =='stream':
                continue
            cleaned_entry[clean_key] = v

        # Ensure missing target keys are set to None (SQL NULL)
        for key in OPTIONAL_KEYS:
            if key not in cleaned_entry:
                cleaned_entry[key] = None

        # Attach file_origin source metadata
        cleaned_entry["file_origin"] = file_origin

        # Dynamically build field names and value placeholders for MySQL (%s)
        columns = list(cleaned_entry.keys())
        values = [cleaned_entry[col] for col in columns]

        # Escape column names with backticks to handle potential MySQL reserved keywords
        formatted_columns = ", ".join([f"`{col}`" for col in columns])
        placeholders = ", ".join(["%s"] * len(columns))

        # MySQL INSERT IGNORE handles composite primary key conflicts gracefully
        insert_query = (f"INSERT IGNORE INTO connections ({formatted_columns})"
                        f"VALUES ({placeholders})")
        if isinstance(self.db, Database):
            self.db.get_cursor().execute(insert_query, values)
            self.db.get_database().commit()

    def parse_raw_text(self, raw_content: str, file_origin: str):
        """Processes raw string buffers containing NDJSON or plain text logs."""

        for line_num, line in enumerate(raw_content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            # Fallback for plain text log entries
            # Converts the string line back into a full Python dictionary
            log_entry = json.loads(line)
            if type(file_origin) is str and file_origin.__contains__("conn"):
                self.in_the_database(log_entry, file_origin)

    def parse(self, dataset_path):

        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                ext = os.path.splitext(file_name)[1].lower()

                #print(f"\n Processing: {file_name}")

                try:
                    # 1. YAML Metadata
                    if ext in [".yaml", ".yml"]:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            #print(f"  [YAML Title]: {data.get('title', 'N/A')}")

                    # 2. Structured JSON / NDJSON Logs
                    elif ext == ".json":
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    event = json.loads(line)
                                    # Do your payload handling / MySQL insertion here
                                    #print(f"  [JSON Event ID]: {event.get('EventID', 'N/A')}")
                    # 3.
                    elif ext in [".log", ".txt"]:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            self.parse_raw_text(f.read(), file_origin=os.path.basename(file_path))


                    # 4. PCAP Network Captures
                    elif ext in [".pcap", ".cap"]:
                        packets = rdpcap(file_path)
                        #print(f"  [PCAP Packet Count]: {len(packets)}")

                    else:
                        #print(f"  [Skipped]: Unsupported format ({ext})")
                        pass

                except Exception as err:
                    print(f"  [Error reading {file_name}]: {err}")



if __name__ == '__main__':
    db = Database("root", "qwerty")
    db.setting_up_config("mav", "mav", "mav")
    hb = HeartBeat(db)
    hb.parse(r"C:\Users\lmavo\PycharmProjects\RI-RAG\OTRF_DATASET_FOR_RAG")
