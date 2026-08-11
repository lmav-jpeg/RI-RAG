'''
Code for the implementation of the database inside responsible of
receiving raw cyber intelligence data.
Mysql should be installed and set up in the computer
prior execution of the code.

@author Laurie MAV    CEO of JK AI    lmavoungou@outlook.be
'''

import mysql.connector
import re

class Database:
    def __init__(self, root_user:str, root_password:str):
        self.root_user = self.check_inputs(root_user)
        self.root_password = root_password
        self.mybd = None
        self.connection = 0

    def check_inputs(self, string):
        if not re.compile("^[a-zA-Z0-9_-]+$", re.IGNORECASE).match(string):
            raise ValueError(" Invalid input")
        return string

    def initialize(self):
        self.mybd = mysql.connector.connect(
            host="localhost",
            user=self.root_user,
            password=self.root_password, )
        self.connection +=1
        print("Configuration set")

    def setting_up_config(self, user_name, password, db_name):
        try :
            self.initialize()
            self.user_name = self.check_inputs(user_name)
            self.password = password
            self.db_name = self.check_inputs(db_name)
            mycursor = self.mybd.cursor()
            self.cursor = mycursor
            if self.mybd.is_connected():
                print("Connected to MySQL")
                mycursor = self.mybd.cursor()

                # Create database
                mycursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.db_name}` "
                    f"CHARACTER SET utf8mb4"
                    f" COLLATE utf8mb4_unicode_ci"
                )
                print(f"Database '{self.db_name}' created/verified.")

                # Create user
                mycursor.execute(
                    f"CREATE USER IF NOT EXISTS '{self.user_name}'@'localhost' "
                    f"IDENTIFIED BY %s", (self.password,)
                )
                print(f"User '{self.user_name}' created/verified.")

                # Give user privileges on the database
                mycursor.execute(
                    f"GRANT ALL PRIVILEGES ON `{self.db_name}`.* "
                    f"TO '{self.user_name}'@'localhost'"
                )
                mycursor.execute("FLUSH PRIVILEGES")
                print(f"Permissions granted to '{self.user_name}'.")
                mycursor.execute(f"USE `{self.db_name}`")
                self.mybd.commit()
                print("Setup completed successfully.")
                mycursor.close()
        except Exception as e:
            print(e)

    def payload_saving(self):
        pass

    def finish(self):
        self.mybd.close()
        self.connection = 0

    def get_database(self):
        return self.mybd
    def get_cursor(self):
        return self.cursor


db = Database("root","qwerty")
db.setting_up_config("mav", "mav", "mav")