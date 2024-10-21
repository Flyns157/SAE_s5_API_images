from neo4j import GraphDatabase
from flask import Flask
import bcrypt

class Database:
    def __init__(self, neo4j_uri:str = None, neo4j_user:str = None, neo4j_password:str = None) -> None:
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        if neo4j_uri and neo4j_password and neo4j_user:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    @staticmethod
    def hash_password(password:str)->bytes:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    @staticmethod
    def check_password(password:str, hashed:bytes)->bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    
    def init_app(self, app:Flask)->None:
        self.app = app
        self.neo4j_uri = app.NEO4J_URI
        self.neo4j_user = app.NEO4J_USER
        self.neo4j_password = app.NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(app.NEO4J_URI, auth=(app.NEO4J_USER, app.NEO4J_PASSWORD))

    def create_user(self, username:str, password:str):
        hashed_password = self.hash_password(password)
        query = "CREATE (u:User {username: $username, password: $password})"
        with self.driver.session() as session:
            session.run(query, username=username, password=hashed_password.decode('utf-8'))

    def find_user_by_username(self, username:str):
        query = "MATCH (u:User {username: $username}) RETURN u.username AS username, u.password AS password"
        with self.driver.session() as session:
            result = session.run(query, username=username)
            return result.single()
