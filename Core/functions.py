import json
import streamlit as st
from hashlib import sha512
import os

def load_users():
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_details.json")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def login(st):
    st.title("Login")
    username = st.text_input("Username").replace(" ", "").lower()
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == '':
            st.error("Please enter username")
            return
        if password == '':
            st.error("Please enter password")
            return

        password_hash = sha512(bytes(password, 'utf-8')).hexdigest()
        users = load_users()
        
        user_found = False
        for user in users:
            if user['Username'] == username and user['Password'] == password_hash:
                user_found = True
                break
        
        if user_found:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Successfully logged in")
            st.rerun()
        else:
            st.error("Invalid username or password")
            return

def logout():
    st.session_state.logged_in = False
    st.rerun()
