import streamlit as st

home_page = st.Page("./pages/1_home.py", title="🏠 Home")
simulation_page = st.Page("./pages/2_simulation.py", title="💬 simulation")
historical_page = st.Page("./pages/3_historical.py", title="⌚ historical")

pg = st.navigation([home_page, simulation_page, historical_page])
pg.run()
