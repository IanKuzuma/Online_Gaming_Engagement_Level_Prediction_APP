import prediction, eda
import streamlit as st
import base64

# Create sidebar navigation
with st.sidebar:
    st.subheader('PAGE NAVIGATION')

    # Create the selectbox for the pages to run each scripts
    page = st.selectbox('Choose Page',
                        ['Data Exploration','Data Prediction'])
    
    # Write sidebar info
    st.subheader('About')

    # Load and encode the gif
    file_ = open("jorb.gif", "rb")
    contents = file_.read()
    data_url = "data:image/gif;base64," + base64.b64encode(contents).decode("utf-8")

    # Display the gif using HTML
    st.markdown(
        f'<img src="{data_url}" alt="gif" style="width:25%;" />',
        unsafe_allow_html=True
    )

    st.markdown('''This project focuses on predicting a player’s Engagement Level, 
                which are categorized as Low, Medium, or High, using behavioral and demographic data.''')

if page == 'Data Exploration':
    eda.run()

else:
    prediction.run()