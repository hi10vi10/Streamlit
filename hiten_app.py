import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport

def main():
    st.title("Data Profiling App")

    uploaded_file = st.file_uploader("Upload a new dataset (CSV or Excel)", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                new_data = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip', engine='python')
            else:
                new_data = pd.read_excel(uploaded_file)
            
            st.write("Uploaded Data:")
            st.write(new_data.head())

            with st.spinner("Generating report..."):
                profile = ProfileReport(new_data, title="Data Profiling Report")
                profile.to_file("output.html")

            st.success("Report generated!")

            st.write("Download Report:")
            with open("output.html", "r", encoding='ISO-8859-1') as f:
                html_content = f.read()
                st.download_button("Download Report", html_content, "report.html", "text/html")

        except Exception as e:
            st.error("An error occurred: " + str(e))

if __name__ == "__main__":
    main()
