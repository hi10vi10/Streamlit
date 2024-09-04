from matplotlib.lines import Line2D
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from ydata_profiling import ProfileReport

# Function to detect fraudulent transactions
def detect_fraudulent_transactions(new_data: pd.DataFrame, model) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    new_data[['Amount', 'Time']] = scaler.fit_transform(new_data[['Amount', 'Time']])
    new_data['Anomaly'] = model.predict(new_data)
    fraudulent_transactions = new_data[new_data['Anomaly'] == -1]
    return fraudulent_transactions

# Streamlit app
def main():
    st.title("Credit Card Fraud Detection")
    
    uploaded_file = st.file_uploader("Upload a new dataset (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            new_data = pd.read_csv(uploaded_file)
        else:
            new_data = pd.read_excel(uploaded_file, sheet_name='creditcard_test')
        new_data = new_data.apply(pd.to_numeric, errors='coerce')
        # Drop rows with any NaN values
        new_data = new_data.dropna()    
        # Display the first few rows of the uploaded data
        st.write("Uploaded Data:")
        st.write(new_data.head())
        
        # Train a basic Isolation Forest model (for demonstration purposes)
        # In practice, you would use a pre-trained model
        model = IsolationForest(contamination=0.002, random_state=42)
        model.fit(new_data.drop(columns=['Class'], errors='ignore'))
        
        # Detect fraudulent transactions
        fraudulent_transactions = detect_fraudulent_transactions(new_data, model)
        
        # Display the results
        st.write("Detected Fraudulent Transactions:")
        st.write(fraudulent_transactions)
        
        # Visualization of detected anomalies
        st.write("Visualization of Detected Anomalies:")
        fig, ax = plt.subplots()
        scatter = plt.scatter(new_data['Amount'], new_data['Time'], c=new_data['Anomaly'])
        plt.xlabel('Amount')
        plt.ylabel('Time')

        # Add a color bar if needed (for visualization purposes)
        plt.colorbar(scatter, ax=ax)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Fraud',
                              markerfacecolor='black', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Safe Transaction',
                              markerfacecolor='yellow', markersize=10)]
    
        # Adding legend to the plot
        ax.legend(handles=legend_elements, title='Classes', loc = "upper right")
        st.pyplot(plt)
        
        # Option to download the results
        st.write("Download Results:")
        fraudulent_transactions.to_csv("fraudulent_transactions.csv", index=False)
        st.download_button(
            label="Download CSV",
            data=fraudulent_transactions.to_csv(index=False),
            file_name="fraudulent_transactions.csv",
            mime="text/csv",
        )

        profile = ProfileReport(new_data)
        profile.to_file(output_file="output.html")
        # Option to download the results
        st.write("Download Results:")
        with open("output.html", "r") as f:
            html_data = f.read()
        st.download_button(
            label="Download HTML",
            data=html_data,
            file_name="output.html",
            mime="text/html",
        )

if __name__ == '__main__':
    main()
