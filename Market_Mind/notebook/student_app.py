import streamlit as st
import student_logic as logic

st.set_page_config(layout="wide")
st.title("Student Exercise – Full Model")

st.write("This app runs the full notebook model using yfinance.")

if st.button("Run Model"):
    outputs = logic.run_full_model()

    st.subheader("Downloaded Data (last 5 rows)")
    st.dataframe(outputs["raw_data"].tail())

    st.subheader("Processed Data (last 5 rows)")
    st.dataframe(outputs["processed_data"].tail())

    col1, col2 = st.columns(2)
    col1.metric("Mean Return", f"{outputs['mean_return']:.5f}")
    col2.metric("Volatility", f"{outputs['volatility']:.5f}")

    st.subheader("Price Chart")
    st.pyplot(outputs["price_plot"])

    st.subheader("Returns Chart")
    st.pyplot(outputs["returns_plot"])

    st.subheader("Candlestick Chart")
    st.plotly_chart(
        logic.plot_candlestick(outputs["processed_data"]),
        use_container_width=True
    )



