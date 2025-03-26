# Banklytics-AI-Powered-Banking-Analytics-and-Financial-Visualization
Banklytics is an interactive platform designed to analyze, classify, and visualize the financial performance of **Panamanian banks**. Leveraging real-world financial data from the **Superintendencia de Bancos de Panamá**, the platform uses machine learning and advanced visualization tools to help uncover insights from key banking metrics.

---

## 📈 Features

- **DuPont ROE Analysis**
  - Net Profit Margin
  - Asset Turnover
  - Equity Multiplier (Leverage)
- **Additional Metrics**
  - Liquidity Ratio
  - Coverage Ratio
  - Capitalization Ratio
  - Deposit Structure & Diversity
- **AI Classification**
  - Tree-based model classifies banks into performance tiers
- **Interactive Visualizations**
  - 2D and 3D Graphs
  - Filters by Bank, Year, Month, Classification
- **Multilingual Dashboard**
  - English and Spanish support

---

## 📆 Data Source

All financial data is obtained from:
**Superintendencia de Bancos de Panamá**  
> https://www.superbancos.gob.pa

---

## ⚙️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io)
- **Backend**: Python (pandas, scikit-learn, plotly)
- **ML Model**: Decision Tree Classifier
- **Visualization**: Plotly (2D/3D), interactive dashboards

---

## 🗒️ Setup

```bash
# Clone the repo
git clone https://github.com/fdelaesp/Banklytics-AI-Powered-Banking-Analytics-and-Financial-Visualization
cd Banklytics

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python app/dashboard.py
```

---

## 📚 License

This project is open-source under the [MIT License](LICENSE).

---

## 🚀 Author

**Francisco de La Espriella**  

---

*"Banklytics: Helping you see the story behind the numbers."*

