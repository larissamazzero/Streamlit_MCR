import numpy as np
import pandas as pd
import plotly.express as px
from pymcr.mcr import McrAR
import streamlit as st

header = st.container()
code = st.container()


with header:
    st.title("MCR-ALS applied to simulated CPMG signal")
    st.markdown('<div style="text-align: justify;"> Our dataset contains simulated CPMG signals, which means two pure spectra and its mixtures. First, we need to upload our dataset.</div>', unsafe_allow_html=True)
    "\n"
with code:
    # file_path = st.file_uploader("Upload your file here")
    # sheet_name = st.text_input("Sheet name")
    file_path = (r"pages\Dados.xlsx")
    sheet_name = "ILT"

    def mcrALS(file_path, sheet_name):
        # Importing the data
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            data.rename(columns={'eixo tempo': 'amostra'}, inplace=True)
        except:
            return st.error("The data can't be loaded")

        # Getting the pure spectra values
        S = data.iloc[0:2, 1::]

        # Mixture row                                                                    Tds (identificar quantas linhas e fazer para todas)
        row = 7

        # Getting the mixtures
        D = data.iloc[row:row+1, 1::]

        # Getting the mixture label
        ref = data['amostra'][row]

        # Pure spectra 1                                                                 Tds (para cada espectro terá uma linha dessa)
        spectra1 = data.iloc[0, 1::].values

        # Pure spectra 2
        spectra2 = data.iloc[1, 1::].values

        # Column time
        time = data.columns.drop('amostra')

        # Spectra values of the mixture
        data_row = D.values[0].tolist()
        spectra_value = data_row

        # Creating a dataframe to plot the mixture curve
        dt = {'time': time, '125 ms': spectra1,
              '200 ms': spectra2, 'spectra': spectra_value}
        df1 = pd.DataFrame(data=dt, index=range(len(dt['time'])))

        # MCR-ALS object and fitting to the data
        mcrar = McrAR()
        mcrar.fit(D, ST=S)

        # Getting the residual error                                                      TDS (identiciar de qual espectro é cada erro)
        E = mcrar.err

        # Getting the concentrations                                                      TDS (identificar quantas concetrações terão de acordo com a quantiade de features)
        c1 = mcrar.C_opt_[0][0]*100
        c2 = mcrar.C_opt_[0][1]*100

        # Getting the mixtues without the pure spectra
        mix = data.iloc[2::, ::]
        # Listing all the possible mixtures
        mix_values = mix['amostra'].values
        # Creating a JSON with the time value
        dt_mix = {'time': time}

        # Looping for all the possible mixtures
        for m in mix_values:
            # Finding the index of each mixture acourding to the label
            m_index = np.where(mix['amostra'] == m)[0][0]
            # Getting the values of the row acourding to the index
            m_values = mix.iloc[m_index, 1::].values
            # Adding in the dictionary like: "label": [array of the values]
            dt_mix[m] = m_values

        # Creating a dataframe where the columns is the time and each spectra with mixtures
        df2 = pd.DataFrame(data=dt_mix, index=range(len(dt_mix['time'])))
        # Adding the columns 200 e 125 ms                                                   TDS (pra cada espectro puro adicionar a coluna com os valores)
        df2['125 ms'] = dt['125 ms']
        df2['200 ms'] = dt['200 ms']

        # All the possible mixtures and the pure spectras
        mixtures = df2.iloc[:, 1::].columns

        # Plotting all the curves
        fig1 = px.line(df2, x=df2['time'], y=mixtures, width=800, height=700)
        fig1.update_layout(title_x=0.5, yaxis_range=[0, 3.5], xaxis_range=[0.01, 0.6], title={
            'text': "ILT - CPMG mixtures",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}, yaxis_title='Intensity (a.u.)', xaxis_title='T2 (s)', legend_title='Pure Spectra')

        # Plotting curve of the predictec mixture spectra + pure spectra
        fig2 = px.line(df1, x=df1['time'], y=[
            '125 ms', '200 ms'], width=800, height=700)
        fig2.add_scatter(x=df1['time'], y=df1['spectra'],
                         mode='lines', name=f"espectro: {ref}")

        fig2.update_layout(title_x=0.5, yaxis_range=[0, 4.0], xaxis_range=[0.01, 0.6], title={
            'text': "ILT - CPMG spectra",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}, yaxis_title='Intensity (a.u.)', xaxis_title='T2 (s)', legend_title='Pure Spectra')

        fig2.add_annotation(text=f"Mixture: {ref}",
                            xref="paper", yref="paper",
                            x=1, y=0.60, showarrow=False)

        fig2.add_annotation(text="125 ms: {0:.2f}%".format(c1),
                            xref="paper", yref="paper",
                            x=1, y=0.50, showarrow=False)

        fig2.add_annotation(text="200 ms: {0:.2f}%".format(c2),
                            xref="paper", yref="paper",
                            x=1, y=0.55, showarrow=False)

        return st.write('\nFinal MSE: \n {:.7e}'.format(mcrar.err[-1])), st.write("Concentração 125ms: \n {:.2f}%".format(c1)), st.write("Concentração 200ms: \n {:.2f}%".format(c2)), st.plotly_chart(fig2, use_container_width=False,
                                                                                                                                                                                                       sharing="streamlit", theme="streamlit"), st.plotly_chart(fig1, use_container_width=False,

                                                                                                                                                                                                                                                                sharing="streamlit", theme="streamlit")
    mcrALS(file_path, sheet_name)
