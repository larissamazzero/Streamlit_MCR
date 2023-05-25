import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pymcr.mcr import McrAR
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="MCR-ALS",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)
image = Image.open('images\wordart.png')
st.image(image, width=400)
# st.image("images\wordart.png", width=400)

# ARGUMENTOS DA FUNÇÃO MCR-ALS
header = st.container()
code = st.container()
content = st.container()

with header:
    st.title("MCR-ALS")
    st.write("""
    ### Insert your .csv or .txt mixture and spectra file! 🤩
    """)
    st.write("Scroll to the end of the page to download datasets, you can also download the graphs by clicking in the superior right corner download option.")
with content:
    mixture_file = st.file_uploader("Insert mixture file", key="mixture_file", type=["csv", "txt"])
    spectra_file = st.file_uploader("Insert pure spectra file", key="spectra_file", type=["csv", "txt"])
    # run_button = st.button("Run")
with code:
    ##################################
    ## This function will verify the file extension and load it
    ##
    ##
    def verify_extension(spectra_file, mixture_file):
        allowed_extensions = ['.csv', '.txt']

        # Verify the extension of the spectras file
        if not any(spectra_file.name.lower().endswith(ext) for ext in allowed_extensions):
            return "Extension for spectra file is not allowed", None, None

        # Verify the extension of the mixtures file
        if not any(mixture_file.name.lower().endswith(ext) for ext in allowed_extensions):
            return "Extension for mixture file is not allowed", None, None

        try:
            # Read spectras file
            if spectra_file.name.lower().endswith(".csv"):
                spectras = pd.read_csv(spectra_file)
            else:
                #spectras = pd.read_fwf(spectra_file, sheet_name="PURE")  # TIRAR SHEET_NAME NO TESTE REAL
                spectras = pd.read_fwf(spectra_file) 

            # Read mixtures file
            if mixture_file.name.lower().endswith(".csv"):
                mixtures = pd.read_csv(mixture_file)
            else:
                #mixtures = pd.read_fwf(mixture_file, sheet_name="MIX")  # TIRAR SHEET_NAME NO TESTE REAL
                mixtures = pd.read_fwf(mixture_file) 

            return None, spectras, mixtures

        except Exception as e:
            return "Error occurred while reading the files: " + str(e), None, None

    ##################################

    ##################################
    ## This function will getting the spectra, mixtures, and informations about the data
    ##
    ##
    def getting_values(spectras, mixtures):

        # Getting the number of rows of each dataset
        num_spec = len(spectras)
        num_mix = len(mixtures)

        # Getting the spectra values
        S = spectras.iloc[0:num_spec, 1:]

        # Getting the mixtures values
        D = mixtures.iloc[0:num_mix, 1:]

        # Getting the spectras and mixtures labels
        spec_labels = spectras['label']
        mix_labels = mixtures['label']              # TIRAR LABEL DA MISTURA

        time = spectras.columns.drop('label')

        return S, D, spec_labels, mix_labels, num_spec, num_mix, time
    ##################################

    ##################################
    ## This function will applyinh the MCR-ALS in the data
    ##
    ##
    def mcrALS(D, S, spec_labels, mix_labels):
        mcrals = McrAR(max_iter=50, tol_increase=0.1)

        errors = []
        concentrations = []
        new_mix = {}
        final_label = []

        # For each mixture
        for _, row in D.iterrows():
            mcrals.fit(row.values.reshape(1, -1), ST=S)    # DA PRA ACEITAR TANTO CONCENTRAÇÃO QUANTO ESPECTRO

            # The residual error
            error = mcrals.err[-1]
            errors.append(error)

            # The Concentration result
            C = mcrals.C_opt_[0].tolist()
            concentrations.append(C)

        # Creating the dataset with each new mixtures
        for i, list_c in enumerate(concentrations):
            for j, c in enumerate(list_c):
                m = c * S.iloc[j, :].values
                label = spec_labels.iloc[j] + ' ' + mix_labels.iloc[i]
                new_mix[label] = m
                final_label.append(label)

        return errors, concentrations, new_mix, final_label
    ##################################

    ##################################
    ## This function will apply the statistics metrics in the data result
    ##
    ##
    def metrics(D, S, concentrations):

        metrics = {}
        cont = 1

        # Calculate the new misture X
        for i, c in enumerate(concentrations):                                                    
            X = np.dot(c, S)

            # Calculate R-squared score
            r2 = r2_score(D.iloc[i,:].T, X.T)

            # Calculate MAE
            mae = mean_absolute_error(D.iloc[i,:].T, X.T)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(D.iloc[i,:].T, X.T, squared=False))
            
            # Appending the metrics
            metrics[i] = {'r2': r2, 'mae': mae, 'rmse': rmse}

        return metrics
    ##################################

    ##################################
    ## This function will generate the dataset with the results
    ##
    ##
    def results(spec_labels, mix_labels, D, new_mix, errors, metrics, concentrations):
        
        dict1 = {}
        columns = ['mix']
        columns.extend(spec_labels.tolist())
        columns.extend(['residual', 'MAE', 'RMSE', 'R2'])
        df_1 = pd.DataFrame()
        df_2 = pd.DataFrame(columns=columns)

        # Adding the labels and the original mixture values in df_1
        for i, label in enumerate(mix_labels.tolist()):
            dict1[label] = D.iloc[i].values.tolist()

        df_1 = pd.DataFrame(dict1)

        # Adding the new mixture values in df_1
        for key, value in new_mix.items():
            df_1[key] = value

        # Adding the label, residual error, concentrations and metrics in df_2
        for i in range(len(mix_labels.tolist())):
            mix_value = mix_labels.tolist()[i]
            residual_value = errors[i]
            mae_value = metrics[i]['mae']
            rmse_value = metrics[i]['rmse']
            r2_value = metrics[i]['r2']
            concentration_values = concentrations[i]
            row_values = [mix_value] + concentration_values + [residual_value, mae_value, rmse_value, r2_value]
            df_2.loc[i] = row_values

        return df_1, df_2
    ##################################

    ##################################
    ## This function will plot the curves
    ##
    ##
    def graphs(spectras, mixtures, concentrations, num_spec, spec_labels, new_mix, time, D):
        data = pd.concat([spectras, mixtures], ignore_index=True)
        keys = list(new_mix.keys())
        data_dict = {}
        fig_list = []

        # Graph 1
        for label in data['label'].unique():
            # Founding the indexes of each row
            indices = data.index[data['label'] == label]
            # Getting the respective row value of each index
            values = data.iloc[indices, 1:].values.tolist()
            # {label: value}
            data_dict[label] = values[0]

        df_3 = pd.DataFrame({'time':time.tolist(), **data_dict})
        spectra_list = list(data_dict.keys())

        fig1 = px.line(df_3, x=df_3['time'], y=spectra_list, width=800, height=700, log_x=True, range_x=[0.01, 10])
        fig1.update_layout(title_x=0.5, title={
                'text': "ILT - CPMG mixtures",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}, yaxis_title='Intensity (a.u.)', xaxis_title='T2 (s)', legend_title='Pure Spectra')
        
        st.plotly_chart(fig1, use_container_width=False, sharing="streamlit", theme="streamlit")

        # Graph 2
        for i in range(len(D)):
            original_mix = D.iloc[i, :].tolist()
            fig2 = px.line(df_3, x=df_3['time'], y=original_mix, width=800, height=700, log_x=True, range_x=[0.01, 10])

            start_index = i * num_spec
            end_index = start_index + num_spec

            c_values = concentrations[i]  
            text_offset = 0  

            for j in range(start_index, end_index):

                key = keys[j]
                m = new_mix[key]

                fig2.add_scatter(x=df_3['time'], y=m, mode='lines', name=f"{key}")

                fig2.update_layout(title_x=0.5, title={
                    'text': "ILT - CPMG spectra",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}, yaxis_title='Intensity (a.u.)', xaxis_title='T2 (s)', legend_title='Spectra')
                
                fig2.add_annotation(text=f"Concentrations: ",
                                    xref="paper", yref="paper",
                                    x=1, y=0.60, showarrow=False)
                
                fig2.add_annotation(text=f"{spec_labels[j - start_index]}: {c_values[j - start_index]*100:0.2f}%",
                                    xref="paper", yref="paper",
                                    x=1, y=0.55 - (0.05 * (j - start_index)) - text_offset, showarrow=False)
                text_offset += 0.02  

            st.plotly_chart(fig2, use_container_width=False, sharing="streamlit", theme="streamlit")
    ##################################

    ##################################
    ## This function will orchestrate the functions
    ##
    ##
    def main(spectra_file, mixture_file):

        # Verify and load file type
        message, spectras, mixtures = verify_extension(spectra_file, mixture_file)

        if message is None:

            # Getting the values and information about spectras and mixtures dataset
            S, D, spec_labels, mix_labels, num_spec, num_mix, time = getting_values(spectras, mixtures)

            # Applying MCR-ALS through the data
            errors, concentrations, new_mix, final_label = mcrALS(D, S, spec_labels, mix_labels)

            # R2, MAE and RMSE
            var_metrics = metrics(D, S, concentrations)

            # Creating a new dataset with the results
            df_new_mix, df_metrics = results(spec_labels, mix_labels, D, new_mix, errors, var_metrics, concentrations)

            # Ploting the curves
            graphs(spectras, mixtures, concentrations, num_spec, spec_labels, new_mix, time, D)

            return df_new_mix, df_metrics, None
        
        else:
            return None, None, message
    ##################################

    if spectra_file is not None and mixture_file is not None:
        df_new_mix, df_metrics, message = main(spectra_file, mixture_file)
        if message is None:
            st.write("Success")
            st.download_button(label='Download new mixture', data=df_new_mix.to_csv(index=False), file_name="new_mix.csv", key="new_mix")
            st.download_button(label='Download metrics', data=df_metrics.to_csv(index=False), file_name="metrics.csv", key="metrics")   
        else:
            st.write(message)
    else:
        st.write("*Both files are needed to run this code!!*")




