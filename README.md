# Berkeley Capstone Project - Time Series forecasting by Deep Learning
Supervisior: Zeyu Zheng, zyzheng@berkeley.edu. \
Advisor: Yunkai Zhang, yunkai_zhang@berkeley.edu. \
This repository applies time series models to the [M5 dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

# M5 Data
- v0: Predict the sales for each product category in each store.
Meta variables: state_id, store_id, product_category_id.
Given variables: time_from_start, snap_accepted, is_sporting_event, is_cultural_event, is_national_event, is_religious_event.

## Models
- **Linear:** We process all the meta variables, the given variables, the target variables (only in the context range),
and the time covariates, and concatenate them together. We feed the concatenated vector into a linear layer to predict 
the target values in the forecast range.
- **Enc_Only_Transformer:** The variables at each time step is treated as one token. The variables
are the meta variables, the given variables, the target variables, and the time covariates.
The encoder only takes in values in the context range. We take the output embeddings from the encoder,
flatten them, and feed them into a linear layer to predict the target values in the forecast range.
- **DeepAR:** Predict one step ahead each time, and feed back into the LSTM to predict the next step.
- **Enc_Dec_Transformer:** We treat the variables at each time step as one token,
similarly as the Enc_Only_Transformer. The encoder takes in values in the context range, and the decoder
takes in the values in the forecast range, except that for the target variables (which we don't have access to 
in the forecast range), for which we pad with zeros. Instead of flattening, we apply a linear layer 
independently to each output token in the decoder to get the prediction for the target variable at
the corresponding time step.

## Acknowledgements
- Time Series Library: https://github.com/thuml/Time-Series-Library

## Visuallized results: Wandb link
[Wandb](https://wandb.ai/zhimingfan-university-of-california-berkeley/capstone?nw=nwuserzhimingfan).
