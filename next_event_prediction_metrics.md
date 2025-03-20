## Performance of trace-based and integrated SEP models for next event prediction

The table below illustrates that SEP models benefit from incorporating log prefixes when predicting the next activity label and next timestamp. This suggests that the poor performance of integrated SEP models in suffix prediction stems from an inherent limitation of the SEP method (i.e. log prefixes remain static during suffix generation) rather than from noise introduced by the log prefix.

The performance of next activity label prediction is evaluated using precision and recall, with higher values indicating better predictive performance. The performance of next timestamp prediction is assessed using Mean Absolute Error (MAE) and Mean Squared Logarithmic  Error (MSLE), with lower values indicating better predictive performance.

Metrics are **bold** when integrated models outperform trace-based models.


|            | **BPIC2017**    |                 |                 |                 |**BPIC2019**     |                 |                 |                 | **BAC**         |                 |                 |                 |
|------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
|            | Precision       | Recall          | MAE             | MSLE            | Precision       | Recall          | MAE             | MSLE            | Precision       | Recall          | MAE             | MSLE            |
| **SEP-XGBoost** |            |                 |                 |                 |                 |                 |                 |                 |                 |                 |                 |                 |
| Trace-based | 0.8796         | 0.8904          | 588             | 3.85            | 0.7123          | 0.7387          | 10147           | 5.62            | 0.7838          | 0.7682          | 51.80           | 0.52            |
| Integrated  | **0.8816**     | **0.8928**      | **585**         | 3.91            | **0.7882**      | **0.7836**      | 10226           | **4.68**        | **0.7870**       | **0.7847**     | 51.80           | **0.51**        |
| **SEP-LSTM**  |              |                 |                 |                 |                 |                 |                 |                 |                 |                 |                 |                 |
| Trace-based | 0.8809         | 0.8911          | 577             | 3.38            | 0.7161          | 0.7424          | 10330           | 5.55            | 0.7799          | 0.7657          | 52.32           | 0.52            |
| Integrated  | **0.8821**     | **0.8936**      | 587             | **3.21**        | **0.7915**      | **0.7836**      | 10771           | **5.12**        | **0.7875**       | **0.7803**     | 52.74           | 0.54            |
