import re
import pyperclip

# Assume the table is defined in an r-string named 'table_content' at the top of the file
table_content = r"""\begin{table*}[ht]
\centering
%\scriptsize
\scalebox{0.8}{
\begin{tabular}{c|c|c|cc|c|cccccccc|cc|c}
\hline
\multicolumn{2}{c|}{Methods}&{IMP.}&\multicolumn{2}{c|}{FITS (ours)}&\multicolumn{1}{c|}{FITS*}&\multicolumn{2}{c|}{DLinear*}&\multicolumn{2}{c|}{NLinear*}&\multicolumn{2}{c|}{PatchTST/64 } &\multicolumn{2}{c|}{PatchTST/42}&\multicolumn{2}{c}{\emph{Repeat}*}&\multicolumn{1}{|c}{ARIMA}\\
\hline
\multicolumn{2}{c|}{Metric} & MSE & MSE&MAE & MSE & MSE&MAE & MSE&MAE & MSE&MAE & MSE&MAE & MSE&MAE & MSE\\
\hline
\multirow{4}{*}{\rotatebox{90}{Weather}}
& 96 & 0.014\%
& 0.143 & 0.195
& 0.143
& 0.176 & 0.237 & 0.182 & 0.232 % DLinear, NLinear
& 0.149 & 0.198 & 0.152 & 0.199 % PatchTST 64, 42
& 0.259 & 0.254 % Repeat
& 0.239 \\ % ARIMA
& 192 & 0.014\%
& 0.186 & 0.236
& 0.186
& 0.220 & 0.282 & 0.225 & 0.269 % DLinear, NLinear
& 0.194 & 0.241 & 0.197 & 0.243 % PatchTST 64, 42
& 0.309 & 0.292 \\ % Repeat
& 336 & 0.014\%
& 0.236 & 0.277
& 0.236
& 0.265 & 0.319 & 0.271 & 0.301 % DLinear, NLinear
& 0.245 & 0.282 & 0.249 & 0.283 % PatchTST 64, 42
& 0.377 & 0.338 \\ % Repeat
& 720 & 0.014\%
& 0.308 & 0.330
& 0.307
& 0.323 & 0.362 & 0.338 & 0.348 % DLinear, NLinear
& 0.314 & 0.334 & 0.320 & 0.335 % PatchTST 64, 42
& 0.465 & 0.394 \\ % Repeat
\hline
\multirow{4}{*}{\rotatebox{90}{Traffic}}
& 96 & 0.014\%
& 0.386 & 0.266
& 0.385
& 0.410 & 0.282 & 0.410 & 0.279 % DLinear, NLinear
& 0.360 & 0.249 & 0.367 & 0.251 % PatchTST 64, 42
& 2.723 & 1.079 % Repeat
& 1.448 \\ % ARIMA
& 192 & 0.014\%
& 0.398 & 0.273
& 0.397
& 0.423 & 0.287 & 0.423 & 0.284 % DLinear, NLinear
& 0.379 & 0.256 & 0.385 & 0.259 % PatchTST 64, 42
& 2.756 & 1.087 \\ % Repeat
& 336 & 0.014\%
& 0.413 & 0.277
& 0.410
& 0.436 & 0.296 & 0.435 & 0.290 % DLinear, NLinear
& 0.392 & 0.264 & 0.398 & 0.265 % PatchTST 64, 42
& 2.791 & 1.095 \\ % Repeat
& 720 & 0.014\%
& 0.449 & 0.297
& 0.448
& 0.466 & 0.315 & 0.464 & 0.307 % DLinear, NLinear
& 0.432 & 0.286 & 0.434 & 0.287 % PatchTST 64, 42
& 2.811 & 1.097 \\ % Repeat
\hline
\multirow{4}{*}{\rotatebox{90}{Electricity}}
& 96 & 0.011\%  
& 0.134 & 0.229 
& 0.134
& 0.140 & 0.237 & 0.141 & 0.237 % DLinear, NLinear
& 0.129 & 0.222 & 0.130 & 0.222 % PatchTST 64, 42
& 1.588 & 0.946 \\ % Repeat
& 192 & 0.012\%
& 0.152 & 0.243
& 0.149
& 0.153 & 0.249 & 0.154 & 0.248 % DLinear, NLinear
& 0.147 & 0.240 & 0.148 & 0.240 % PatchTST 64, 42
& 1.595 & 0.950 \\ % Repeat
& 336 & 0.013\%
& 0.165 & 0.256
& 0.165
& 0.169 & 0.267 & 0.171 & 0.265 % DLinear, NLinear
& 0.163 & 0.259 & 0.167 & 0.261 % PatchTST 64, 42
& 1.617 & 0.961 \\ % Repeat
& 720 & 0.014\%
& 0.204 & 0.292
& 0.203
& 0.203 & 0.301 & 0.210 & 0.297 % DLinear, NLinear
& 0.197 & 0.290 & 0.202 & 0.291 % PatchTST 64, 42
& 1.647 & 0.975 \\ % Repeat
\hline
\multirow{4}{*}{\rotatebox{90}{ILI}}
& 24 & 0.014\%
& 2.497 & 0.988
& -
& 2.215 & 1.081 & 1.683 & 0.858 % DLinear, NLinear
& 1.319 & 0.754 & 1.522 & 0.814 % PatchTST 64, 42
& 6.587 & 1.701 \\ % Repeat
& 36 & 0.014\%
& 2.407 & 0.995
& -
& 1.963 & 0.963 & 1.703 & 0.859 % DLinear, NLinear
& 1.579 & 0.870 & 1.430 & 0.834 % PatchTST 64, 42
& 7.130 & 1.884 \\ % Repeat
& 48 & 0.014\%
& 2.472 & 1.038
& -
& 2.130 & 1.024 & 1.719 & 0.884 % DLinear, NLinear
& 1.553 & 0.815 & 1.673 & 0.854 % PatchTST 64, 42
& 6.575 & 1.798 \\ % Repeat
& 60 & 0.014\%
& 2.421 & 1.033
& -
& 2.368 & 1.096 & 1.819 & 0.917 % DLinear, NLinear
& 1.470 & 0.788 & 1.529 & 0.862 % PatchTST 64, 42
& 5.893 & 1.677 \\ % Repeat 
\hline
\multirow{4}{*}{\rotatebox{90}{ETTh1}}
& 96 & 0.014\%
& 0.373 & 0.395
& 0.372
& 0.375 & 0.399 & 0.374 & 0.394 % DLinear, NLinear
& 0.370 & 0.400 & 0.375 & 0.399 % PatchTST 64, 42
& 1.295 &0.713 % Repeat
& 0.839 \\ % ARIMA 
& 192 & 0.014\%
& 0.407 & 0.415
& 0.404
& 0.405 & 0.416 & 0.408 & 0.415 % DLinear, NLinear
& 0.413 & 0.429 & 0.414 & 0.421 % PatchTST 64, 42
& 1.325 & 0.733 \\ % Repeat
& 336 & 0.014\%
& 0.429 & 0.429
& 0.427
& 0.439 & 0.443 & 0.429 & 0.427 % DLinear, NLinear
& 0.422 & 0.440 & 0.431 & 0.436 % PatchTST 64, 42
& 1.323 &0.744 \\ % Repeat
& 720 & 0.014\%
& 0.428 & 0.448
& 0.424
& 0.472 & 0.490 & 0.440 & 0.453 % DLinear, NLinear
& 0.447 & 0.468 & 0.449 & 0.466 % PatchTST 64, 42
& 1.339 &0.756 \\ % Repeat
\hline
\multirow{4}{*}{\rotatebox{90}{ETTh2}}
& 96 & 0.014\%
& 0.270 & 0.322
& 0.271
& 0.289 & 0.353 & 0.277 & 0.338 % DLinear, NLinear
& 0.274 & 0.337 & 0.274 & 0.336 % PatchTST 64, 42
& 0.432 & 0.422 % Repeat
& 0.444 \\ % ARIMA
& 192 & 0.014\%
& 0.341 & 0.365
& 0.331
& 0.383 & 0.418 & 0.344 & 0.381 % DLinear, NLinear
& 0.341 & 0.382 & 0.339 & 0.379 % PatchTST 64, 42
& 0.534 & 0.473 \\ % Repeat
& 336 & 0.014\%
& 0.359 & 0.439
& 0.354
& 0.448 & 0.465 & 0.357 & 0.400 % DLinear, NLinear
& 0.329 & 0.384 & 0.331 & 0.380 % PatchTST 64, 42
& 0.591 & 0.508 \\ % Repeat
& 720 & 0.014\%
& 0.380 & 0.457
& 0.377
& 0.605 & 0.551 & 0.394 & 0.436 % DLinear, NLinear
& 0.379 & 0.422 & 0.379 & 0.422 % PatchTST 64, 42
& 0.588 & 0.517 \\ % Repeat
\hline
\multirow{4}{*}{\rotatebox{90}{ETTm1}}
& 96  & 0.014\%
& 0.308 & 0.345
& 0.303
& 0.299 & 0.343 & 0.306 & 0.348 % DLinear, NLinear
& 0.293 & 0.346 & 0.290 & 0.342 % PatchTST 64, 42
& 1.214 & 0.665 % Repeat
& 0.910 \\ % ARIMA
& 192 & 0.014\%
& 0.337 & 0.363
& 0.337
& 0.335 & 0.365 & 0.349 & 0.375 % DLinear, NLinear
& 0.333 & 0.370 & 0.332 & 0.369 % PatchTST 64, 42
& 1.261 & 0.690 \\ % Repeat
& 336 & 0.014\%
& 0.366 & 0.381
& 0.366
& 0.369 & 0.386 & 0.375 & 0.388 % DLinear, NLinear
& 0.369 & 0.392 & 0.366 & 0.392 % PatchTST 64, 42
& 1.283 & 0.707 \\ % Repeat
& 720 & 0.014\%
& 0.415 & 0.408
& 0.415
& 0.425 & 0.421 & 0.433 & 0.422 % DLinear, NLinear
& 0.416 & 0.420 & 0.420 & 0.424 % PatchTST 64, 42
& 1.319 & 0.729 \\ % Repeat
\hline
\multirow{4}{*}{\rotatebox{90}{ETTm2}} 
& 96  & 0.014\%
& 0.162 & 0.242
& 0.162
& 0.167 & 0.262 & 0.167 & 0.255 % DLinear, NLinear
& 0.166 & 0.260 & 0.165 & 0.255 % PatchTST 64, 42
& 0.266 & 0.328 % Repeat
& 0.224 \\ % ARIMA
& 192 & 0.014\%
& 0.216 & 0.276
& 0.216
& 0.224 & 0.303 & 0.221 & 0.293 % DLinear, NLinear
& 0.223 & 0.303 & 0.220 & 0.292 % PatchTST 64, 42
& 0.340 & 0.371 \\ % Repeat
& 336 & 0.014\%
& 0.268 & 0.309
& 0.268
& 0.281 & 0.342 & 0.274 & 0.327 % DLinear, NLinear
& 0.274 & 0.342 & 0.278 & 0.329 % PatchTST 64, 42
& 0.412 & 0.410 \\ % Repeat
& 720 & 0.014\%
& 0.349 & 0.360
& 0.348
& 0.397 & 0.421 & 0.368 & 0.384 % DLinear, NLinear
& 0.362 & 0.421 & 0.367 & 0.385 % PatchTST 64, 42
& 0.521 &0.465 \\ % Repeat
\hline
\multirow{4}{*}{\rotatebox{90}{Exchange}}
& 96 & 0.014\%
& 0.087 & 0.204
& -
& 0.081 & 0.203 & 0.089 & 0.208 % DLinear, NLinear
& - & - & - & - % PatchTST 64, 42
& 0.081 & 0.196 \\ % Repeat
& 192 & 0.014\%
& 0.180 & 0.295
& -
& 0.157 & 0.293 & 0.180 & 0.300 % DLinear, NLinear
& - & - & - & - % PatchTST 64, 42
& 0.167 & 0.289 \\ % Repeat
& 336 & 0.014\%
& 0.333 & 0.405 
& -
& 0.305 & 0.414 & 0.331 & 0.415 % DLinear, NLinear
& - & - & - & - % PatchTST 64, 42
& 0.305 & 0.396 \\ % Repeat
& 720 & 0.014\%
& 0.951 & 0.713
& -
& 0.643 & 0.601 & 1.033 & 0.780 % DLinear, NLinear
& - & - & - & - % PatchTST 64, 42
& 0.823 & 0.681 \\ % Repeat
\hline
\end{tabular}
}
\begin{tablenotes}
    \scriptsize
    {
    \item *Results borrowed from \cite{xu2024fits}, \cite{DLinear}, and \cite{PatchTST}.
    }
\end{tablenotes}
\vspace{-0.2cm}
\caption{Multivariate long-term forecasting errors in terms of MSE and MAE, the lower the better. Note that the ILI dataset has prediction length $T \in \{24,36,48,60\}$ whereas the others have prediction length $T \in \{96,192,336,720\}$. \emph{Repeat} repeats the last value in the look-back window and serves as our baseline. The best results are highlighted in bold and the second best results are highlighted with underline. IMP denotes the relative improvement or downgrade from FITS (ours) and the second best or best model.}
\vspace{-0.5cm}
\label{tab:gigatable}
\end{table*}
"""

def find_lowest_values(row):
    values = re.findall(r'(\d+\.\d+)', row)
    values = [float(v) for v in values]
    if not values:
        return None, None
    mse_values = values[::2]
    mae_values = values[1::2]
    lowest_mse = min(mse_values) if mse_values else None
    lowest_mae = min(mae_values) if mae_values else None
    return lowest_mse, lowest_mae

def highlight_lowest_values(row):
    lowest_mse, lowest_mae = find_lowest_values(row)
    if lowest_mse is not None:
        row = re.sub(f'({lowest_mse:.3f})', r'\\textbf{\1}', row)
    if lowest_mae is not None:
        row = re.sub(f'({lowest_mae:.3f})', r'\\textbf{\1}', row)
    return row

# Split the table into rows
rows = table_content.split('\n')

# Process each row
for i in range(len(rows)):
    if any(dataset in rows[i] for dataset in ['Weather', 'Traffic', 'Electricity', 'ILI', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange']):
        rows[i] = highlight_lowest_values(rows[i])

# Join the rows back together
output_table = '\n'.join(rows)

# Print the result
print(output_table)

# Copy the result to clipboard
pyperclip.copy(output_table)

print("\nThe processed table has been copied to your clipboard.")